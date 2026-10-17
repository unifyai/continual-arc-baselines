"""Confine a system's process tree so the benchmark cannot leak into it.

Systems execute model-written code. Left alone, that code can read the benchmark
source, import the Re-ARC generators and verifiers from the very environment that
launched it, or reach the internet. A run under prime-agent did
exactly that on 2026-09-17: it located ``continual_arc`` in the baselines venv and
called the task verifier to compute ground truth before submitting.

The rule is therefore: a system sees only what it needs, and the tracking proxy is
its only way out.

* read: the system's own installation, its Python runtime, the OS;
* read and write: its private state under the attempt directory, a private HOME
  and a private temporary directory;
* network: TCP to the proxy on the loopback interface, and UNIX sockets under its
  private temporary directory (for anything the system spawns for itself);
* nothing else. In particular no other file under the user's home directory, so
  neither the repositories nor the virtual environments are reachable.

Backends: ``seatbelt`` (macOS ``sandbox-exec``) is implemented; it wraps the
command in place and applies to every descendant process. On other platforms a
backend has to be added (bubblewrap on Linux, or a container per system) before
results can be trusted, so :func:`resolve_backend` refuses to fall back silently:
``sandbox="none"`` has to be spelled out and is recorded in the run parameters.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SEATBELT = "/usr/bin/sandbox-exec"
BACKENDS = ("auto", "seatbelt", "bwrap", "none")

# Operating-system paths any process needs; nothing under /Users or /home.
_SYSTEM_READ = (
    "/usr", "/bin", "/sbin", "/System", "/Library", "/private/etc", "/etc",
    "/private/var/db", "/private/var/select", "/var/db", "/var/select", "/dev",
    "/Applications/Xcode.app/Contents/Developer/usr",  # harmless if absent
)


class SandboxUnavailable(RuntimeError):
    """No sandbox backend for this platform, and none was explicitly waived."""


@dataclass
class SandboxSpec:
    """What a confined process tree may touch. All paths are made absolute and
    symlink-free, because the sandbox checks the resolved path."""

    read_paths: list[Path] = field(default_factory=list)
    write_paths: list[Path] = field(default_factory=list)
    tmp_dir: Path | None = None
    home: Path | None = None
    mask_paths: list[Path] = field(default_factory=list)
    """Paths hidden from the system (bubblewrap only; seatbelt denies by default).
    The benchmark source, its data with the generators and verifiers, and the
    virtual environments that hold them."""
    loopback_ports: list[int] = field(default_factory=list)
    """Loopback TCP ports the system may connect to (the proxy)."""
    listen_ports: list[int] = field(default_factory=list)
    """Loopback TCP ports the system may listen on and connect to (a gateway it
    runs for itself, such as openclaw's)."""
    unix_socket_dirs: list[Path] = field(default_factory=list)

    def normalised(self) -> SandboxSpec:
        def fix(paths):
            out = []
            for p in paths:
                p = Path(p).expanduser()
                if p.exists():
                    out.append(p.resolve())
            return out

        return SandboxSpec(
            read_paths=fix(self.read_paths), write_paths=fix(self.write_paths),
            mask_paths=fix(self.mask_paths),
            tmp_dir=self.tmp_dir.resolve() if self.tmp_dir else None,
            home=self.home.resolve() if self.home else None,
            loopback_ports=list(self.loopback_ports),
            listen_ports=list(self.listen_ports),
            unix_socket_dirs=fix(self.unix_socket_dirs),
        )


def resolve_backend(requested: str = "auto") -> str:
    """Pick the sandbox backend: seatbelt on macOS, bubblewrap on Linux, never a
    silent fallback to none."""
    if requested not in BACKENDS:
        raise ValueError(f"sandbox must be one of {BACKENDS}, got {requested!r}")
    if requested == "none":
        return "none"
    system = platform.system()
    if requested == "seatbelt" or (requested == "auto" and system == "Darwin"):
        if system == "Darwin" and os.path.exists(SEATBELT):
            return "seatbelt"
        raise SandboxUnavailable("seatbelt (sandbox-exec) is only available on macOS")
    if requested == "bwrap" or (requested == "auto" and system == "Linux"):
        if shutil.which("bwrap"):
            return "bwrap"
        raise SandboxUnavailable("bubblewrap (bwrap) is not installed; apt install bubblewrap, "
                                 "or pass sandbox=\"none\" explicitly (recorded in the run)")
    raise SandboxUnavailable(f"no sandbox backend for {system}; pass sandbox=\"none\" explicitly")


def default_mask_paths() -> list[Path]:
    """What a system must not be able to read on a Linux node: the benchmark and
    baselines source, the pinned data (generators and verifiers), and every
    virtual environment that has the benchmark installed."""
    home = Path.home()
    candidates = [
        home / "continual-arc", home / "continual-arc-baselines" / "src",
        home / "continual-arc-baselines" / "envs", home / "continual-arc-baselines" / ".venv",
        home / ".cache" / "continual-arc", home / "prime-rl", home / "continual-arc-paper",
        # every other cell's state, transcripts, records and schedules (23 Sep: a harness walked
        # the whole tree searching for its task id); a cell's own state is bound back in below
        home / "continual-arc-baselines" / ".workbench",
    ]
    return [p for p in candidates if p.exists()]


NETBRIDGE = r'''"""Forward between a loopback TCP port and a UNIX socket (stdlib only).

Two roles, one script: on the host, ``unix2tcp SOCK PORT`` accepts connections on the
UNIX socket and relays them to the tracking proxy on 127.0.0.1:PORT; inside a sandbox
with a private network namespace, ``tcp2unix PORT SOCK`` listens on 127.0.0.1:PORT and
relays to the UNIX socket, which crosses the namespace boundary as a file. With
``--`` and a command, ``tcp2unix`` starts the relay in the background and execs the
command, so the system finds the proxy at the same address it would on the host.
"""
import os
import socket
import sys
import threading


def _pump(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _serve(listener, connect):
    while True:
        try:
            client, _ = listener.accept()
        except OSError:
            return
        try:
            upstream = connect()
        except OSError:
            client.close()
            continue
        threading.Thread(target=_pump, args=(client, upstream), daemon=True).start()
        threading.Thread(target=_pump, args=(upstream, client), daemon=True).start()


def unix2tcp(sock_path, port):
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(sock_path)
    os.chmod(sock_path, 0o600)
    listener.listen(64)

    def connect():
        return socket.create_connection(("127.0.0.1", port))

    _serve(listener, connect)


def tcp2unix(port, sock_path):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(64)

    def connect():
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(sock_path)
        return s

    _serve(listener, connect)


def main(argv):
    role = argv[0]
    if role == "unix2tcp":
        unix2tcp(argv[1], int(argv[2]))
        return 0
    if role == "tcp2unix":
        tcp2unix(int(argv[1]), argv[2])
        return 0
    raise SystemExit(f"unknown role {role}")


if __name__ == "__main__":
    # tcp2unix with a command: keep a relay process alive alongside the command.
    if len(sys.argv) > 4 and sys.argv[1] == "tcp2unix" and sys.argv[4] == "--":
        port, sock_path, cmd = int(sys.argv[2]), sys.argv[3], sys.argv[5:]
        if os.fork() == 0:
            tcp2unix(port, sock_path)
            os._exit(0)
        os.execvp(cmd[0], cmd)
    sys.exit(main(sys.argv[1:]))
'''


def proxy_port(url: str) -> int | None:
    """The loopback port of the tracking proxy in ``url``, or None for a remote URL."""
    from urllib.parse import urlparse
    u = urlparse(url)
    if u.hostname in ("127.0.0.1", "localhost", "::1") and u.port:
        return int(u.port)
    return None


def start_bridge(argv: list[str], port: int | None, sock_path: Path, script_path: Path):
    """For a private network namespace: write the forwarder, start its host half
    (UNIX socket -> proxy port) and return ``(argv wrapped to start the sandbox
    half, host process)`` once the host half's socket exists, since the sandbox
    binds that file in as it starts. With no loopback proxy there is nothing to bridge.

    A socket file left at ``sock_path`` by an earlier bridge is removed first: a bridge that is ended leaves its file,
    and an adapter may start a bridge for every session in the same private directory. The wait found that file
    at once, so bubblewrap started while the new bridge was removing it and binding its own, and failed with "Can't bind
    mount .../proxy.sock ... No such file or directory" or "Unable to find .../proxy.sock (deleted) in mount table"
    (27-28 Sep 2026: 12 of 1,722 sessions in one harness's persistent cells, none in its stateless cells, whose directory
    is new for every instance)."""
    if port is None:
        return list(argv), None
    script_path.write_text(NETBRIDGE)
    sock_path.unlink(missing_ok=True)
    host = subprocess.Popen([sys.executable, str(script_path), "unix2tcp", str(sock_path), str(port)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 10
    while not sock_path.exists() and host.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if not sock_path.exists():
        host.kill()
        raise RuntimeError(f"network bridge did not come up on {sock_path} (UNIX socket paths must be "
                           "short; keep it under the run's /tmp directory)")
    inner = ["/usr/bin/python3", str(script_path), "tcp2unix", str(port), str(sock_path), "--", *argv]
    return inner, host


# Trees a Linux sandbox hides whole before binding back what the system needs: every home directory (the
# repositories, their virtual environments and secrets, every run's state and transcripts, caches, anything an
# operator leaves there), a container's volume, and the host's temporary directory (other systems' private
# directories and sockets).
HIDDEN_ROOTS = ("/home", "/root", "/workspace", "/mnt", "/tmp")


def hidden_roots() -> list[Path]:
    roots = [Path(p) for p in HIDDEN_ROOTS]
    home = Path.home()
    if not any(home == r or r in home.parents for r in roots):
        roots.append(home)  # a home outside them
    return [r for r in roots if r.exists()]


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _links_to(path: str | Path) -> tuple[list[tuple[str, Path]], Path]:
    """The symlinks a lookup of ``path`` passes through, as ``(target as written, link path)`` in the order it meets
    them, and the real path it ends at. Recreated inside the sandbox as links, they keep what a program learns about
    itself through them as it is on the host: a launcher in ``~/.local/bin`` that finds its release through
    ``/proc/self/exe``, a venv interpreter that finds its runtime (25 Sep: bound in place of its link, prime-agent
    looked for its package.json in ``~/.local/bin`` and would not start)."""
    links: list[tuple[str, Path]] = []
    pending = list(Path(os.path.abspath(Path(path).expanduser())).parts[1:])
    current = Path("/")
    while pending and len(links) < 40:
        current = current / pending.pop(0)
        if os.path.islink(current):
            target = os.readlink(current)
            links.append((target, current))
            # the rest of the lookup continues from the target (a relative one from the link's directory, which is
            # already free of links)
            pending = list(Path(os.path.normpath(current.parent / target)).parts[1:]) + pending
            current = Path("/")
    return links, current


def bwrap_command(argv: list[str], spec: SandboxSpec, share_net: bool = False) -> list[str]:
    """Bubblewrap, as an allowlist like the seatbelt profile. The operating system is readable, every home directory,
    container volume and the host's temporary directory are hidden whole (:func:`hidden_roots`), and only what the
    system is declared to need is bound back: its read paths and any existing absolute path its command line names
    (its executable, the bridge script) read-only, then its private state, HOME and temporary directory writable. The
    benchmark trees stay hidden wherever they lie, even under something bound back. Private network and pid
    namespaces: loopback only (bubblewrap brings ``lo`` up), so the system reaches the tracking proxy through the
    bridge of :func:`start_bridge` and nothing else. ``share_net`` is for a system's own maintenance commands that
    must reach a gateway it runs (never for a model-driven turn).

    25 Sep 2026: this was the whole filesystem with a list of benchmark trees hidden, and a harness under it read
    other runs' transcripts in ``~/runs``, which the list did not name; the operator's ``.env`` was readable too."""
    raw = spec
    spec = spec.normalised()
    hidden = hidden_roots()
    covered = hidden + [m for m in spec.mask_paths if not any(_under(m, h) for h in hidden)]
    cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
           "--die-with-parent", "--unshare-pid", "--share-net" if share_net else "--unshare-net"]
    for p in covered:
        cmd += ["--tmpfs", str(p)]
    hidden_here = lambda p: any(_under(p, c) for c in covered)  # noqa: E731
    reals: set[Path] = set()
    links: dict[Path, str] = {}
    for path in [*raw.read_paths, *(a for a in argv if a.startswith("/"))]:
        chain, real = _links_to(path)
        if not real.exists():
            continue
        if hidden_here(real):
            reals.add(real)
        links.update({link: target for target, link in chain if hidden_here(link)})
    ro: list[Path] = []
    for real in sorted(reals, key=lambda p: (len(p.parts), str(p))):
        # parents first; a path inside a directory already bound back is visible through it
        if not any(d.is_dir() and _under(real, d) for d in ro):
            ro.append(real)
    for real in ro:
        cmd += ["--ro-bind", str(real), str(real)]
    # a link inside a directory bound back is already there, as on the host
    for link, target in sorted(links.items(), key=lambda kv: (len(kv[0].parts), str(kv[0]))):
        if not any(d.is_dir() and _under(link, d) and link != d for d in ro):
            cmd += ["--symlink", target, str(link)]
    rw = list(spec.write_paths) + [p for p in (spec.tmp_dir, spec.home) if p]
    for p in rw:
        cmd += ["--bind", str(p), str(p)]
    for m in spec.mask_paths:
        if any(_under(m, d) and m != d for d in [*ro, *rw]):
            cmd += ["--tmpfs", str(m)]
    return cmd + ["--", *argv]


def _sb_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def seatbelt_profile(spec: SandboxSpec) -> str:
    """Render a Seatbelt profile: deny by default, then the allowances in ``spec``."""
    spec = spec.normalised()
    rw = list(spec.write_paths)
    if spec.tmp_dir:
        rw.append(spec.tmp_dir)
    if spec.home:
        rw.append(spec.home)
    socket_dirs = list(spec.unix_socket_dirs) + ([spec.tmp_dir] if spec.tmp_dir else [])
    reads = [Path(p) for p in _SYSTEM_READ if Path(p).exists()] + spec.read_paths + rw

    lines = [
        "(version 1)",
        "(deny default)",
        ";; process bookkeeping every runtime needs; none of it reaches user data",
        "(allow process-exec process-fork process-info*)",
        "(allow signal (target same-sandbox))",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow ipc-posix*)",
        "(allow file-ioctl)",
        "(allow pseudo-tty)",
        "(allow file-map-executable)",
        ";; path resolution needs stat() on parents; this leaks names, never contents",
        "(allow file-read-metadata)",
        ";; readable: the operating system, the system's own install and runtime, its state",
        '(allow file-read* (literal "/"))',  # dyld reads the root directory; every exec aborts without it
    ]
    for p in reads:
        lines.append(f"(allow file-read* (subpath {_sb_str(str(p))}))")
    lines.append(";; writable: only the system's private state, HOME and temporary directory")
    for p in rw:
        lines.append(f"(allow file-write* (subpath {_sb_str(str(p))}))")
    lines.append("(allow file-write* (literal \"/dev/null\") (literal \"/dev/tty\") (regex #\"^/dev/tty[a-z0-9]+$\"))")
    lines.append(";; network: the tracking proxy on loopback, and private UNIX sockets")
    for port in spec.loopback_ports:
        lines.append(f'(allow network-outbound (remote ip "localhost:{int(port)}"))')
    for port in spec.listen_ports:
        lines.append(f'(allow network-bind (local ip "localhost:{int(port)}"))')
        lines.append(f'(allow network-inbound (local ip "localhost:{int(port)}"))')
        lines.append(f'(allow network-outbound (remote ip "localhost:{int(port)}"))')
    for d in socket_dirs:
        s = _sb_str(str(d))
        lines.append(f"(allow network-outbound (remote unix-socket (subpath {s})))")
        lines.append(f"(allow network-bind (local unix-socket (subpath {s})))")
        lines.append(f"(allow network-inbound (local unix-socket (subpath {s})))")
    return "\n".join(lines) + "\n"


def wrap_command(backend: str, argv: list[str], spec: SandboxSpec, profile_path: Path,
                 share_net: bool = False) -> list[str]:
    """Return the command line that runs ``argv`` inside the sandbox."""
    if backend == "none":
        return list(argv)
    if backend == "seatbelt":
        profile_path.write_text(seatbelt_profile(spec))
        return [SEATBELT, "-f", str(profile_path), *argv]
    if backend == "bwrap":
        return bwrap_command(argv, spec, share_net=share_net)
    raise ValueError(f"unknown sandbox backend {backend!r}")


def python_runtime_root(python: Path) -> Path | None:
    """The installation a venv's interpreter comes from (``home`` in pyvenv.cfg)."""
    cfg = Path(python).absolute().parent.parent / "pyvenv.cfg"  # the venv, not the resolved binary
    if not cfg.is_file():
        return None
    for line in cfg.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "home" and value.strip():
            home = Path(value.strip())
            return home.parent if home.name == "bin" else home
    return None


def darwin_user_dirs() -> list[Path]:
    """The per-user temporary and cache directories macOS libraries use regardless
    of ``TMPDIR`` (``/var/folders/.../T`` and ``.../C``). Empty elsewhere."""
    if platform.system() != "Darwin":
        return []
    out = []
    for name in ("DARWIN_USER_TEMP_DIR", "DARWIN_USER_CACHE_DIR"):
        try:
            value = subprocess.run(["getconf", name], capture_output=True, text=True,
                                   timeout=10, check=False).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            continue
        if value and Path(value).is_dir():
            out.append(Path(value))
    return out


def package_manager_roots(path: Path) -> list[Path]:
    """Extra read roots an executable under a package manager prefix needs.

    Homebrew binaries link against libraries in ``<prefix>/opt/*`` and
    ``<prefix>/Cellar/*`` rather than next to themselves, so a formula's own
    directory is not enough to run it."""
    resolved = Path(path).resolve()
    out: list[Path] = []
    for prefix in (Path("/opt/homebrew"), Path("/usr/local"), Path("/home/linuxbrew/.linuxbrew")):
        if prefix in resolved.parents:
            for sub in ("opt", "Cellar", "lib", "bin", "etc", "share"):
                if (prefix / sub).exists():
                    out.append(prefix / sub)
    return out


def install_root(executable: str) -> Path | None:
    """The directory tree an executable lives in, symlinks resolved.

    ``~/.local/bin/prime-agent -> ~/.local/share/prime-agent/bin/prime-agent`` gives
    ``~/.local/share/prime-agent`` (which also holds the release the launcher runs).
    """
    found = shutil.which(executable) if os.path.sep not in executable else executable
    if not found:
        return None
    real = Path(found).resolve()
    parent = real.parent
    return parent.parent if parent.name == "bin" else parent
