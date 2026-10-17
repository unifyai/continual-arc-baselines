"""The Linux sandbox (systems/sandbox.py, bubblewrap) as an allowlist: every home directory and the host's temporary
directory hidden whole, and only what the system is declared to need bound back (25 Sep 2026: under the old
denylist a harness read other runs' transcripts in ~/runs)."""

import os
import platform
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from continual_arc_baselines.systems import sandbox
from continual_arc_baselines.systems.sandbox import SandboxSpec, bwrap_command, start_bridge


def _pairs(cmd, flag):
    return [(cmd[i + 1], cmd[i + 2]) for i, arg in enumerate(cmd[:-2]) if arg == flag]


def test_the_command_hides_the_roots_and_binds_back_only_the_declared_paths(tmp_path, monkeypatch):
    home, tmp = tmp_path / "home", tmp_path / "tmp"
    runtime = home / "u" / ".local" / "share" / "python-3.11.16"
    (runtime / "bin").mkdir(parents=True)
    minor = home / "u" / ".local" / "share" / "python-3.11"
    minor.symlink_to(runtime)  # the minor-version link a venv's interpreter goes through
    install = home / "u" / ".local" / "share" / "agent"
    (install / "bin").mkdir(parents=True)
    (install / "bin" / "agent").write_text("#!/bin/sh\n")
    launcher = home / "u" / ".local" / "bin" / "agent"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(install / "bin" / "agent")
    state = home / "u" / "runs" / "r1" / "users" / "u000" / "system" / "state"
    state.mkdir(parents=True)
    bridge = state.parent / "netbridge.py"
    bridge.write_text("")
    bench = home / "u" / "bench"
    (bench / "src").mkdir(parents=True)
    private_tmp = tmp / "arc-pa-1"
    private_tmp.mkdir(parents=True)
    monkeypatch.setattr(sandbox, "hidden_roots", lambda: [home, tmp])
    spec = SandboxSpec(read_paths=[install, minor, bench], write_paths=[state], tmp_dir=private_tmp,
                       mask_paths=[bench / "src"])
    argv = [str(launcher), "--cwd", str(state), str(bridge)]
    cmd = bwrap_command(argv, spec)
    tmpfs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--tmpfs"]
    assert tmpfs[:2] == [str(home), str(tmp)]  # hidden whole, first
    ro, rw, links = _pairs(cmd, "--ro-bind"), _pairs(cmd, "--bind"), _pairs(cmd, "--symlink")
    assert (str(install), str(install)) in ro and (str(runtime), str(runtime)) in ro and (str(bench), str(bench)) in ro
    assert (str(bridge), str(bridge)) in ro  # named on the command line
    # links stay links, so a program that finds itself through one sees what it sees on the host
    assert (str(runtime), str(minor)) in links and (str(install / "bin" / "agent"), str(launcher)) in links
    assert rw == [(str(state), str(state)), (str(private_tmp), str(private_tmp))]
    # writable paths come after every read-only bind and link, so a writable path wins where they meet
    assert cmd.index("--bind") > max(i for i, a in enumerate(cmd) if a in ("--ro-bind", "--symlink"))
    # a benchmark tree under something bound back is hidden again, last
    assert tmpfs[-1] == str(bench / "src") and cmd.index(str(bench / "src")) > cmd.index("--bind")
    # nothing else under the hidden roots comes back: no other run, no parent directory
    bound = {dest for _, dest in ro + rw + links}
    assert not any(str(b) in bound for b in (home / "u", home / "u" / "runs", state.parent))
    assert cmd[cmd.index("--") + 1:] == argv


@pytest.mark.skipif(platform.system() != "Linux" or shutil.which("bwrap") is None, reason="bubblewrap on Linux only")
def test_inside_the_sandbox_only_the_declared_paths_exist(tmp_path):
    secret = tmp_path / "other-run" / "turns.jsonl"  # under /tmp: another system's files
    secret.parent.mkdir()
    secret.write_text("the answer")
    state = tmp_path / "state"
    state.mkdir()
    readable = tmp_path / "install"
    readable.mkdir()
    (readable / "tool.txt").write_text("tool\n")
    spec = SandboxSpec(read_paths=[readable], write_paths=[state])
    script = (f"test ! -e {secret} && echo hidden; cat {readable}/tool.txt; echo ok > {state}/written && echo wrote; "
              f"ls -A {Path.home()} | wc -l; ls -A /tmp | wc -l")
    out = subprocess.run(bwrap_command(["/bin/sh", "-c", script], spec), capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.split()
    assert lines[:3] == ["hidden", "tool", "wrote"]
    assert (state / "written").read_text() == "ok\n"
    # the home directory holds only what was bound back into it (nothing, unless tmp_path lies under it)
    assert int(lines[3]) <= 1 and int(lines[4]) <= 1


def test_a_bridge_returns_once_its_own_socket_exists_not_one_an_earlier_bridge_left(tmp_path):
    """An adapter may start a bridge for every session in the same private directory, and an ended bridge
    leaves its socket file. start_bridge found that file at once and bubblewrap then raced the new bridge, which
    removes it before binding its own (27-28 Sep: "bwrap: Can't bind mount .../proxy.sock ... No such file
    or directory"). When start_bridge returns, the file must be the new bridge's socket, and it must reach the proxy."""
    proxy = socket.create_server(("127.0.0.1", 0))  # stands in for the tracking proxy
    short = Path(tempfile.mkdtemp(prefix="arc-t-", dir="/tmp"))  # UNIX socket paths must be short
    sock_path = short / "proxy.sock"
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(sock_path))  # what an ended bridge leaves: a socket file nothing listens on; held open, so the new
    stale_inode = os.stat(sock_path).st_ino  # socket cannot be given the same inode number
    host = None
    try:
        _, host = start_bridge(["true"], proxy.getsockname()[1], sock_path, tmp_path / "netbridge.py")
        assert os.stat(sock_path).st_ino != stale_inode
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        deadline = time.monotonic() + 5
        while True:  # the file exists once the bridge has bound it; it listens a moment later
            try:
                client.connect(str(sock_path))
                break
            except ConnectionRefusedError:
                assert time.monotonic() < deadline
                time.sleep(0.01)
        proxy.settimeout(5)
        upstream, _ = proxy.accept()
        client.sendall(b"ping")
        upstream.settimeout(5)
        assert upstream.recv(4) == b"ping"
        client.close()
        upstream.close()
    finally:
        if host is not None:
            host.kill()
            host.wait(timeout=10)
        stale.close()
        proxy.close()
        shutil.rmtree(short, ignore_errors=True)
