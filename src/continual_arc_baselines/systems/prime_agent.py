"""prime-agent (Prime Intellect's self-improving RLM harness) as a learner.

One ``prime-agent --mode rpc`` process lives for the whole run (or until the
schedule wipes state). Its persistent IPython kernel, memories, skills and
sub-agent specs are the continual-learning method under test; the adapter only
talks to it over the documented RPC protocol on stdin/stdout.

Model calls go through the proxy: a ``models.json`` in the run's private agent
directory defines a provider whose ``baseUrl`` is the proxy, so prime-agent
never sees the upstream credential. ``--thinking`` sets the reasoning effort the
harness asks for (``xhigh`` by default).

Confinement. The whole prime-agent tree (client, per-run daemon, workers, kernel)
runs inside a sandbox (:mod:`continual_arc_baselines.systems.sandbox`): it can
read its own installation and runtime, read and write only its private state
under the attempt directory plus a private HOME and TMPDIR, and open TCP only to
the proxy. Its environment is built from an allowlist, not inherited.

Liveness. One benchmark action is one prime-agent turn, and a turn is an
unbounded loop of model calls and Python cells. Four guards bound it:

* ``quiet_probe_s``: prime-agent 0.9.5 runs the agent in a daemon worker and
  relays its events to the RPC client through a supervisor; that relay has been
  observed to drop the tail of a turn (``turn_end``/``agent_end``, sometimes
  ``message_end`` too) while the command path keeps working. When the event
  stream is quiet for this long the worker is asked directly (``get_state``);
  if it is no longer streaming, the reply is read with
  ``get_last_assistant_text``. Two consecutive losses mean the attachment is
  dead and prime-agent is restarted before the next prompt.
* ``idle_timeout_s``: no RPC event and no byte through the proxy for this long
  means the system is hung; the turn is aborted and the system restarted.
* ``final_turn_grace_s``: if a turn ends on a final text reply and nothing
  follows, that reply is taken after this grace period instead of waiting.
* ``action_budget_s`` / ``max_model_calls``: a soft budget per action. When it
  is exceeded the turn is aborted and the system is asked once to decide from
  what it has (``reprompt`` budgets apply); an unusable answer is a give-up.

``timeout_s`` remains the hard ceiling for one turn. Both clocks leave out the seconds
the proxy credits (``Tracker.credit_s``): time its calls spent on work that is not the
model's reply, which only a row whose proxy answers some calls itself has, so that such a row's turns are held
to the same budget of model
time as any other row's.

Requires the ``prime-agent`` CLI on ``PATH``:
``curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh``.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from continual_arc_baselines.systems.base import SystemLearner, SystemUnavailable
from continual_arc_baselines.systems.protocol import BUDGET_NOTICE
from continual_arc_baselines.systems.sandbox import (
    SandboxSpec,
    darwin_user_dirs,
    default_mask_paths,
    install_root,
    proxy_port,
    python_runtime_root,
    resolve_backend,
    start_bridge,
    wrap_command,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "IFM/K2-Horizon-7B"
OPENROUTER = "https://openrouter.ai/api/v1"
KEY_VARIABLE = "ARC_LLM_API_KEY"
REFINE_PROMPT = (
    "This piece of work is finished. Run `await refine.run(global_=True)` in your Python REPL now, "
    "so that whatever you learned here that will help with future work is refined into your global "
    "harness store (memories, prompt notes, skills), then reply with the single word: done."
)
# The instructions of a refinement the adapter starts itself (refine_via "rpc"): the verdict, which no closing turn has
# delivered, and what to keep. prime-agent puts them in the refinement prompt as <user_refine_instructions>, beside the
# conversation (its last 80,000 characters) and the current harness state.
REFINE_INSTRUCTIONS = (
    "{verdict}\n\nThis piece of work is finished. Refine whatever you learned in it that will help with future work "
    "into the global harness store (memories, prompt notes, skills)."
)
PRESERVED = ("models.json", "settings.json", "kernel-venv")  # survive a state wipe
INTEGRATION_SKILLS = ("linear", "notion", "prime-intellect")  # MCP integrations; off
INHERITED_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ")
PROXY_ENV = ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_KEY", "OPENROUTER_BASE_URL",
             "OPENROUTER_API_KEY", "ARC_LLM_BASE_URL")
ACTIVITY = ("agent_start", "turn_start", "message_start")  # a new run or turn is underway


def _last_role(event: dict) -> str | None:
    messages = [m for m in event.get("messages") or [] if isinstance(m, dict)]
    return messages[-1].get("role") if messages else None


def _text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content
                       if isinstance(part, dict) and part.get("type") == "text")
    return ""


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


class PrimeAgentLearner(SystemLearner):
    name = "prime-agent"
    # The storage turn's one fact about prime-agent (storage_turn): its continual harness writes the session's local store
    # unless told global_=True (its system prompt says so), and each instance is a session of its own
    storage_note = ("In your continual harness, entries created without global_=True belong to this session and are "
                    "lost with it.")

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        thinking: str = "xhigh",
        command: str | list[str] = "prime-agent",
        provider_name: str = "arc-proxy",
        extra_args: list[str] | None = None,
        timeout_s: float = 1800.0,
        idle_timeout_s: float = 600.0,
        quiet_probe_s: float = 15.0,
        final_turn_grace_s: float = 20.0,
        action_budget_s: float = 900.0,
        max_model_calls: int = 40,
        reprompt_budget_s: float = 300.0,
        reprompt_max_model_calls: int = 6,
        provider_retries: int = 5,
        provider_backoff_s: float = 60.0,
        startup_timeout_s: float = 300.0,
        envelope_timeout_s: float = 300.0,
        kernel_venv: str | None = None,
        context_window: int = 262_144,
        max_tokens: int = 65_536,
        compact_at: int | None = None,
        compact_keep: int | None = None,
        refine_via: str = "agent",
        refine_timeout_s: float = 600.0,
        sandbox: str = "auto",
        sandbox_read_paths: list[str] | None = None,
        passthrough_env: list[str] | None = None,
        **base: Any,
    ):
        super().__init__(**base)
        self.model = model
        self.thinking = thinking
        self.context_window = context_window
        self.max_tokens = max_tokens
        # prime-agent compacts (summarises the history, keeping the last ~20k tokens) once a finished turn's context
        # passes contextWindow - compaction.reserveTokens; None keeps its default reserve (16,384). It checks only
        # between turns, never inside a tool loop, and its summary reply may take 0.8 x the reserve.
        if compact_at is not None and not 0 < compact_at < context_window:
            raise ValueError(f"compact_at must be within the context window ({context_window}), not {compact_at}")
        self.compact_at = compact_at
        # what a compaction keeps word for word (keepRecentTokens, prime-agent's default 20,000): with the system prompt
        # and the summary it must stay well under compact_at, or prime-agent compacts after nearly every tool call
        # (25 Sep: at 32k with the default, 230 compactions in the first 21 minutes of 50 instances)
        self.compact_keep = compact_keep
        # How the refinement after an instance (refine_every) is started. "agent": a closing turn with the verdict, then
        # a turn asking the model to run refine.run(global_=True) in its Python REPL (the harness rows since 22 Sep).
        # "rpc": the adapter starts it itself, with prime-agent 0.9.5's RPC refine command (global), the verdict as its
        # instructions, so the only model call is the refinement's own (the team rows since 26 Sep, Dan: the storage the
        # update grades is the refinement, not the turns around it). refine_timeout_s bounds the wait for its answer.
        if refine_via not in ("agent", "rpc"):
            raise ValueError(f"refine_via must be 'agent' or 'rpc', not {refine_via!r}")
        self.refine_via = refine_via
        self.refine_timeout_s = refine_timeout_s
        self._verdict: list[str] = []   # the ended instance's verdict, kept for the refinement (refine_via rpc)
        self.command = [command] if isinstance(command, str) else list(command)
        self.provider_name = provider_name
        self.extra_args = list(extra_args or [])
        self.timeout_s = timeout_s
        self.idle_timeout_s = idle_timeout_s
        self.quiet_probe_s = quiet_probe_s
        self.final_turn_grace_s = final_turn_grace_s
        self.action_budget_s = action_budget_s
        self.max_model_calls = max_model_calls
        self.reprompt_budget_s = reprompt_budget_s
        self.reprompt_max_model_calls = reprompt_max_model_calls
        self.provider_retries = provider_retries
        self.provider_backoff_s = provider_backoff_s
        self.startup_timeout_s = startup_timeout_s
        self.envelope_timeout_s = envelope_timeout_s
        self.kernel_venv = kernel_venv or os.environ.get("PRIME_AGENT_KERNEL_VENV")
        self.sandbox = sandbox
        self.sandbox_read_paths = list(sandbox_read_paths or [])
        self.passthrough_env = list(passthrough_env or [])
        self._backend: str | None = None
        self._tmp_dir: Path | None = None
        self._refine_event: dict | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._events: queue.Queue[dict | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._log = None
        self._usage = {"input": 0, "output": 0, "cost": 0.0, "turns": 0}
        self._version: str | None = None
        self._events_log = None
        self._diag = None
        self._needs_settle = False
        self._resume_next_prompt = False  # the prompt after an abort carries a streamingBehavior
        self._budget_pending = False
        self._lost_tails = 0
        self._reattach_needed = False
        self._last_reply = ""

    # -- paths ---------------------------------------------------------------

    @property
    def agent_dir(self) -> Path:
        return self.state_dir / "agent"

    @property
    def sessions_dir(self) -> Path:
        return self.state_dir / "sessions"

    def transcript_paths(self) -> list[Path]:
        # the session transcripts; not the agent directory, whose copy of the kernel venv is most of its 300 MB
        return [self.sessions_dir, self.state_dir / "session-artifacts"]

    @property
    def workspace(self) -> Path:
        return self.state_dir / "workspace"

    @property
    def home_dir(self) -> Path:
        return self.state_dir / "home"

    def _venv(self) -> Path:
        return Path(self.kernel_venv or Path.home() / ".prime" / "agent" / "kernel-venv")

    # -- configuration -------------------------------------------------------

    def _base_url(self) -> str:
        return self.environ.get("ARC_LLM_BASE_URL") or OPENROUTER

    def _link_home_agent_dir(self) -> None:
        """``~/.prime/agent`` is a link to this adapter's agent directory. prime-agent's own docs (its skill-creator
        skill) put a personal skill in ``~/.prime/agent/skills/``, its default agent directory, but the adapter gives it a
        private one (``PRIME_AGENT_CODING_AGENT_DIR``) and a private HOME, and 0.9.5 loads personal skills from the agent
        directory's ``skills/`` (read from its binary, 26 Sep), so a skill written where its docs say was never loaded.
        A relative link, so it holds wherever the sandbox mounts the state directory. A directory already there (a skill
        written there before the link) has its skills moved into the agent directory's and the rest kept aside."""
        link = self.home_dir / ".prime" / "agent"
        target = os.path.relpath(self.agent_dir, link.parent)
        if link.is_symlink():
            if os.readlink(link) == target:
                return
            link.unlink()
        elif link.exists():
            skills = link / "skills"
            if skills.is_dir():
                (self.agent_dir / "skills").mkdir(parents=True, exist_ok=True)
                for skill in skills.iterdir():
                    if not (self.agent_dir / "skills" / skill.name).exists():
                        shutil.move(str(skill), str(self.agent_dir / "skills" / skill.name))
            aside = link.with_name(f"agent.before-link.{int(time.time())}")
            link.rename(aside)
            log.info("moved %s aside to %s and linked it to the agent directory", link, aside)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)

    def _write_config(self) -> None:
        for path in (self.agent_dir, self.sessions_dir, self.workspace, self.home_dir):
            path.mkdir(parents=True, exist_ok=True)
        self._link_home_agent_dir()
        # Start from a copy of the shared kernel venv instead of bootstrapping one per run
        # (which needs the network, and the sandbox has none). A copy, not a link: Python-backed
        # skills the system writes are installed into the venv at session start, and the shared
        # one sits under the sandbox's read-only mount.
        venv = self._venv()
        own = self.agent_dir / "kernel-venv"
        if venv.is_dir() and not own.exists():
            shutil.copytree(venv.resolve(), own, symlinks=True)
        models = {"providers": {self.provider_name: {
            "baseUrl": self._base_url(), "api": "openai-completions", "apiKey": KEY_VARIABLE,
            # thinkingLevelMap: prime-agent's model layer (pi-ai) offers xhigh and max only for a
            # model whose declaration maps them, and clamps a higher request down to high; without
            # this line the xhigh and max cells ran at high (found 23 Sep from reasoning tokens per
            # call, flat above medium while the floors' rose eightfold).
            "models": [{"id": self.model, "reasoning": True, "contextWindow": self.context_window,
                        "maxTokens": self.max_tokens,
                        "thinkingLevelMap": {"xhigh": "xhigh", "max": "max"}}],
        }}}
        (self.agent_dir / "models.json").write_text(json.dumps(models, indent=1))
        # prime-agent runs as shipped: its built-in skills (edit, compact, goal, refine,
        # skill-creator, ...) are the harness's own mechanics. Its integrations are off:
        # web search has no key and the sandbox has no route out; the Linear, Notion and
        # Prime Intellect skills are MCP integrations that prime-agent itself disables
        # without a login, made explicit here so the record does not depend on that.
        settings = self.agent_dir / "settings.json"
        if not settings.exists():
            config: dict[str, Any] = {
                "bundledSkills": {"websearch": False},
                "skills": [f"-{name}/SKILL.md" for name in INTEGRATION_SKILLS],
            }
            if self.compact_at is not None:
                config["compaction"] = {"reserveTokens": self.context_window - self.compact_at}
            if self.compact_keep is not None:
                config.setdefault("compaction", {})["keepRecentTokens"] = self.compact_keep
            settings.write_text(json.dumps(config, indent=1))

    def _child_env(self) -> dict[str, str]:
        """An allowlisted environment: nothing from the host leaks in by accident."""
        env = {k: v for k, v in self.environ.items()
               if k in INHERITED_ENV or k.startswith("LC_") or k in self.passthrough_env}
        if self.environ.get("ARC_LLM_BASE_URL"):
            # The proxy's address and placeholder key, for systems that read them from
            # the environment. Without a proxy nothing about the upstream is passed.
            env.update({k: v for k, v in self.environ.items() if k in PROXY_ENV})
        env.update({
            "HOME": str(self.home_dir),
            "TMPDIR": str(self._tmp_dir or tempfile.gettempdir()),
            "PRIME_AGENT_CODING_AGENT_DIR": str(self.agent_dir),
            "PRIME_AGENT_SESSION_DIR": str(self.sessions_dir),
            "PI_SKIP_VERSION_CHECK": "1",
            KEY_VARIABLE: self.environ.get(KEY_VARIABLE) or os.environ.get("OPENROUTER_API_KEY", ""),
        })
        python = self._venv() / "bin" / "python"
        if python.exists():
            # The venv's interpreter, not the base interpreter it links to: only the
            # venv has prime-agent-runtime and the default kernel packages.
            env["PRIME_AGENT_KERNEL_PYTHON"] = str(python.absolute())
        return env

    def _argv(self) -> list[str]:
        # the executable by its absolute path: behind the proxy bridge it is looked up inside the sandbox, where
        # every PATH entry under a home directory is hidden (25 Sep)
        exe = shutil.which(self.command[0]) or self.command[0]
        argv = [
            exe, *self.command[1:], "--mode", "rpc", "--provider", self.provider_name, "--model", self.model,
            "--thinking", self.thinking, "--cwd", str(self.workspace),
            "--session-dir", str(self.sessions_dir), "--offline",
        ]
        if self._tmp_dir is not None:
            # A daemon per run, on a private socket: no sharing between runs, and the
            # daemon and its kernels inherit this run's sandbox.
            argv += ["--daemon-socket", str(self._tmp_dir / "daemon.sock")]
        return argv + self.extra_args

    def _sandbox_spec(self) -> SandboxSpec:
        reads: list[Path] = []
        root = install_root(self.command[0])
        if root:
            reads.append(root)
        venv = self._venv()
        if venv.exists():
            reads.append(venv)
            runtime = python_runtime_root(venv / "bin" / "python")
            if runtime:
                reads.append(runtime)
        reads += [Path(p) for p in self.sandbox_read_paths]
        reads.append(self.workdir / "netbridge.py")  # the sandbox half of the proxy bridge, inside the masked workbench
        parsed = urlparse(self._base_url())
        ports = [parsed.port] if parsed.hostname in ("127.0.0.1", "localhost", "::1") and parsed.port else []
        temp_dirs = darwin_user_dirs()
        return SandboxSpec(
            read_paths=reads,
            write_paths=[self.state_dir, *temp_dirs],
            tmp_dir=self._tmp_dir, mask_paths=default_mask_paths(), home=self.home_dir, loopback_ports=ports,
            unix_socket_dirs=temp_dirs,
        )

    # -- diagnostics ---------------------------------------------------------

    def note(self, message: str) -> None:
        self._note(message)

    def _note(self, message: str) -> None:
        """Adapter diagnostics, kept with the evidence (``adapter.log``)."""
        log.warning(message)
        if self._diag is None:
            self._diag = (self.workdir / "adapter.log").open("a")
        self._diag.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")
        self._diag.flush()

    def _record_event(self, event: dict) -> None:
        """Every RPC event except streaming deltas, with message bodies trimmed, for
        forensics (``rpc-events.jsonl``). Deltas arrive per token and would dominate."""
        if str(event.get("type", "")).endswith("_update"):
            return
        if self._events_log is None:
            self._events_log = (self.workdir / "rpc-events.jsonl").open("a")
        slim = dict(event)
        if "messages" in slim:
            slim["messages"] = [
                {"role": m.get("role"), "stopReason": m.get("stopReason"), "usage": m.get("usage"),
                 "parts": [p.get("type") for p in m.get("content")] if isinstance(m.get("content"), list) else type(m.get("content")).__name__,
                 "text": _text_of(m)[-2000:]}
                for m in slim["messages"] if isinstance(m, dict)
            ]
        for key in ("text", "delta", "data"):
            if isinstance(slim.get(key), str) and len(slim[key]) > 500:
                slim[key] = slim[key][:500] + "..."
        self._events_log.write(json.dumps({"t": time.time(), **slim}, default=str) + "\n")
        self._events_log.flush()

    # -- process -------------------------------------------------------------

    def start(self) -> None:
        self._backend = resolve_backend(self.sandbox)
        # Short path: UNIX socket names are limited to ~100 bytes, and attempt
        # directories are far longer than that.
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="arc-pa-", dir="/tmp" if os.path.isdir("/tmp") else None))
        self._write_config()
        inner = self._argv()
        self._bridge = None
        if self._backend == "bwrap":
            # Private network namespace: the only way out is the bridge to the proxy.
            inner, self._bridge = start_bridge(inner, proxy_port(self._base_url()),
                                               self._tmp_dir / "proxy.sock", self.workdir / "netbridge.py")
        argv = wrap_command(self._backend, inner, self._sandbox_spec(), self.workdir / "sandbox.sb")
        self._log = (self.workdir / "prime-agent.log").open("a")
        self._events = queue.Queue()
        self._needs_settle = False
        try:
            self._proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                cwd=self.workspace, env=self._child_env(), text=True, bufsize=1,
                start_new_session=True,
            )
        except OSError as error:
            raise SystemUnavailable(f"cannot start {self.command[0]}: {error}") from error
        self._reader = threading.Thread(target=self._pump, daemon=True, name="prime-agent-rpc")
        self._reader.start()
        try:
            self._send({"type": "get_state"})
            self._wait_response("get_state", timeout=self.startup_timeout_s)
        except SystemUnavailable:
            self.stop()  # leave no daemon, worker or temporary directory behind
            raise
        self._note(f"started prime-agent (sandbox={self._backend}, tmp={self._tmp_dir})")

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.close()  # type: ignore[union-attr]
                proc.wait(timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._shutdown_daemon()
        if proc is not None and proc.poll() is None:
            self._kill_group(proc)
        bridge, self._bridge = getattr(self, "_bridge", None), None
        if bridge is not None:
            try:
                bridge.kill()
            except OSError:
                pass
        for name in ("_log", "_events_log", "_diag"):
            handle = getattr(self, name)
            if handle is not None:
                handle.close()
                setattr(self, name, None)
        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    def _shutdown_daemon(self) -> None:
        """Stop this run's daemon, workers and kernels (they outlive the RPC client)."""
        if self._tmp_dir is None:
            return
        socket = self._tmp_dir / "daemon.sock"
        if socket.exists():
            # `shutdown` finds this run's daemon through the same HOME/agent directory.
            try:
                subprocess.run([*self.command, "shutdown", "--force", "--json"],
                               env=self._child_env(), capture_output=True, timeout=60, check=False)
            except (OSError, subprocess.TimeoutExpired) as error:
                log.warning("prime-agent shutdown failed: %s", error)
        # Belt and braces: the daemon runs in its own process group (it detaches from
        # the client), so anything still advertising this run's socket is killed by group.
        try:
            out = subprocess.run(["pgrep", "-f", f"daemon-socket {socket}"], capture_output=True,
                                 text=True, timeout=15, check=False).stdout.split()
        except (OSError, subprocess.TimeoutExpired):
            out = []
        groups = set()
        for pid in out:
            try:
                groups.add(os.getpgid(int(pid)))
            except (ProcessLookupError, ValueError, PermissionError):
                continue
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pgid in groups:
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    continue
            if groups:
                time.sleep(1.0)

    @staticmethod
    def _kill_group(proc: subprocess.Popen) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=10)
                return
            except subprocess.TimeoutExpired:
                continue

    def _pump(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                log.debug("non-JSON line from prime-agent: %s", line[:200])
                continue
            self._record_event(event)
            self._events.put(event)
        self._events.put(None)

    def _send(self, command: dict) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            raise SystemUnavailable("prime-agent is not running")
        try:
            proc.stdin.write(json.dumps(command) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError) as error:
            raise SystemUnavailable(f"cannot write to prime-agent: {error}") from error

    def _next(self, deadline: float) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        try:
            event = self._events.get(timeout=remaining)
        except queue.Empty:
            raise TimeoutError from None
        if event is None:
            raise SystemUnavailable("prime-agent exited")
        return event

    def _wait_response(self, command: str, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        try:
            while True:
                event = self._next(deadline)
                if event.get("type") == "response" and event.get("command") == command:
                    if not event.get("success", False):
                        raise SystemUnavailable(f"{command} failed: {event.get('error')}")
                    return event
        except TimeoutError:
            self._note(f"no response envelope for {command} within {timeout:g}s")
            raise SystemUnavailable(f"no response to {command} within {timeout:g}s") from None

    # -- turn control --------------------------------------------------------

    def _proxy_activity(self) -> float:
        return float(getattr(self.tracker, "last_activity", 0.0) or 0.0)

    def _proxy_credit(self) -> float:
        """The seconds the proxy's calls have spent so far on work that is not the model's reply (Tracker.credit_s), which a turn's budget and hard timeout leave out; 0 for any other row."""
        return float(getattr(self.tracker, "credit_s", 0.0) or 0.0)

    def _abort(self, collect_s: float = 0.0) -> str:
        """Abort the running turn. With ``collect_s`` > 0, wait that long for the
        ``agent_end`` of the aborted run and return its last text."""
        text = ""
        try:
            self._send({"type": "abort"})
            self._wait_response("abort", timeout=30.0)
        except SystemUnavailable as error:
            log.warning("abort failed: %s", error)
        if collect_s > 0:
            deadline = time.monotonic() + collect_s
            try:
                while True:
                    event = self._next(deadline)
                    if event.get("type") == "agent_end":
                        text, _ = self._finish_turn(event)
                        break
            except (TimeoutError, SystemUnavailable):
                pass
        self._needs_settle = True
        return text

    def _settle(self) -> None:
        """After an abort: drain what is left of the old run, then make sure the
        agent is idle before the next prompt (else it answers 'already processing')."""
        quiet_until = time.monotonic() + 2.0
        hard = time.monotonic() + 30.0
        while time.monotonic() < min(quiet_until, hard):
            try:
                event = self._next(quiet_until)
            except TimeoutError:
                break
            if event.get("type") in ACTIVITY:
                quiet_until = time.monotonic() + 2.0
        self._send({"type": "get_state"})
        state = self._wait_response("get_state", timeout=30.0).get("data") or {}
        if state.get("isStreaming"):
            raise SystemUnavailable("agent still streaming after abort")
        # abort suspends the session's input pump; until it is resumed every plain prompt is
        # refused with "Cannot admit a session action while queued session input is
        # suspended". A prompt that names a streamingBehavior is admitted with resumeIfIdle
        # (prime-agent 0.9.5's RPC handler), so the next prompt after an abort carries one.
        self._resume_next_prompt = True
        self._needs_settle = False

    def _wait_idle(self, timeout: float) -> None:
        """Wait until prime-agent is neither answering nor compacting."""
        deadline = time.monotonic() + timeout
        while True:
            self._send({"type": "get_state"})
            state = self._wait_response("get_state", timeout=30.0).get("data") or {}
            if not state.get("isStreaming") and not state.get("isCompacting"):
                return
            if time.monotonic() >= deadline:
                raise SystemUnavailable(f"prime-agent still busy after {timeout:g}s")
            time.sleep(2.0)

    def ask(self, message: str, *, reprompt: bool = False, until: tuple[str, ...] = ("agent_end",)) -> str:
        if self._reattach_needed:
            self._reattach_needed = False
            raise SystemUnavailable("event stream lost on consecutive turns; reattaching")
        if self._needs_settle:
            self._settle()
        budget_s = self.reprompt_budget_s if reprompt else self.action_budget_s
        max_calls = self.reprompt_max_model_calls if reprompt else self.max_model_calls
        if reprompt and self._budget_pending:
            message = BUDGET_NOTICE + "\n\n" + message
        self._budget_pending = False
        # A provider failure (rate limit, 5xx) is infrastructure, not the system's answer.
        # prime-agent retries model calls itself; if it gives up, the prompt is re-sent
        # with a backoff before the turn is declared empty (which costs the instance).
        backoff = self.provider_backoff_s
        for attempt in range(self.provider_retries + 1):
            outcome, text = self._turn(message, budget_s=budget_s, max_calls=max_calls, until=until)
            self.last_turn_cut = outcome == "budget"
            self.last_turn_outcome = {"stop_reason": outcome, **({"provider_retries": attempt} if attempt else {})}
            if outcome != "provider_error":
                self._last_reply = text
                return text
            if attempt == self.provider_retries:
                self._note(f"provider errors persisted through {attempt} re-sent prompts; the turn is empty")
                return ""
            self.stats["provider_retries"] += 1
            self._note(f"the model call failed and prime-agent stopped retrying; re-sending the "
                       f"prompt in {backoff:g}s ({attempt + 1}/{self.provider_retries})")
            time.sleep(backoff)
            backoff = min(backoff * 2, 300.0)
            self._settle()
        return ""

    def _turn(self, message: str, *, budget_s: float, max_calls: int,
              until: tuple[str, ...]) -> tuple[str, str]:
        """Run one prompt to completion. Returns ``(outcome, text)`` with outcome one of
        ``reply`` (a final reply, possibly empty), ``budget`` (aborted for budget) or
        ``provider_error`` (the model call failed and nothing was said)."""
        request_id = uuid.uuid4().hex
        command = {"type": "prompt", "message": message, "id": request_id}
        if self._resume_next_prompt:
            # delivered when the run is idle, which it is after an abort; lifts the suspension
            command["streamingBehavior"] = "followUp"
            self._resume_next_prompt = False
        self._send(command)
        try:
            self._wait_response("prompt", timeout=self.envelope_timeout_s)
        except SystemUnavailable as error:
            if "already processing" not in str(error):
                raise
            # prime-agent compacts after a turn whose context passed its threshold, and refuses a prompt until it is
            # done: wait for it, then send the prompt again. (25 Sep: taken for a dead system, the harness was
            # restarted, lost its place, and most instances that compacted were given up.)
            self.stats["busy_waits"] = self.stats.get("busy_waits", 0) + 1
            self._note("prime-agent is busy (compacting); waiting for it to finish before the prompt")
            self._wait_idle(self.timeout_s)
            command["id"] = request_id = uuid.uuid4().hex
            self._send(command)
            self._wait_response("prompt", timeout=self.envelope_timeout_s)

        start = time.monotonic()
        credit0 = self._proxy_credit()
        hard = start + self.timeout_s
        budget_end = start + budget_s
        last_rpc = start  # last event from the worker's event stream (not our own probes)
        last_probe = start
        probe: tuple[str, str] | None = None  # (command id, stage) of an in-flight probe
        retry_pending_until = 0.0  # prime-agent announced an auto-retry due at this time
        calls = 0
        final: str | None = None  # text of a turn_end that ended on a final reply
        final_at = 0.0
        seen_final: str | None = None  # last final assistant message_end seen this turn
        seen_error = False  # the last assistant message_end seen this turn was an error
        while True:
            now = time.monotonic()
            # what the proxy credited during this turn moves both ends out: the budget counts the model's own time
            credited = self._proxy_credit() - credit0
            hard = start + self.timeout_s + credited
            budget_end = start + budget_s + credited
            if now >= hard:
                self._note(f"no agent_end within {self.timeout_s:g}s; aborting the turn")
                self._abort()
                raise SystemUnavailable("turn timed out")
            if final is not None and now - final_at >= self.final_turn_grace_s:
                self.stats["stalls"] += 1
                self._note(f"turn ended on a final reply but no agent_end followed within "
                           f"{self.final_turn_grace_s:g}s; taking the reply")
                self._abort()
                return "reply", final
            idle = now - max(last_rpc, self._proxy_activity(), retry_pending_until)
            if idle >= self.idle_timeout_s:
                self.stats["idle_timeouts"] += 1
                self._note(f"no RPC event and no proxy traffic for {idle:.0f}s; aborting the turn")
                self._abort()
                raise SystemUnavailable(f"system idle for {idle:.0f}s")
            if now >= budget_end or calls > max_calls:
                self.stats["budget_hits"] += 1
                what = f"{calls} model calls" if calls > max_calls else f"{now - start - credited:.0f}s"
                self._note(f"action budget exceeded ({what}); aborting the turn and asking for a decision")
                text = self._abort(collect_s=60.0)
                self._budget_pending = True  # the follow-up carries BUDGET_NOTICE
                return "budget", text
            # The event stream has gone quiet. prime-agent 0.9.5 can finish a turn and never
            # deliver its closing events to an RPC client, while the command path keeps
            # working: ask the worker directly whether it is still running.
            if (self.quiet_probe_s > 0 and probe is None and now >= retry_pending_until
                    and now - last_rpc >= self.quiet_probe_s and now - last_probe >= self.quiet_probe_s):
                probe = (uuid.uuid4().hex, "state")
                last_probe = now
                self._send({"type": "get_state", "id": probe[0]})

            wake = min(hard, budget_end, last_rpc + self.idle_timeout_s)
            if final is not None:
                wake = min(wake, final_at + self.final_turn_grace_s)
            if self.quiet_probe_s > 0:
                wake = min(wake, max(last_rpc, last_probe, retry_pending_until) + self.quiet_probe_s)
            try:
                event = self._events.get(timeout=max(0.05, min(wake - now, 5.0)))
            except queue.Empty:
                continue
            if event is None:
                raise SystemUnavailable("prime-agent exited")
            kind = event.get("type")
            if kind == "response" and probe is not None and event.get("id") == probe[0]:
                data = event.get("data") if isinstance(event.get("data"), dict) else {}
                stage, probe = probe[1], None
                if stage == "state":
                    if not (event.get("success") and data.get("isStreaming") is False):
                        continue  # still running; probe again after another quiet period
                    if seen_final is not None:
                        return "reply", self._lost_tail(seen_final, now - last_rpc)
                    if seen_error:
                        # The run ended on a model error, no retry is pending and the
                        # worker is idle: the turn produced nothing.
                        return "provider_error", ""
                    probe = (uuid.uuid4().hex, "text")
                    self._send({"type": "get_last_assistant_text", "id": probe[0]})
                    continue
                text = data.get("text") or ""
                # Only a reply this turn produced counts; the last assistant text can be
                # the previous turn's if the current one has not answered yet.
                if text.strip() and text != self._last_reply:
                    return "reply", self._lost_tail(text, now - last_rpc)
                continue
            last_rpc = time.monotonic()
            message_obj = event.get("message") if isinstance(event.get("message"), dict) else {}
            if kind == "message_start" and message_obj.get("role") == "assistant":
                calls += 1
            elif kind == "message_end" and message_obj.get("role") == "assistant":
                text = _text_of(message_obj)
                if message_obj.get("stopReason") == "error":
                    seen_error, seen_final = True, None
                elif message_obj.get("stopReason") == "stop" and text.strip():
                    seen_error, seen_final = False, text
            elif kind == "auto_retry_start":
                delay = float(event.get("delayMs") or 0) / 1000.0
                retry_pending_until = last_rpc + delay + 15.0
                self._note(f"prime-agent is retrying the model call (attempt "
                           f"{event.get('attempt')}/{event.get('maxAttempts')}) in {delay:.0f}s: "
                           f"{str(event.get('errorMessage'))[:120]!r}")
            if kind in ("refine_complete", "refine_failed"):
                # A refinement requested from the kernel runs when the agent's turn ends and the
                # agent is then resumed, so this arrives before the final agent_end: remember it.
                self._refine_event = event
            if kind == "turn_end":
                text = _text_of(message_obj)
                if message_obj.get("stopReason") == "stop" and text.strip():
                    final, final_at = text, last_rpc
                else:
                    final = None
            elif kind in ACTIVITY or kind == "tool_execution_start":
                final = None
                if kind in ("agent_start", "turn_start"):
                    seen_final, seen_error = None, False
            if kind not in until:
                continue
            ended_on_tool = _last_role(event) == "toolResult"
            text, errored = self._finish_turn(event)
            if ended_on_tool and not errored and self._resumes_after_compaction(deadline=hard):
                # prime-agent ends a run whose context passed its compaction threshold after a tool result, compacts,
                # and continues it (25 Sep, compaction at 32k: taken for an empty reply, the harness was restarted)
                self._note("the run stopped after a tool result to compact; prime-agent resumed it")
                last_rpc = time.monotonic()
                final, seen_final, seen_error = None, None, False
                continue
            if text.strip() or not errored:
                self._lost_tails = 0
                return "reply", text
            # The model call errored and prime-agent retries in a fresh agent run: wait for
            # the retry it announces (its backoff grows to minutes), then keep listening.
            self._note("turn ended on a model error with no text; waiting for prime-agent's retry")
            if not self._resumes(deadline=hard):
                return "provider_error", ""
            last_rpc = time.monotonic()
            retry_pending_until = 0.0
            final, seen_final, seen_error = None, None, False

    def _resumes_after_compaction(self, *, deadline: float, within: float = 15.0) -> bool:
        """After a run that stopped on a tool result: True if prime-agent compacts and then starts the run again
        (a compaction's summary call can take minutes)."""
        limit = min(deadline, time.monotonic() + within)
        compacting = False
        try:
            while True:
                event = self._next(limit)
                kind = event.get("type")
                if kind == "compaction_start":
                    compacting, limit = True, deadline
                elif kind == "compaction_end":
                    if event.get("aborted"):
                        return False
                    limit = min(deadline, time.monotonic() + within)
                elif kind in ACTIVITY:
                    return compacting
        except TimeoutError:
            return False

    def _lost_tail(self, text: str, quiet: float) -> str:
        self.stats["stalls"] += 1
        self._lost_tails += 1
        self._note(f"the worker finished the turn but its closing events never arrived "
                   f"(quiet {quiet:.0f}s); recovered the reply over the command path")
        if self._lost_tails >= 2:
            self._note("closing events lost on consecutive turns; the event attachment is dead, "
                       "prime-agent will be restarted before the next prompt")
            self._reattach_needed = True
        return text

    def _resumes(self, *, deadline: float, within: float = 15.0) -> bool:
        """After an errored run: True if a new agent run starts. ``auto_retry_start`` extends
        the wait to the delay prime-agent announces; a failed ``auto_retry_end`` ends it."""
        limit = min(deadline, time.monotonic() + within)
        try:
            while True:
                event = self._next(limit)
                kind = event.get("type")
                if kind == "auto_retry_start":
                    delay = float(event.get("delayMs") or 0) / 1000.0
                    limit = min(deadline, time.monotonic() + delay + within)
                    self._note(f"prime-agent is retrying the model call (attempt "
                               f"{event.get('attempt')}/{event.get('maxAttempts')}) in {delay:.0f}s")
                elif kind == "auto_retry_end" and event.get("success") is False:
                    return False
                elif kind in ACTIVITY:
                    return True
        except TimeoutError:
            return False

    def _finish_turn(self, event: dict) -> tuple[str, bool]:
        """Fold usage into the counters; return the final assistant text and whether the
        turn's last assistant message ended on a model error."""
        messages = [m for m in event.get("messages") or [] if isinstance(m, dict)]
        assistant = [m for m in messages if m.get("role") == "assistant"]
        for message in assistant:
            usage = message.get("usage") or {}
            self._usage["input"] += int(usage.get("input") or 0)
            self._usage["output"] += int(usage.get("output") or 0)
            cost = usage.get("cost")
            total = cost.get("total") if isinstance(cost, dict) else cost
            if isinstance(total, (int, float)):
                self._usage["cost"] += float(total)
        self._usage["turns"] += len(assistant)
        if not assistant:
            return "", False
        # The reply is the last assistant message that carries text; a trailing
        # error or tool-only message must not hide it.
        text = next((_text_of(m) for m in reversed(assistant) if _text_of(m).strip()), "")
        return text, assistant[-1].get("stopReason") == "error"

    # -- SystemLearner -------------------------------------------------------

    def new_session(self) -> None:
        self._send({"type": "new_session"})
        self._wait_response("new_session", timeout=60.0)
        self._fresh_session = True

    def wipe_state(self) -> None:
        """Delete everything prime-agent persisted: memories, skills, sessions, session
        artifacts, HOME, the workspace. The provider config, the kernel venv link and
        the adapter's own logs are kept for the record."""
        if not self.state_dir.is_dir():
            return
        for child in self.state_dir.iterdir():
            if child == self.agent_dir:
                for inner in child.iterdir():
                    if inner.name in PRESERVED:
                        continue
                    _remove(inner)
            else:
                _remove(child)

    def _close_instance(self, obs: Any) -> None:
        """With ``refine_via="rpc"``, a refinement to follow and no storage turn, the verdict is kept for the refinement's
        instructions instead of being sent in a closing turn, which would cost a model call answering "done"."""
        if self.refine_via == "rpc" and self._refine_due() and not (self.storage_turn and not self.stateless):
            self._verdict = self.transcript.take()
            return
        super()._close_instance(obs)

    def refine(self) -> None:
        """prime-agent's own cross-session refinement, at the end of every instance.

        ``/refine`` reviews the trajectory but writes the session-local store, which dies
        with the session. The global (cross-session) store is reached through the kernel
        call its refine skill documents, ``refine.run(global_=True)``, so that is what the
        adapter asks for (``refine_via="agent"``). Refinement runs when the turn ends and emits
        ``refine_complete``; the agent's loop then goes on to its reply, so the adapter waits
        for the whole exchange to go quiet before the next session is opened. With
        ``refine_via="rpc"`` the adapter starts it itself (:meth:`_refine_rpc`)."""
        if self.refine_via == "rpc":
            self._refine_rpc()
            return
        self._refine_event = None
        try:
            started = time.monotonic()
            reply = self.ask(REFINE_PROMPT)
            self.log_turn("refine", REFINE_PROMPT, reply, seconds=round(time.monotonic() - started, 2))
            self._await_refine()
        except SystemUnavailable as error:
            log.warning("refine failed: %s", error)

    def _refine_rpc(self) -> None:
        """The global refinement, started by the adapter: prime-agent 0.9.5's RPC ``refine`` command with ``global``
        and the ended instance's verdict as its instructions (:data:`REFINE_INSTRUCTIONS`). prime-agent makes the
        refinement's one model call, applies its edits to the global store and answers the command; the refinement's
        own ``refine_complete`` event, should it come first, ends the wait as well. Logged as a turn of kind
        ``refine``: the instructions, and the summary of what was refined."""
        verdict, self._verdict = self._verdict, []
        instructions = REFINE_INSTRUCTIONS.format(verdict="\n".join(verdict) if verdict else "The instance is over.")
        self._refine_event = None
        started = time.monotonic()
        summary = ""
        try:
            if self._needs_settle:
                self._settle()
            self._send({"type": "refine", "global": True, "instructions": instructions})
            deadline = started + self.refine_timeout_s
            while True:
                event = self._next(deadline)
                kind = event.get("type")
                if kind == "response" and event.get("command") == "refine":
                    if not event.get("success", False):
                        raise SystemUnavailable(f"refine failed: {event.get('error')}")
                    result = event.get("data") or {}
                    break
                if kind in ("refine_complete", "refine_failed"):
                    if kind == "refine_failed":
                        raise SystemUnavailable(f"refine failed: {json.dumps(event)[:300]}")
                    result = event.get("result") or {}
                    break
            summary = str(result.get("summary") or "")
            self._note(f"refine ({result.get('scope', '?')}): {summary[:200]}")
        except TimeoutError:
            self._note(f"no answer to the refine command within {self.refine_timeout_s:g}s")
        except SystemUnavailable as error:
            log.warning("refine failed: %s", error)
            self._note(f"refine failed: {error}")
        self.log_turn("refine", instructions, summary, seconds=round(time.monotonic() - started, 2))

    def _await_refine(self, timeout_s: float = 300.0) -> None:
        if self._refine_event is not None:
            self._log_refine(self._refine_event)
            return
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                event = self._next(deadline)
            except TimeoutError:
                self._note(f"no refine_complete within {timeout_s:g}s of the refine request")
                return
            if event.get("type") in ("refine_complete", "refine_failed"):
                self._log_refine(event)
                return

    def _log_refine(self, event: dict) -> None:
        if event.get("type") == "refine_complete":
            result = event.get("result") or {}
            self._note(f"refine ({result.get('scope', '?')}): {str(result.get('summary', ''))[:200]}")
        else:
            self._note(f"refine failed: {json.dumps(event)[:300]}")

    def describe(self) -> dict[str, Any]:
        if self._version is None:
            try:
                out = subprocess.run([*self.command, "--version"], capture_output=True, text=True,
                                     timeout=60, check=False)
                self._version = (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr).strip() else "unknown"
            except (OSError, subprocess.TimeoutExpired):
                self._version = "unknown"
        return {**super().describe(), "system_version": self._version, "model": self.model,
                "thinking": self.thinking, "refine_every": self.refine_every, "refine_via": self.refine_via,
                "timeout_s": self.timeout_s, "idle_timeout_s": self.idle_timeout_s,
                "quiet_probe_s": self.quiet_probe_s,
                "final_turn_grace_s": self.final_turn_grace_s,
                "action_budget_s": self.action_budget_s, "max_model_calls": self.max_model_calls,
                "reprompt_budget_s": self.reprompt_budget_s,
                "reprompt_max_model_calls": self.reprompt_max_model_calls,
                "provider_retries": self.provider_retries,
                "provider_backoff_s": self.provider_backoff_s,
                "sandbox": resolve_backend(self.sandbox),
                "context_window": self.context_window, "max_tokens": self.max_tokens, "compact_at": self.compact_at,
                "compact_keep": self.compact_keep}

    def system_usage(self) -> dict[str, float]:
        return {
            "system_input_tokens": float(self._usage["input"]),
            "system_output_tokens": float(self._usage["output"]),
            "system_cost_usd": float(self._usage["cost"]),
            "system_turns": float(self._usage["turns"]),
        }
