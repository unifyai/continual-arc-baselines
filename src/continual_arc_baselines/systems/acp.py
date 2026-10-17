"""Systems that speak the Agent Client Protocol (ACP) over stdio.

hermes-agent, openclaw and opencode each ship an ACP server: one long-running
process, one JSON-RPC session, one ``prompt`` per turn, and the reply streamed
back as ``agent_message_chunk`` updates. This module holds everything those
three adapters share; each subclass says only how to launch its system, which
files to write for it, and which of its state to wipe.

The transport is the ``agent-client-protocol`` package (the same one the
verifiers harnesses use for RL rollouts, so a system runs the same way in the
frozen rows and in the trained ones). It is asynchronous; the learner interface
is not, so a private event loop runs on a thread and every call is a
``run_coroutine_threadsafe`` with a timeout.

Confinement is the same as for prime-agent: the whole process tree runs in the
sandbox (:mod:`continual_arc_baselines.systems.sandbox`), reads only its own
installation and runtime, writes only under its private state directory, and
opens TCP only to the tracking proxy (and, for a system that runs a gateway for
itself, that gateway's loopback port).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import AsyncExitStack, suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from continual_arc_baselines import secrets
from continual_arc_baselines.systems.base import SystemLearner, SystemUnavailable
from continual_arc_baselines.systems.protocol import BUDGET_NOTICE
from continual_arc_baselines.systems.sandbox import (
    SandboxSpec,
    darwin_user_dirs,
    default_mask_paths,
    install_root,
    package_manager_roots,
    proxy_port,
    resolve_backend,
    start_bridge,
    wrap_command,
)

log = logging.getLogger(__name__)

OPENROUTER = "https://openrouter.ai/api/v1"
KEY_VARIABLE = "ARC_LLM_API_KEY"
INHERITED_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ")
PROXY_ENV = ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_KEY", "OPENROUTER_BASE_URL",
             "OPENROUTER_API_KEY", "ARC_LLM_BASE_URL")


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _Client:
    """The client half of the protocol: collects the streamed reply, approves
    every permission request (the sandbox is the policy, not the prompt), and
    counts tool calls for the record."""

    def __init__(self) -> None:
        self.text = ""
        self.message_id: str | None = None
        self.tool_calls = 0
        self.updates = 0
        self.last_update = time.monotonic()
        self.reset()

    def reset(self) -> None:
        self.text = ""
        self.message_id = None
        self.last_update = time.monotonic()

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        from acp.schema import AgentMessageChunk, TextContentBlock

        self.updates += 1
        self.last_update = time.monotonic()
        kind = getattr(update, "session_update", None)
        if kind == "tool_call":
            self.tool_calls += 1
        if isinstance(update, AgentMessageChunk) and isinstance(update.content, TextContentBlock):
            message_id = getattr(update, "message_id", None)
            if message_id is not None and message_id != self.message_id:
                self.text = ""
                self.message_id = message_id
            self.text += update.content.text

    async def request_permission(self, session_id: str, tool_call: Any, options: list, **kwargs: Any):
        from acp.schema import AllowedOutcome, DeniedOutcome, RequestPermissionResponse

        option = next((o for o in options if o.kind in ("allow_once", "allow_always")), None)
        outcome = (AllowedOutcome(outcome="selected", option_id=option.option_id) if option
                   else DeniedOutcome(outcome="cancelled"))
        return RequestPermissionResponse(outcome=outcome)

    # The agent may ask for these when the client advertises the capability; it
    # does not, so refusing is the correct answer if one arrives anyway.
    async def read_text_file(self, *args: Any, **kwargs: Any):
        raise RuntimeError("client file access is not offered")

    async def write_text_file(self, *args: Any, **kwargs: Any):
        raise RuntimeError("client file access is not offered")

    async def create_terminal(self, *args: Any, **kwargs: Any):
        raise RuntimeError("client terminals are not offered")

    async def terminal_output(self, *args: Any, **kwargs: Any):
        raise RuntimeError("client terminals are not offered")

    async def wait_for_terminal_exit(self, *args: Any, **kwargs: Any):
        raise RuntimeError("client terminals are not offered")

    async def kill_terminal(self, *args: Any, **kwargs: Any):
        raise RuntimeError("client terminals are not offered")

    async def release_terminal(self, *args: Any, **kwargs: Any):
        raise RuntimeError("client terminals are not offered")

    async def create_elicitation(self, *args: Any, **kwargs: Any):
        raise RuntimeError("elicitation is not offered")

    async def complete_elicitation(self, *args: Any, **kwargs: Any):
        return None

    async def ext_method(self, *args: Any, **kwargs: Any):
        return {}

    async def ext_notification(self, *args: Any, **kwargs: Any):
        return None

    def on_connect(self, *args: Any, **kwargs: Any) -> None:
        return None


class ACPSystemLearner(SystemLearner):
    """Base for systems driven over ACP. Subclasses implement :meth:`argv`,
    :meth:`prepare` (write the system's config into its private state), and
    optionally :meth:`extra_env`, :meth:`extra_read_paths`, :meth:`listen_ports`,
    :meth:`preserved` and :meth:`refine`."""

    name = "acp"

    def __init__(
        self,
        model: str,
        command: str | list[str] | None = None,
        timeout_s: float = 1800.0,
        idle_timeout_s: float = 600.0,
        startup_timeout_s: float = 300.0,
        reprompt_timeout_s: float = 600.0,
        cancel_grace_s: float = 60.0,
        context_window: int = 262_144,
        max_tokens: int = 65_536,
        sandbox: str = "auto",
        sandbox_read_paths: list[str] | None = None,
        passthrough_env: list[str] | None = None,
        **base: Any,
    ):
        super().__init__(**base)
        self.model = model
        if command is None:
            command = self.default_command()
        self.command = [command] if isinstance(command, str) else list(command)
        self.timeout_s = timeout_s
        self.idle_timeout_s = idle_timeout_s
        self.startup_timeout_s = startup_timeout_s
        self.reprompt_timeout_s = reprompt_timeout_s
        self.cancel_grace_s = cancel_grace_s
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.sandbox = sandbox
        self.sandbox_read_paths = list(sandbox_read_paths or [])
        self.passthrough_env = list(passthrough_env or [])
        self._backend: str | None = None
        self._tmp_dir: Path | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stack: AsyncExitStack | None = None
        self._conn: Any = None
        self._proc: Any = None
        self._session_id: str | None = None
        self._budget_pending = False  # the next re-prompt carries the budget notice
        self._capabilities: Any = None
        self._client = _Client()
        self._log = None
        self._diag = None
        self._usage = {"input": 0, "output": 0, "turns": 0, "tool_calls": 0}
        # ACP's PromptResponse.usage is the session total across all turns (schema: "Total input
        # tokens across all turns"), not the turn's own; _usage_seen is the latest total of the
        # current session and _usage_base what earlier sessions of this cell reported.
        self._usage_base = {"input": 0, "output": 0}
        self._usage_seen = {"input": 0, "output": 0}
        self._version: str | None = None

    # -- to implement per system -------------------------------------------

    def default_command(self) -> list[str]:
        raise NotImplementedError

    def argv(self) -> list[str]:
        """The command line that starts the ACP server on stdio."""
        raise NotImplementedError

    def prepare(self) -> None:
        """Write the system's configuration under :attr:`state_dir`."""
        raise NotImplementedError

    def extra_env(self) -> dict[str, str]:
        return {}

    def extra_read_paths(self) -> list[Path]:
        return []

    def listen_ports(self) -> list[int]:
        return []

    def extra_write_paths(self) -> list[Path]:
        """Directories outside the state directory the system insists on writing
        (created by the adapter before launch so the sandbox can name them)."""
        return []

    def preserved(self) -> tuple[str, ...]:
        """Names directly under :attr:`state_dir` that survive a state wipe
        (configuration the adapter wrote, not state the system accumulated)."""
        return ()

    def version(self) -> str:
        return "unknown"

    # -- paths ---------------------------------------------------------------

    @property
    def state_dir(self) -> Path:
        return self.workdir / "state"

    @property
    def home_dir(self) -> Path:
        return self.state_dir / "home"

    @property
    def workspace(self) -> Path:
        return self.state_dir / "workspace"

    def base_url(self) -> str:
        return self.environ.get("ARC_LLM_BASE_URL") or OPENROUTER

    def api_key(self) -> str:
        return self.environ.get(KEY_VARIABLE) or secrets.get("OPENROUTER_API_KEY", default="")

    # -- diagnostics ---------------------------------------------------------

    def note(self, message: str) -> None:
        log.warning(message)
        if self._diag is None:
            self.workdir.mkdir(parents=True, exist_ok=True)
            self._diag = (self.workdir / "adapter.log").open("a")
        self._diag.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")
        self._diag.flush()

    # -- environment and sandbox --------------------------------------------

    def child_env(self) -> dict[str, str]:
        env = {k: v for k, v in self.environ.items()
               if k in INHERITED_ENV or k.startswith("LC_") or k in self.passthrough_env}
        if self.environ.get("ARC_LLM_BASE_URL"):
            env.update({k: v for k, v in self.environ.items() if k in PROXY_ENV})
        env.update({
            "HOME": str(self.home_dir),
            "TMPDIR": str(self._tmp_dir or tempfile.gettempdir()),
            "NO_COLOR": "1",
            KEY_VARIABLE: self.api_key(),
        })
        env.update(self.extra_env())
        return env

    def sandbox_spec(self) -> SandboxSpec:
        reads: list[Path] = []
        root = install_root(self.command[0])
        if root:
            reads.append(root)
        reads += self.extra_read_paths()
        reads += [Path(p).expanduser() for p in self.sandbox_read_paths]
        reads.append(self.workdir / "netbridge.py")  # the sandbox half of the proxy bridge, inside the masked workbench
        for path in list(reads):
            reads += package_manager_roots(path)
        parsed = urlparse(self.base_url())
        ports = [parsed.port] if parsed.hostname in ("127.0.0.1", "localhost", "::1") and parsed.port else []
        temp_dirs = darwin_user_dirs()
        return SandboxSpec(
            read_paths=reads, write_paths=[self.state_dir, *temp_dirs, *self.extra_write_paths()],
            tmp_dir=self._tmp_dir, mask_paths=default_mask_paths(),
            home=self.home_dir, loopback_ports=ports, listen_ports=self.listen_ports(),
            unix_socket_dirs=temp_dirs,
        )

    # -- the event loop ------------------------------------------------------

    def _start_loop(self) -> None:
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name=f"{self.name}-acp")
        self._thread.start()

    def _stop_loop(self) -> None:
        loop, self._loop = self._loop, None
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._thread = None
        with suppress(Exception):
            loop.close()

    def _call(self, coro, timeout: float, what: str):
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise SystemUnavailable(f"{what} did not complete within {timeout:g}s") from None
        except Exception as error:
            raise SystemUnavailable(f"{what} failed: {type(error).__name__}: {error}") from error

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._backend = resolve_backend(self.sandbox)
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="arc-acp-", dir="/tmp" if os.path.isdir("/tmp") else None))
        for path in (self.state_dir, self.home_dir, self.workspace):
            path.mkdir(parents=True, exist_ok=True)
        self.prepare()
        inner = self.argv()
        if self._backend == "bwrap":
            # Private network namespace: the only way out is the bridge to the proxy.
            inner, self._bridge = start_bridge(inner, proxy_port(self.base_url()),
                                               self._tmp_dir / "proxy.sock", self.workdir / "netbridge.py")
        argv = wrap_command(self._backend, inner, self.sandbox_spec(), self.workdir / "sandbox.sb")
        self._log = (self.workdir / f"{self.name}.log").open("a")
        self._client = _Client()
        self._roll_usage()
        self._start_loop()
        env = self.child_env()

        async def spawn():
            from acp import PROTOCOL_VERSION, spawn_agent_process
            from acp.schema import ClientCapabilities

            stack = AsyncExitStack()
            try:
                conn, proc = await stack.enter_async_context(spawn_agent_process(
                    self._client, argv[0], *argv[1:], env=env, cwd=str(self.workspace),
                    transport_kwargs={"stderr": self._log.fileno()},
                ))
                init = await conn.initialize(protocol_version=PROTOCOL_VERSION,
                                             client_capabilities=ClientCapabilities())
                session = await conn.new_session(cwd=str(self.workspace), mcp_servers=[])
            except BaseException:
                with suppress(BaseException):
                    await stack.aclose()
                raise
            return stack, conn, proc, session.session_id, init.agent_capabilities

        try:
            self._stack, self._conn, self._proc, self._session_id, self._capabilities = self._call(
                spawn(), self.startup_timeout_s, "start")
        except SystemUnavailable:
            self.stop()
            raise
        self.note(f"started {self.name} (sandbox={self._backend}, tmp={self._tmp_dir})")

    def stop(self) -> None:
        if self._loop is not None and self._stack is not None:
            async def close():
                with suppress(Exception):
                    caps = self._capabilities and getattr(self._capabilities, "session_capabilities", None)
                    if caps is not None and getattr(caps, "close", None) is not None and self._session_id:
                        await asyncio.wait_for(self._conn.close_session(session_id=self._session_id), 10)
                with suppress(Exception):
                    await asyncio.wait_for(self._stack.aclose(), 20)

            with suppress(Exception):
                self._call(close(), 40, "stop")
        proc, self._proc = self._proc, None
        if proc is not None and getattr(proc, "returncode", None) is None:
            self._kill_tree(proc.pid)
        self._stack = self._conn = self._session_id = self._capabilities = None
        bridge, self._bridge = getattr(self, "_bridge", None), None
        if bridge is not None:
            with suppress(OSError):
                bridge.kill()
        self._stop_loop()
        for name in ("_log", "_diag"):
            handle = getattr(self, name)
            if handle is not None:
                handle.close()
                setattr(self, name, None)
        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    @staticmethod
    def _descendants(pid: int) -> list[int]:
        """``pid`` and every process below it, parents first (portable: pgrep)."""
        out = [pid]
        frontier = [pid]
        while frontier:
            parent = frontier.pop()
            try:
                kids = subprocess.run(["pgrep", "-P", str(parent)], capture_output=True, text=True,
                                      timeout=10, check=False).stdout.split()
            except (OSError, subprocess.TimeoutExpired):
                kids = []
            for kid in kids:
                try:
                    out.append(int(kid))
                    frontier.append(int(kid))
                except ValueError:
                    continue
        return out

    @classmethod
    def _kill_tree(cls, pid: int) -> None:
        """Terminate the server and everything it spawned. The protocol library
        cannot put the child in its own session, so the tree is walked instead."""
        pids = cls._descendants(pid)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            alive = False
            for p in reversed(pids):
                try:
                    os.kill(p, sig)
                    alive = True
                except (ProcessLookupError, PermissionError):
                    continue
            if not alive:
                return
            time.sleep(1.0)

    # -- turns ---------------------------------------------------------------

    def ask(self, message: str, *, reprompt: bool = False) -> str:
        if self._conn is None or self._session_id is None:
            raise SystemUnavailable(f"{self.name} is not running")
        budget = self.reprompt_timeout_s if reprompt else self.timeout_s
        if reprompt and self._budget_pending:
            message = BUDGET_NOTICE + "\n\n" + message
        self._budget_pending = False
        self.last_turn_cut = False

        pending: dict[str, Any] = {}
        started_at = time.time()

        async def turn():
            from acp import text_block

            self._client.reset()
            task = asyncio.ensure_future(
                self._conn.prompt(session_id=self._session_id, prompt=[text_block(message)]))
            pending["task"] = task
            # Idle watchdog: no update from the agent and no traffic through the
            # proxy for idle_timeout_s means the system is hung.
            poll = max(0.2, min(5.0, self.idle_timeout_s / 2))
            while True:
                try:
                    return await asyncio.wait_for(asyncio.shield(task), timeout=poll)
                except TimeoutError:
                    if task.done():
                        return task.result()
                    last = max(self._client.last_update, self._proxy_activity())
                    if time.monotonic() - last >= self.idle_timeout_s:
                        with suppress(Exception):
                            await asyncio.wait_for(self._conn.cancel(session_id=self._session_id), 10)
                        task.cancel()
                        raise SystemUnavailable(f"no activity for {self.idle_timeout_s:g}s") from None

        try:
            response = self._call(turn(), budget, "prompt")
        except SystemUnavailable as error:
            if "did not complete within" in str(error):
                # The action budget, handled as prime-agent's is: cancel the turn, keep what
                # was said, and let the follow-up carry the budget notice. The system is
                # working, not dead, so it is not restarted.
                self.stats["budget_hits"] += 1
                self.note(f"action budget exceeded ({budget:g}s); cancelling the turn and asking for a decision")

                async def cancel_and_wait():
                    with suppress(Exception):
                        await asyncio.wait_for(self._conn.cancel(session_id=self._session_id), 10)
                    task = pending.get("task")
                    if task is not None:
                        # the agent ends the turn it was on; a prompt sent before that is only
                        # queued behind it ("Queued for the next turn"), so wait for the end
                        await asyncio.wait_for(asyncio.shield(task), self.cancel_grace_s)

                try:
                    self._call(cancel_and_wait(), self.cancel_grace_s + 15, "cancel")
                except SystemUnavailable as stop_error:
                    self.note(f"the turn did not end within {self.cancel_grace_s:g}s of cancel ({stop_error}); restarting")
                    raise SystemUnavailable("turn did not end after cancel") from None
                self.last_turn_cut = True
                self._budget_pending = True
                return self._client.text
            self.stats["idle_timeouts" if "activity" in str(error) else "stalls"] += 1
            self.note(f"turn failed: {error}")
            raise
        self._usage["turns"] += 1
        self._usage["tool_calls"] = self._client.tool_calls
        usage = getattr(response, "usage", None)
        if usage is not None:
            # session totals, so the latest report replaces the previous one (see __init__); what a
            # harness puts in them is its own affair (the proxy's llm_* counters are the measure)
            for key, attr in (("input", "input_tokens"), ("output", "output_tokens")):
                self._usage_seen[key] = int(getattr(usage, attr, 0) or 0)
                self._usage[key] = self._usage_base[key] + self._usage_seen[key]
        stop = getattr(response, "stop_reason", None)
        text = self._client.text
        self.last_turn_outcome = {"stop_reason": str(stop) if stop is not None else None}
        if not text.strip():
            # an ACP bridge may end a failed run as an ordinary end_turn: ask the harness's own records
            self.last_turn_outcome.update(self.turn_diagnostics(started_at))
            self.note(f"empty reply (stop_reason={stop!r}; {self.last_turn_outcome})")
        return text

    def turn_diagnostics(self, started_at: float) -> dict[str, Any]:
        """What the harness's own records say about a turn that returned no text (wall-clock start ``started_at``),
        for the turn log: nothing by default."""
        return {}

    def _proxy_activity(self) -> float:
        return float(getattr(self.tracker, "last_activity", 0.0) or 0.0)

    def new_session(self) -> None:
        if self._conn is None:
            raise SystemUnavailable(f"{self.name} is not running")

        async def fresh():
            session = await self._conn.new_session(cwd=str(self.workspace), mcp_servers=[])
            return session.session_id

        self._session_id = self._call(fresh(), 60, "new_session")
        self._roll_usage()
        self._fresh_session = True

    def _roll_usage(self) -> None:
        """A new session starts the harness's usage total from zero: bank what the old one reported."""
        for key in ("input", "output"):
            self._usage_base[key] += self._usage_seen[key]
            self._usage_seen[key] = 0

    def wipe_state(self) -> None:
        if not self.state_dir.is_dir():
            return
        keep = set(self.preserved())
        for child in self.state_dir.iterdir():
            if child.name in keep:
                continue
            _remove(child)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "system_version": self.version(), "model": self.model,
                "refine_every": self.refine_every, "timeout_s": self.timeout_s,
                "idle_timeout_s": self.idle_timeout_s, "sandbox": resolve_backend(self.sandbox),
                "context_window": self.context_window, "max_tokens": self.max_tokens,
                "command": self.command}

    def system_usage(self) -> dict[str, float]:
        return {
            "system_input_tokens": float(self._usage["input"]),
            "system_output_tokens": float(self._usage["output"]),
            "system_turns": float(self._usage["turns"]),
            "system_tool_calls": float(self._usage["tool_calls"]),
        }

    def run_cli(self, argv: list[str], timeout: float = 300.0) -> subprocess.CompletedProcess:
        """Run one of the system's own commands with the same environment and
        confinement as the server (for maintenance steps such as applying
        skill proposals)."""
        backend = self._backend or resolve_backend(self.sandbox)
        # A maintenance command may need the gateway the server runs on loopback, which
        # lives in the server's own network namespace: the CLI shares the host network
        # instead. It is not a model-driven turn.
        wrapped = wrap_command(backend, argv, self.sandbox_spec(), self.workdir / "sandbox-cli.sb", share_net=True)
        return subprocess.run(wrapped, env=self.child_env(), cwd=self.workspace, capture_output=True,
                              text=True, timeout=timeout, check=False)
