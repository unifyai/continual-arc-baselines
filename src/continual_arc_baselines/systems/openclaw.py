"""openclaw as a learner: a private Gateway plus its ACP bridge.

openclaw's agent runs inside a Gateway process; ``openclaw acp`` bridges ACP on
stdio to one Gateway session. The adapter starts both under one launcher
script, in the sandbox, with the Gateway bound to a loopback port chosen per
run. Everything the system keeps lives under a private ``OPENCLAW_STATE_DIR``:
its config, per-agent state database, session transcripts, workspace memory
files (``MEMORY.md``, ``DREAMS.md``) and workshop skills.

The system starts as a blank slate: every bundled skill is blocked
(``skills.allowBundled`` names nothing), no messaging channel is loaded
(``OPENCLAW_SKIP_CHANNELS``), and the workspace holds no memory files until the
system writes them.

Self-improvement in openclaw is two mechanisms, both switched by
``self_improve``: the skill workshop's autonomous mode (``"auto"`` captures
reusable workflows from conversations into skills and applies them, ``"off"``
never captures) and dreaming (background memory consolidation that promotes
session material into ``MEMORY.md``). Row H0 runs with both off, row H1 with
both on. ``refine()`` additionally sends ``/learn`` and applies any pending
workshop proposals, for an explicit cadence on top of the autonomous mode.

Install: ``curl -fsSL https://openclaw.ai/install-cli.sh | bash`` (gives
``openclaw`` on PATH), or a built checkout run as ``node openclaw.mjs``; pass
``command`` accordingly.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

from continual_arc_baselines.systems.acp import ACPSystemLearner, free_port
from continual_arc_baselines.systems.sandbox import install_root

CHECKOUT = Path.home() / "openclaw"
PROVIDER = "arc-proxy"
BRIDGE_KEY = "agent:main:acp-bridge"   # the one session the ACP bridge drives (LAUNCHER)
# openclaw's memory search set explicitly (docs/concepts/memory-search.md, docs/reference/memory-config.md in the
# package): keyword-only ranking on purpose (provider "none"; the sandbox reaches no embedding provider), over the
# memory files and the session transcripts, including those from before each instance's /reset.
MEMORY_SEARCH = {"search": {"provider": "none", "experimental": {"sessionMemory": True}, "sources": ["memory", "sessions"]}}

# Starts the Gateway in the background, waits for it, then runs the ACP bridge on
# the inherited stdio. The bridge's exit ends the Gateway. Everything is
# addressed through OPENCLAW_* variables set by the adapter.
LAUNCHER = r'''
set -u
"$@" gateway run --auth token --bind loopback >"$OPENCLAW_STATE_DIR/gateway.log" 2>&1 &
gw=$!
i=0
until curl -fsS "http://127.0.0.1:$OPENCLAW_GATEWAY_PORT/readyz" >/dev/null 2>&1; do
    if ! kill -0 "$gw" 2>/dev/null; then echo "gateway exited" >&2; tail -50 "$OPENCLAW_STATE_DIR/gateway.log" >&2; exit 1; fi
    i=$((i + 1)); [ "$i" -lt 180 ] || { echo "gateway not ready" >&2; tail -50 "$OPENCLAW_STATE_DIR/gateway.log" >&2; kill "$gw"; exit 1; }
    sleep 1
done
"$@" acp --session "agent:main:acp-bridge" --no-prefix-cwd --token "$OPENCLAW_GATEWAY_TOKEN"
status=$?
kill "$gw" 2>/dev/null
wait "$gw" 2>/dev/null
exit $status
'''


def settle_bridge_session(openclaw_dir: Path) -> str | None:
    """End the bridge session's run if a gateway that died mid-run left it running; return what was done.

    openclaw's main-session restart recovery replays such a run at the next boot (it marks the entry
    aborted, then starts it again). It exempts only session keys with an ``acp`` segment, which the
    bridge's key lacks, so the bridge session is replayed like a chat, and the replay holds the session
    when the adapter's first prompt arrives: the prompt gets an empty reply, the adapter restarts, the
    gateway goes down before the replay ends, and every boot replays it again. Row H then gave up every
    instance (26 Sep). A restart in the protocol starts afresh, and the
    adapter drives no gateway while this runs (before every start), so it closes the run as a clean stop
    leaves it: status done, abortedLastRun false, an end time, and no recovery cycle.
    """
    db = openclaw_dir / "agents" / "main" / "agent" / "openclaw-agent.sqlite"
    if not db.exists():
        return None
    con = sqlite3.connect(db, timeout=30)
    try:
        row = con.execute("select status, entry_json from session_nodes where session_key = ?", (BRIDGE_KEY,)).fetchone()
        if row is None:
            return None
        status, entry_json = row
        entry = json.loads(entry_json)
        if status != "running" and entry.get("status") != "running":
            return None
        now = int(time.time() * 1000)
        cycle = entry.get("mainRestartRecovery")
        for key in ("mainRestartRecovery", "lifecycleRunId", "restartRecoveryRuns", "restartRecoveryDeliveryRunId"):
            entry.pop(key, None)
        entry.update(status="done", abortedLastRun=False, endedAt=now, updatedAt=now)
        if isinstance(entry.get("startedAt"), (int, float)):
            entry["runtimeMs"] = max(0, now - int(entry["startedAt"]))
        con.execute("update session_nodes set status = 'done', entry_json = ?, updated_at = ? where session_key = ?",
                    (json.dumps(entry, ensure_ascii=False, separators=(",", ":")), now, BRIDGE_KEY))
        con.commit()
        attempts = cycle.get("chargedAttempts") if isinstance(cycle, dict) else None
        return "settled the bridge session's interrupted run" + (f" (openclaw's recovery at {attempts} attempts)" if attempts else "")
    except (sqlite3.Error, ValueError) as error:
        return f"could not settle the bridge session: {error}"
    finally:
        con.close()


class OpenClawLearner(ACPSystemLearner):
    name = "openclaw"

    @property
    def agent_db(self) -> Path:
        return self.openclaw_dir / "agents" / "main" / "agent" / "openclaw-agent.sqlite"

    def turn_diagnostics(self, started_at: float) -> dict[str, Any]:
        """How openclaw's last run in the bridge session ended (``harness_stop_reason``, ``harness_error``) and how many
        subagent sessions it opened since ``started_at``, read from its agent database. Its ACP bridge reports a run that
        failed as an ordinary end_turn, and a subagent's answer stays in the subagent's session (28 Sep: of 177 empty
        replies in the persistent cells, 70 turns whose work had gone to subagents and 63 failed runs)."""
        if not self.agent_db.exists():
            return {}
        out: dict[str, Any] = {}
        try:
            con = sqlite3.connect(f"file:{self.agent_db}?mode=ro", uri=True, timeout=5)
            try:
                row = con.execute("select current_session_id from session_nodes where session_key = ?",
                                  (BRIDGE_KEY,)).fetchone()
                if row is not None:
                    for (event_json,) in con.execute("select event_json from transcript_events where session_id = ? "
                                                     "order by seq desc limit 40", (row[0],)):
                        message = (json.loads(event_json).get("message") or {})
                        if message.get("role") == "assistant":
                            out["harness_stop_reason"] = message.get("stopReason")
                            if message.get("errorMessage"):
                                out["harness_error"] = str(message["errorMessage"])[:300]
                            break
                out["subagent_sessions"] = int(con.execute(
                    "select count(*) from session_windows where session_key like 'agent:main:subagent:%' "
                    "and created_at >= ?", (int(started_at * 1000),)).fetchone()[0])
            finally:
                con.close()
        except (sqlite3.Error, ValueError, OSError) as error:
            out["diagnostics_error"] = str(error)[:200]
        return out

    def transcript_paths(self) -> list[Path]:
        # the agent database (transcripts, session windows) and the gateway's log; not the 50 MB cache
        return [self.openclaw_dir / "agents", self.openclaw_dir / "state", self.openclaw_dir / "gateway.log"]

    def __init__(self, model: str = "IFM/K2-Horizon-7B", memory_search: bool = False, **kwargs: Any):
        # The workshop's experience review starts only after 30 s of quiet with no run active.
        kwargs.setdefault("instance_gap_s", 40.0)
        self._port: int | None = None
        self._token = secrets.token_hex(16)
        # memory_search configures openclaw's memory search explicitly (MEMORY_SEARCH); off, it is left unset, as the
        # grid cells of 27 and 28 Sep ran it. Unset, openclaw asks for OpenAI embeddings the sandbox cannot reach,
        # builds a keyword index instead and later refuses searches against it ("index was built for model fts-only,
        # expected text-embedding-3-small"), and memory_search over session transcripts answers "Session transcript
        # search is not enabled." (18 of the 23 memory searches at low effort, H-openclaw-luna6-low-ws0, 29 Sep).
        self.memory_search = memory_search
        super().__init__(model=model, **kwargs)

    def default_command(self) -> list[str]:
        found = shutil.which("openclaw")
        if found:
            return [found]
        script = CHECKOUT / "openclaw.mjs"
        node = shutil.which("node") or "node"
        return [node, str(script)]

    @property
    def openclaw_dir(self) -> Path:
        return self.state_dir / "openclaw"

    @property
    def config_path(self) -> Path:
        return self.openclaw_dir / "openclaw.json"

    def argv(self) -> list[str]:
        return ["sh", "-c", LAUNCHER, "openclaw-launcher", *self.command]

    def prepare(self) -> None:
        self.openclaw_dir.mkdir(parents=True, exist_ok=True)
        settled = settle_bridge_session(self.openclaw_dir)
        if settled:
            self.note(settled)
        self._port = free_port()
        config = {
            "gateway": {"mode": "local", "bind": "loopback", "port": self._port,
                        "auth": {"mode": "token", "token": "${OPENCLAW_GATEWAY_TOKEN}"}},
            "messages": {"queue": {"mode": "interrupt"}},
            "agents": {"defaults": {
                "workspace": str(self.workspace), "skipBootstrap": True,
                "heartbeat": {"every": "0m"}, "sandbox": {"mode": "off"},
                "model": {"primary": f"{PROVIDER}/{self.model}"},
            }},
            "tools": {"profile": "coding", "fs": {"workspaceOnly": True},
                      "exec": {"host": "gateway", "mode": "full"}, "deny": []},
            "models": {"providers": {PROVIDER: {
                "baseUrl": self.base_url(), "apiKey": "${ARC_LLM_API_KEY}", "api": "openai-completions",
                "timeoutSeconds": 600, "request": {"allowPrivateNetwork": True},
                "models": [{"id": self.model, "name": self.model, "reasoning": True, "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": self.context_window, "maxTokens": self.max_tokens}],
            }}},
            # Blank slate: ``allowBundled`` is an allowlist over openclaw's bundled
            # skills; a name that matches nothing (the sentinel openclaw's own e2e
            # configs use) blocks all of them. Workshop skills the system writes are
            # unaffected. Channels are skipped via OPENCLAW_SKIP_CHANNELS.
            "skills": {"allowBundled": ["__none__"],
                       "workshop": {"autonomous": {"mode": "auto"}, "approvalPolicy": "auto"}},
            # Dreaming (promotion of recalled notes into MEMORY.md) is a nightly cron as shipped; the
            # stream stands for weeks of work, so its schedule is hourly, about one sweep per pass
            # over the task set at the observed pace.
            "plugins": {"entries": {"memory-core": {"config": {"dreaming": {"enabled": True, "frequency": "0 * * * *"}}}}},
        }
        if self.memory_search:
            config["memory"] = MEMORY_SEARCH
        self.config_path.write_text(json.dumps(config, indent=1))

    def extra_env(self) -> dict[str, str]:
        assert self._port is not None
        return {
            "OPENCLAW_STATE_DIR": str(self.openclaw_dir),
            "OPENCLAW_CONFIG_PATH": str(self.config_path),
            "OPENCLAW_GATEWAY_TOKEN": self._token,
            "OPENCLAW_GATEWAY_PORT": str(self._port),
            "OPENCLAW_HIDE_BANNER": "1",
            "OPENCLAW_SUPPRESS_NOTES": "1",
            "OPENCLAW_SKIP_CHANNELS": "1",
            # At every boot the gateway and then the ACP bridge run openclaw's "repair missing
            # configured plugin installs" step, which asks npm for the Perplexity web-search plugin.
            # The sandbox has no route to the registry (private network namespace; only the bridge
            # to the proxy), so each attempt failed on DNS and then walked npm's retry ladder: two
            # packages, two retries each with backoff, about 70 s per process, 140 s per boot, which
            # was nine tenths of row H0's time between instances (measured 23 Sep, 158 s of which
            # 140 s npm). One attempt, no retries: the same outcome ("continuing with degraded
            # state") in about a second. Nothing the agent does with npm can succeed either way.
            "npm_config_fetch_retries": "0",
            "npm_config_fetch_retry_mintimeout": "1000",
            "npm_config_fetch_retry_maxtimeout": "1000",
        }

    def listen_ports(self) -> list[int]:
        return [self._port] if self._port else []

    def extra_write_paths(self) -> list[Path]:
        # openclaw keeps config health-state locks under /tmp regardless of TMPDIR.
        locks = Path("/tmp") / f"openclaw-state-locks-{os.getuid()}"
        locks.mkdir(parents=True, exist_ok=True)
        return [locks]

    def extra_read_paths(self) -> list[Path]:
        reads: list[Path] = []
        for part in self.command:
            path = Path(part).expanduser()
            if path.exists():
                root = install_root(str(path))
                if root:
                    reads.append(root)
                if path.suffix == ".mjs":
                    reads.append(path.parent)
        node = shutil.which("node")
        if node:
            root = install_root(node)
            if root:
                reads.append(root)
        return reads

    def version(self) -> str:
        if self._version is None:
            try:
                out = subprocess.run([*self.command, "--version"], capture_output=True, text=True,
                                     timeout=60, check=False, env={**self.environ, **self.extra_env()}
                                     if self._port else self.environ)
                text = (out.stdout or out.stderr).strip()
                self._version = text.splitlines()[0] if text else "unknown"
            except (OSError, subprocess.TimeoutExpired, AssertionError):
                self._version = "unknown"
        return self._version

    def new_session(self) -> None:
        """openclaw's own in-place session reset (``/reset``): the transcript is dropped; the
        session key, workspace, memory files and skills stay. The bridge runs on one fixed
        session key, so an ACP ``session/new`` would reattach to the same conversation."""
        self.ask("/reset")
        self._roll_usage()
        self._fresh_session = True

    def refine(self) -> None:
        """Ask the agent to distil the conversation into a skill (``/learn``),
        then apply every pending workshop proposal."""
        try:
            self.ask("/learn")
        except Exception as error:  # a failed refine must not cost the run
            self.note(f"/learn failed: {error}")
            return
        try:
            listing = self.run_cli([*self.command, "skills", "workshop", "list", "--json"])
            proposals = json.loads(listing.stdout or "[]")
        except (subprocess.TimeoutExpired, ValueError, OSError) as error:
            self.note(f"workshop list failed: {error}")
            return
        items = proposals if isinstance(proposals, list) else proposals.get("proposals", proposals.get("pending", []))
        for item in items or []:
            pid = item.get("id") if isinstance(item, dict) else None
            status = (item.get("status") or "pending") if isinstance(item, dict) else "pending"
            if not pid or status != "pending":
                continue
            result = self.run_cli([*self.command, "skills", "workshop", "apply", str(pid), "--json"])
            self.note(f"workshop apply {pid}: exit {result.returncode}")

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "memory_search": self.memory_search}
