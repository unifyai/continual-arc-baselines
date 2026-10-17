import json
import sys
from pathlib import Path

import pytest
from continual_arc.env import GiveUp, Observation, Submit

from continual_arc_baselines.systems.acp import ACPSystemLearner
from continual_arc_baselines.systems.base import SystemUnavailable

FAKE = Path(__file__).with_name("fake_acp_agent.py")


class FakeSystem(ACPSystemLearner):
    name = "fake"

    def default_command(self):
        return [sys.executable, str(FAKE)]

    def argv(self):
        return list(self.command)

    def prepare(self):
        (self.state_dir / "config.json").write_text("{}")

    def extra_env(self):
        return {"FAKE_ACP_STATE": str(self.state_dir / "agent"),
                "FAKE_ACP_REPLY": self.reply, "FAKE_ACP_SLEEP": "3",
                "PYTHONPATH": ":".join(sys.path)}

    def preserved(self):
        return ("config.json",)

    reply = json.dumps({"action": "submit", "grid": [[1]]})


def obs(step=0):
    return Observation(step=step, n_steps=3, task_id="t1", phase="stream", input=[[0]],
                       exposure_index=0, attempts_used=0, demo_requests_used=0, attempt_cap=8,
                       demo_cap=8, reset_before=False)


@pytest.fixture
def system(tmp_path):
    s = FakeSystem(model="fake-model", sandbox="none", startup_timeout_s=60, timeout_s=60)
    s.attach(tmp_path / "system", {})
    yield s
    s.close()


def test_turn_roundtrip_and_state_wipe(system, tmp_path):
    action = system.act(obs())
    assert isinstance(action, Submit) and action.grid == [[1]]
    state = tmp_path / "system" / "state"
    assert (state / "agent" / "prompt-1.txt").read_text().startswith("You are being evaluated")
    # a second turn does not repeat the rules and carries the pending feedback
    from continual_arc.env import SubmitFeedback
    system.observe(obs(), action, SubmitFeedback(correct=False, valid=True, attempts_used=1, failed=False))
    system.act(obs())
    second = (state / "agent" / "prompt-2.txt").read_text()
    assert "INCORRECT" in second and not second.startswith("You are being evaluated")
    system.reset()  # stops, wipes, restarts lazily
    assert (state / "config.json").exists()
    assert not (state / "agent").exists()
    assert isinstance(system.act(obs()), Submit)
    assert (state / "agent" / "prompt-1.txt").exists()
    usage = system.resources()
    assert usage["system_turns"] == 3.0 and usage["system_resets"] == 1.0
    # the agent reports session totals (100, 200 before the reset; 100 after): totals, not their sum
    assert usage["system_input_tokens"] == 300.0 and usage["system_output_tokens"] == 30.0


def test_idle_timeout_becomes_a_restart(tmp_path):
    s = FakeSystem(model="m", sandbox="none", timeout_s=60, idle_timeout_s=1.0, startup_timeout_s=60)
    s.attach(tmp_path / "system", {})
    try:
        s._ensure_started()
        with pytest.raises(SystemUnavailable):
            s.ask("SLEEP please")
        assert s.stats["idle_timeouts"] == 1
    finally:
        s.close()


def test_hard_timeout_is_a_budget_hit_not_a_restart(tmp_path):
    """A prompt that overruns its budget is cancelled and re-prompted with the budget notice,
    as prime-agent's is; the system is working, so it is not restarted."""
    s = FakeSystem(model="m", sandbox="none", timeout_s=1.0, idle_timeout_s=60, startup_timeout_s=60,
                   max_restarts=1, alias_tasks=False)
    s.attach(tmp_path / "system", {})
    try:
        s.reply = json.dumps({"action": "request_demos"})
        action = s.act(Observation(step=0, n_steps=1, task_id="SLEEP", phase="stream", input=[[0]],
                                   exposure_index=0, attempts_used=0, demo_requests_used=0,
                                   attempt_cap=8, demo_cap=8, reset_before=False))
        assert s.stats["budget_hits"] == 1 and s.stats["restarts"] == 0 and s.stats["forced_give_ups"] == 0
        assert not isinstance(action, GiveUp)
    finally:
        s.close()


# -- blank slate: no bundled skills, no channels -----------------------------

def test_hermes_starts_without_bundled_skills(tmp_path):
    from continual_arc_baselines.systems.hermes_agent import HermesAgentLearner
    s = HermesAgentLearner(model="m", sandbox="none")
    s.attach(tmp_path / "system", {})
    s.prepare()
    assert (s.hermes_home / ".no-bundled-skills").exists()
    config = json.loads((s.hermes_home / "config.yaml").read_text())
    # The post-turn memory and skill review stays at its shipped default (on): nothing overrides it.
    assert "background_review" not in (config.get("auxiliary") or {})


def test_openclaw_blocks_bundled_skills_and_channels(tmp_path):
    from continual_arc_baselines.systems.openclaw import OpenClawLearner
    s = OpenClawLearner(model="m", sandbox="none")
    s.attach(tmp_path / "system", {})
    s.prepare()
    config = json.loads(s.config_path.read_text())
    assert config["skills"]["allowBundled"] == ["__none__"]
    assert s.extra_env()["OPENCLAW_SKIP_CHANNELS"] == "1"


def test_openclaw_memory_search_is_set_only_when_asked(tmp_path):
    """Off (the grid cells of 27 and 28 Sep), memory search stays unset; on, it is keyword-only over the memory files
    and the session transcripts, and the run record says which."""
    from continual_arc_baselines.systems.openclaw import OpenClawLearner
    for on in (False, True):
        s = OpenClawLearner(model="m", sandbox="none", memory_search=on)
        s.attach(tmp_path / f"system-{on}", {})
        s.prepare()
        config = json.loads(s.config_path.read_text())
        assert s.describe()["memory_search"] is on
        if not on:
            assert "memory" not in config
            continue
        search = config["memory"]["search"]
        assert search["provider"] == "none" and search["experimental"]["sessionMemory"] is True
        assert search["sources"] == ["memory", "sessions"]


def test_openclaw_settles_a_run_its_dead_gateway_left_running(tmp_path):
    """A gateway killed mid-run leaves the bridge session running, which openclaw's restart recovery would
    replay at the next boot; prepare() closes that run as a clean stop leaves it, and leaves anything else."""
    import sqlite3

    from continual_arc_baselines.systems.openclaw import BRIDGE_KEY, OpenClawLearner
    s = OpenClawLearner(model="m", sandbox="none")
    s.attach(tmp_path / "system", {})
    db = s.openclaw_dir / "agents" / "main" / "agent" / "openclaw-agent.sqlite"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute("create table session_nodes (session_key text primary key, status text, entry_json text, updated_at integer)")
    stuck = {"sessionId": "s1", "status": "running", "abortedLastRun": False, "startedAt": 1000, "activeWriterRunId": "w",
             "lifecycleRunId": "r", "mainRestartRecovery": {"cycleId": "c", "chargedAttempts": 156}, "restartRecoveryRuns": [1]}
    con.execute("insert into session_nodes values (?, 'running', ?, 1)", (BRIDGE_KEY, json.dumps(stuck)))
    con.execute("insert into session_nodes values ('agent:main:cron:x', 'running', ?, 1)", (json.dumps({"status": "running"}),))
    con.commit()
    con.close()
    notes = []
    s.note = notes.append
    s.prepare()
    con = sqlite3.connect(db)
    status, entry, updated = con.execute("select status, entry_json, updated_at from session_nodes where session_key = ?", (BRIDGE_KEY,)).fetchone()
    entry = json.loads(entry)
    assert status == "done" and entry["status"] == "done" and entry["abortedLastRun"] is False and entry["endedAt"] >= 1000
    assert not {"mainRestartRecovery", "lifecycleRunId", "restartRecoveryRuns"} & set(entry)
    assert entry["activeWriterRunId"] == "w" and entry["sessionId"] == "s1"
    assert con.execute("select status from session_nodes where session_key = 'agent:main:cron:x'").fetchone()[0] == "running"
    con.close()
    assert notes and "156 attempts" in notes[0]
    s.prepare()                                     # a settled session is left as it is
    assert len(notes) == 1
    assert sqlite3.connect(db).execute("select updated_at from session_nodes where session_key = ?", (BRIDGE_KEY,)).fetchone()[0] == updated


def test_opencode_starts_without_plugins_or_external_skills(tmp_path):
    from continual_arc_baselines.systems.opencode import OpenCodeLearner
    s = OpenCodeLearner(model="m", sandbox="none")
    s.attach(tmp_path / "system", {})
    s.prepare()
    env = s.extra_env()
    assert env["OPENCODE_DISABLE_DEFAULT_PLUGINS"] == "1"
    # Both discovery switches stay off: enabling either makes opencode write into its read-only
    # install tree at start. Its private global config directory is what persists.
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1" and env["OPENCODE_DISABLE_EXTERNAL_SKILLS"] == "1"


def test_stateless_row_wipes_everything_between_instances(tmp_path):
    """Row H0 on an ACP system: a correct submission ends the instance; the next fresh
    observation stops the agent, deletes its state and starts it again, so the second
    instance's first prompt is prompt-1 again and carries the rules, and nothing the first
    instance wrote survives."""
    from continual_arc.env import SubmitFeedback

    s = FakeSystem(model="fake-model", sandbox="none", startup_timeout_s=60, timeout_s=60, stateless=True)
    s.attach(tmp_path / "system", {})
    try:
        state = tmp_path / "system" / "state"
        first = s.act(obs(step=0))
        (state / "agent" / "memory.md").write_text("what the system learned")   # something it persisted
        s.observe(obs(step=0), first, SubmitFeedback(correct=True, valid=True, attempts_used=0, failed=False))
        assert s._session_pending
        second = s.act(obs(step=1))
        assert isinstance(second, Submit)
        assert not (state / "agent" / "memory.md").exists()
        assert not (state / "agent" / "prompt-2.txt").exists()
        assert (state / "agent" / "prompt-1.txt").read_text().startswith("You are being evaluated")
        assert (state / "config.json").exists()   # the adapter's configuration, rewritten by prepare()
        usage = s.resources()
        assert usage["system_wipes"] == 1.0 and usage["system_resets"] == 1.0 and usage["system_sessions"] == 1.0
    finally:
        s.close()


def test_openclaw_npm_fails_fast():
    """The gateway's boot-time plugin install must not retry against a registry it cannot reach."""
    from continual_arc_baselines.systems.openclaw import OpenClawLearner
    learner = OpenClawLearner.__new__(OpenClawLearner)
    learner._port = 1
    learner._token = "t"
    learner.workdir = Path("/x")
    env = learner.extra_env()
    assert env["npm_config_fetch_retries"] == "0"
