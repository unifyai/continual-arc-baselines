"""An empty reply under ``empty_reply="reprompt"`` (every launch since 28 Sep 2026, Dan): an unusable reply like any other,
re-prompted in the same conversation, and a restart only for a system that is dead, silent, or has answered nothing for
longer than one action's re-prompts. Cells launched before keep "restart". The contract every adapter is held to before
a launch: a turn with no text and a failed model call are each re-prompted, not restarted, and the turn log says how the
turn ended (``outcome``)."""

import json
import sqlite3
import tarfile
import time

from continual_arc.env import RequestDemos, Submit, SubmitFeedback

from continual_arc_baselines.proxy import LLMProxy
from continual_arc_baselines.systems.bare import BareModelLearner
from continual_arc_baselines.systems.base import SystemUnavailable
from continual_arc_baselines.systems.openclaw import BRIDGE_KEY, OpenClawLearner

from .fake_upstream import FakeUpstream
from .test_acp import FakeSystem, obs
from .test_finish import CORRECT, FINISH, Scripted, play
from .test_prime_agent import records_of, run_mode
from .test_prime_agent import upstream as upstream  # noqa: F401  (the fixture)

REPROMPT_START = "I could not find a valid action in your reply (empty reply)"


def turns_of(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# -- the rule, on a scripted system ------------------------------------------------------------------------------------

def test_an_empty_reply_is_reprompted_in_the_same_conversation(tmp_path):
    s = Scripted(["", CORRECT, FINISH], end="finish", empty_reply="reprompt")
    env = play(s, tmp_path)
    assert env.records[0].solved and env.records[0].cost == 0
    assert s.sent[1].startswith(REPROMPT_START)
    assert s.stats["restarts"] == 0 and s.stats["empty_replies"] == 1 and s.stats["reprompts"] == 1


def test_the_old_rule_still_restarts(tmp_path):
    s = Scripted(["", CORRECT, FINISH], end="finish")
    play(s, tmp_path)
    assert s.empty_reply == "restart" and s.stats["restarts"] == 1 and s.stats["reprompts"] == 0
    assert s.sent[1].startswith("You are being evaluated")   # the restart sent the rules and the observation again


def test_nothing_for_longer_than_an_actions_reprompts_is_silence(tmp_path):
    """max_reprompts + 2 empty replies in a row: the system is restarted, as a silent one is."""
    s = Scripted(["", "", "", "", CORRECT], empty_reply="reprompt", max_reprompts=2)
    s.attach(tmp_path / "system", {})
    replies = [s._ask_resilient(lambda fresh: "rules" if fresh else "again") for _ in range(4)]
    assert replies == ["", "", "", CORRECT]
    assert s.stats["restarts"] == 1 and s.stats["empty_replies"] == 4


def test_the_describe_records_the_rule():
    assert Scripted([], empty_reply="reprompt", archive_wipes=True).describe()["empty_reply"] == "reprompt"
    assert Scripted([]).describe()["archive_wipes"] is False


# -- ACP (hermes-agent, openclaw) --------------------------------------------------------------------------------------

class EmptyFirst(FakeSystem):
    empty_first = "1"

    def extra_env(self):
        return {**super().extra_env(), "FAKE_ACP_EMPTY": self.empty_first}

    def transcript_paths(self):
        return [self.state_dir / "agent"]


def test_acp_a_turn_with_no_text_is_reprompted_and_logged(tmp_path):
    s = EmptyFirst(model="fake-model", sandbox="none", startup_timeout_s=60, timeout_s=60, empty_reply="reprompt")
    s.attach(tmp_path / "system", {})
    try:
        action = s.act(obs())
        assert isinstance(action, Submit)
        assert s.stats["restarts"] == 0 and s.stats["empty_replies"] == 1
    finally:
        s.close()
    turns = turns_of(tmp_path / "system" / "turns.jsonl")
    assert turns[0]["reply"] == "" and turns[0]["outcome"]["stop_reason"] == "end_turn"
    assert turns[1]["kind"] == "reprompt" and turns[1]["message"].startswith(REPROMPT_START)
    assert turns[1]["outcome"]["stop_reason"] == "end_turn" and not turns[1]["fresh"]


def test_acp_the_old_rule_restarts(tmp_path):
    s = EmptyFirst(model="fake-model", sandbox="none", startup_timeout_s=60, timeout_s=60)
    s.attach(tmp_path / "system", {})
    try:
        assert isinstance(s.act(obs()), Submit)
        assert s.stats["restarts"] == 1
    finally:
        s.close()


def test_a_stateless_row_archives_the_transcripts_before_the_wipe(tmp_path):
    s = EmptyFirst(model="fake-model", sandbox="none", startup_timeout_s=60, timeout_s=60, stateless=True,
                   empty_reply="reprompt", archive_wipes=True)
    s.empty_first = "0"
    s.attach(tmp_path / "system", {})
    try:
        first = s.act(obs(step=0))
        s.observe(obs(step=0), first, SubmitFeedback(correct=True, valid=True, attempts_used=0, failed=False))
        s.act(obs(step=1))   # the wipe before the second instance
    finally:
        s.close()
    archive = tmp_path / "system" / "wipe-archive" / "instance-0000.tar.gz"
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert "state/agent/prompt-1.txt" in names
    assert not (tmp_path / "system" / "state" / "agent" / "prompt-2.txt").exists()   # the wipe still happened


def test_openclaw_reads_how_its_run_ended_from_its_own_database(tmp_path):
    """openclaw's bridge ends a failed run as an ordinary end_turn; the adapter asks openclaw's agent database."""
    s = OpenClawLearner(model="m")
    s.attach(tmp_path / "system", {})
    db = s.agent_db
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute("create table session_nodes (session_key text primary key, current_session_id text)")
    con.execute("create table transcript_events (session_id text, seq integer, event_json text, created_at integer)")
    con.execute("create table session_windows (session_id text, session_key text, created_at integer)")
    con.execute("insert into session_nodes values (?, ?)", (BRIDGE_KEY, "s1"))
    started = time.time()
    events = [{"type": "message", "message": {"role": "user", "content": "observation"}},
              {"type": "message", "message": {"role": "assistant", "content": [{"type": "toolCall", "name": "exec"}],
                                              "stopReason": "toolUse"}},
              {"type": "message", "message": {"role": "assistant", "content": [], "stopReason": "error",
                                              "errorMessage": "Provider returned an incomplete or malformed tool call"}}]
    for seq, event in enumerate(events):
        con.execute("insert into transcript_events values (?, ?, ?, ?)", ("s1", seq, json.dumps(event), 0))
    ms = int(started * 1000)
    con.executemany("insert into session_windows values (?, ?, ?)",
                    [("a", "agent:main:subagent:1", ms + 10), ("b", "agent:main:subagent:2", ms + 20),
                     ("c", "agent:main:subagent:0", ms - 60_000), ("d", BRIDGE_KEY, ms + 30)])
    con.commit()
    con.close()
    out = s.turn_diagnostics(started)
    assert out == {"harness_stop_reason": "error", "subagent_sessions": 2,
                   "harness_error": "Provider returned an incomplete or malformed tool call"}
    assert s.turn_diagnostics(started + 3600)["subagent_sessions"] == 0
    assert [p.name for p in s.transcript_paths()] == ["agents", "state", "gateway.log"]


# -- prime-agent -------------------------------------------------------------------------------------------------------

def test_prime_agent_a_failed_model_call_is_reprompted_not_restarted(tmp_path, monkeypatch, upstream):  # noqa: F811
    """The model call fails and prime-agent gives up on it (provider_retries 0): a turn with no text."""
    monkeypatch.setenv("FAKE_AGENT_PROVIDER_DOWN", "1")
    context, _ = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                          {"quiet_probe_s": 0.3, "provider_retries": 0, "empty_reply": "reprompt"})
    records = records_of(context)
    assert all(r["solved"] for r in records)
    assert sum(r["resources"]["system_restarts"] for r in records) == 0
    assert sum(r["resources"]["system_empty_replies"] for r in records) == 1
    turns = turns_of(context.directory / "system" / "turns.jsonl")
    empty = [t for t in turns if not t["reply"].strip()]
    assert empty[0]["outcome"]["stop_reason"] == "provider_error"
    after = turns[turns.index(empty[0]) + 1]
    assert after["kind"] == "reprompt" and after["message"].startswith(REPROMPT_START)


def test_prime_agent_the_old_rule_restarts(tmp_path, monkeypatch, upstream):  # noqa: F811
    monkeypatch.setenv("FAKE_AGENT_PROVIDER_DOWN", "1")
    context, _ = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"quiet_probe_s": 0.3, "provider_retries": 0})
    assert sum(r["resources"]["system_restarts"] for r in records_of(context)) == 1


# -- the bare model (rows F and M) -------------------------------------------------------------------------------------

def _bare(tmp_path, server, **kwargs):
    proxy = LLMProxy(server.base_url, api_key="sk-real").start()
    bare = BareModelLearner("vendor/model", max_tokens=1000, **kwargs)
    bare.attach(tmp_path / "system", proxy.environ())
    return bare, proxy


def test_bare_an_empty_reply_is_reprompted_in_one_conversation(tmp_path):
    server = FakeUpstream(replies=["", '{"action": "request_demos"}']).start()
    bare, proxy = _bare(tmp_path, server, empty_reply="reprompt")
    try:
        assert isinstance(bare.act(obs()), RequestDemos)
        assert bare.stats["restarts"] == 0 and bare.stats["empty_replies"] == 1
    finally:
        bare.close()
        proxy.stop()
        server.stop()
    second = server.requests[1]["body"]["messages"]
    # the unanswered observation and the re-prompt are one user turn: no empty assistant message is sent
    assert [m["role"] for m in second] == ["user"]
    assert "You are being evaluated" in second[0]["content"] and REPROMPT_START in second[0]["content"]
    turns = turns_of(tmp_path / "system" / "turns.jsonl")
    assert turns[0]["outcome"] == {"finish_reason": "stop"}


def test_bare_a_failed_model_call_is_reprompted_not_restarted(tmp_path):
    server = FakeUpstream(status=[400, 200], reply='{"action": "request_demos"}',
                          error={"error": {"message": "context too long"}}).start()
    bare, proxy = _bare(tmp_path, server, empty_reply="reprompt")
    try:
        assert isinstance(bare.act(obs()), RequestDemos)
        assert bare.stats["restarts"] == 0 and bare.stats["empty_replies"] == 1
    finally:
        bare.close()
        proxy.stop()
        server.stop()
    turns = turns_of(tmp_path / "system" / "turns.jsonl")
    assert turns[0]["reply"] == "" and turns[0]["outcome"]["error"].startswith("HTTP 400")
    assert turns[1]["kind"] == "reprompt"


def test_bare_the_old_rule_restarts_on_a_failed_call(tmp_path):
    server = FakeUpstream(status=[400, 200], reply='{"action": "request_demos"}').start()
    bare, proxy = _bare(tmp_path, server)
    try:
        assert isinstance(bare.act(obs()), RequestDemos)
        assert bare.stats["restarts"] == 1
    finally:
        bare.close()
        proxy.stop()
        server.stop()


def test_bare_a_silent_upstream_is_still_a_dead_system(tmp_path):
    bare = BareModelLearner("vendor/model", empty_reply="reprompt", timeout_s=1)
    bare.attach(tmp_path / "system", {"ARC_LLM_BASE_URL": "http://127.0.0.1:9/v1"})   # nothing listens there
    bare.start()
    try:
        try:
            bare.ask("hello")
            raise AssertionError("an unreachable proxy was not taken for a dead system")
        except SystemUnavailable:
            pass
    finally:
        bare.stop()
