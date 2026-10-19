"""The bare model behind the harness protocol (systems/bare.py): one conversation per instance, one streamed call per
message through the proxy, the action parsed as a harness's is, and no closing turn."""

import json

from continual_arc.env import (
    DemosFeedback,
    LearnerObservation,
    RequestDemos,
    Submit,
    SubmitFeedback,
)

from continual_arc_baselines.proxy import LLMProxy
from continual_arc_baselines.systems.bare import BareModelLearner

from .fake_upstream import FakeUpstream


def obs(step=0, demos=0, attempts=0, task="t1"):
    return LearnerObservation(step=step, task_id=task, input=[[0]], attempts_used=attempts, demo_requests_used=demos,
                              attempt_cap=8, demo_cap=1)


def test_the_bare_model_plays_the_protocol_in_one_conversation_per_instance(tmp_path):
    server = FakeUpstream(reply='I need the rule first.\n{"action": "request_demos"}').start()
    try:
        with LLMProxy(server.base_url, api_key="sk-real") as proxy:
            bare = BareModelLearner("vendor/model", max_tokens=1000)
            bare.attach(tmp_path / "system", proxy.environ())
            first = bare.act(obs())
            assert isinstance(first, RequestDemos)
            bare.observe(obs(), first, DemosFeedback(pairs=[], refused=False, demo_requests_used=1))
            server.reply = '{"action": "submit", "grid": [[1]]}'
            second = bare.act(obs(demos=1))
            assert isinstance(second, Submit) and second.grid == [[1]]
            bare.observe(obs(demos=1), second, SubmitFeedback(correct=True, valid=True, attempts_used=1, failed=False))
            # the next instance opens a fresh conversation, with the rules again
            server.reply = '{"action": "request_demos"}'
            bare.act(obs(step=1, task="t2"))
            bare.close()
    finally:
        server.stop()
    bodies = [r["body"] for r in server.requests]
    assert len(bodies) == 3  # no closing turn after the verdict
    assert all(b["stream"] and b["max_tokens"] == 1000 and b["model"] == "vendor/model" for b in bodies)
    assert all(r["authorization"] == "Bearer sk-real" for r in server.requests)  # the proxy's key upstream
    assert [m["role"] for m in bodies[1]["messages"]] == ["user", "assistant", "user"]
    assert bodies[1]["messages"][1]["content"].endswith('{"action": "request_demos"}')
    assert "You are being evaluated" in bodies[0]["messages"][0]["content"]
    assert [m["role"] for m in bodies[2]["messages"]] == ["user"]
    assert "You are being evaluated" in bodies[2]["messages"][0]["content"]
    turns = [json.loads(line) for line in (tmp_path / "system" / "turns.jsonl").read_text().splitlines()]
    assert [t["kind"] for t in turns] == ["observation"] * 3


def _pair(a, b):
    from continual_arc.grids import Grid
    from continual_arc.tasks import Pair
    return Pair(Grid([[a]]), Grid([[b]]))


def _play(learner, o, replies, server, feedbacks):
    """One instance: each reply answers one message; each feedback follows the action it answers."""
    for reply, feedback in zip(replies, feedbacks, strict=True):
        server.reply = reply
        action = learner.act(o)
        learner.observe(o, action, feedback)
        o = obs(step=o.step, task=o.task_id, demos=o.demo_requests_used + isinstance(action, RequestDemos),
                attempts=o.attempts_used)
    return o


def test_the_demo_cache_floor_shows_the_pairs_it_bought_when_the_task_returns(tmp_path):
    from continual_arc_baselines.systems.bare import BareDemoCacheLearner

    server = FakeUpstream(reply='{"action": "request_demos"}').start()
    request, submit = '{"action": "request_demos"}', '{"action": "submit", "grid": [[1]]}'
    solved = SubmitFeedback(correct=True, valid=True, attempts_used=0, failed=False)
    try:
        with LLMProxy(server.base_url, api_key="sk-real") as proxy:
            cache = BareDemoCacheLearner("vendor/model", max_tokens=1000, end="finish")
            cache.attach(tmp_path / "system", proxy.environ())
            _play(cache, obs(step=0, task="t1"), [request, submit], server,
                  [DemosFeedback(pairs=[_pair(3, 4)], refused=False, demo_requests_used=1), solved])
            _play(cache, obs(step=1, task="t2"), [submit], server, [solved])
            _play(cache, obs(step=2, task="t1"), [submit], server, [solved])
            cache.close()
    finally:
        server.stop()
    firsts = [r["body"]["messages"][0]["content"] for r in server.requests if len(r["body"]["messages"]) == 1]
    assert len(server.requests) == 4 and len(firsts) == 3   # no verdict turn
    assert "Demonstration pairs of task" not in firsts[0] and "Demonstration pairs of task" not in firsts[1]
    shown = firsts[2]
    assert "that you received before, shown again at no cost" in shown
    assert "Earlier pair 1 input (1x1):\n3\nEarlier pair 1 output (1x1):\n4" in shown
    assert shown.index("Earlier pair 1") < shown.index("New instance. Task id:")   # before the instance
    assert "finish is refused until the instance is over" in " ".join(shown.split())   # the harness rows' rules
    assert cache.resources()["state_bytes"] > 0


def test_the_demo_cache_keeps_at_most_its_cap_and_is_wiped_with_the_state():
    from continual_arc_baselines.systems.bare import BareDemoCacheLearner

    cache = BareDemoCacheLearner("vendor/model", cache_cap=2)
    o = obs(task="t1")
    for k in range(3):
        cache.observe(o, RequestDemos(), DemosFeedback(pairs=[_pair(k, k)], refused=False, demo_requests_used=k + 1))
    cache.observe(o, RequestDemos(), DemosFeedback(pairs=[], refused=True, demo_requests_used=4))
    assert [p.input for p in cache.cache["t1"]] == [[[0]], [[1]]]
    assert cache.instance_preface(obs(task="t2")) == ""
    cache.wipe_state()
    assert cache.cache == {} and cache.instance_preface(o) == ""
