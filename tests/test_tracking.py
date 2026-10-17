"""A harness row's usage per instance: every model call an instance made is in its record."""

from __future__ import annotations

import json
import logging
import threading
import time
from types import SimpleNamespace

from continual_arc.env import DemosFeedback, GiveUp, GiveUpFeedback, RequestDemos
from continual_arc.learner import BaseLearner
from continual_arc.runner import run
from continual_arc.schedule import stream_schedule
from continual_arc.tasks import TaskRegistry
from continual_arc.testing import FakeSource

from continual_arc_baselines.proxy import LLMProxy
from continual_arc_baselines.tracking import LLMCall, TrackedLearner, Tracker

from .fake_harness import FakeHarness, StoringHarness, late_reports
from .fake_upstream import FakeUpstream


class Alone(FakeHarness, BaseLearner):
    """A FakeHarness as the only user: one model call per action."""


class StoringAlone(StoringHarness, BaseLearner):
    """A StoringHarness as the only user: its storage step asks the model too."""


def run_alone(tmp_path, harness: type, instances: int, **tracked_args) -> tuple[list, FakeUpstream, list[LLMCall]]:
    """``harness`` served ``instances`` instances as the CLI serves a harness row: one proxy for the run and its tracker's
    counters in the learner's resources. Returns the records, the upstream and every call reported to the tracker, once
    the end of every call has been reported."""
    source = FakeSource(n_tasks=2)
    server = FakeUpstream().start()
    tracker = Tracker()
    calls: list[LLMCall] = []
    tracker.subscribe(calls.append)
    proxy = LLMProxy(server.base_url, tracker=tracker, capture_content=True).start()
    learner = harness(0)
    learner.attach(tmp_path / "system", proxy.environ(), tracker=tracker)
    try:
        result = run(TrackedLearner(learner, tracker, **tracked_args),
                     stream_schedule(source.task_ids(), instances, seed=0, source=source.name),
                     out_dir=tmp_path / "run", registry=TaskRegistry({source.name: source}))
        deadline = time.monotonic() + 10
        while sum(c.status == "started" for c in calls) > sum(c.status != "started" for c in calls) \
                and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        proxy.stop()
        server.stop()
    return result.records, server, calls


def test_every_call_an_instance_made_is_in_its_record_though_the_proxy_reports_it_after_the_reply(tmp_path, monkeypatch):
    # a call's tokens and cost are counted when the proxy reports its end, which it does once it has written the reply,
    # so the harness can take the reply and end the instance first; here every report comes 50 ms late (without the
    # wait at an instance's end, 3 of 12 records were short and 2 of the 14 calls in no record)
    late_reports(monkeypatch, 0.05)
    records, server, _ = run_alone(tmp_path, Alone, 12)
    assert len(records) == 12
    for r in records:  # FakeUpstream reports 3 output tokens a call
        assert r.resources["llm_output_tokens"] == 3 * r.resources["llm_requests"] >= 3, (r.step, r.resources)
    assert sum(r.resources["llm_requests"] for r in records) == len(server.requests)


def test_a_call_still_open_after_the_wait_is_left_out_of_the_record_and_logged(tmp_path, monkeypatch, caplog):
    # the storage step's call ends 1 s after its reply, and the learner waits 0.2 s for it
    late_reports(monkeypatch, 1.0, lambda call: "storage" in json.dumps(call.request_body))
    with caplog.at_level(logging.ERROR, logger="continual_arc_baselines.tracking"):
        records, _, calls = run_alone(tmp_path, StoringAlone, 1, calls_wait_s=0.2)
    [storage] = [c.call_id for c in calls if c.status == "started" and "storage" in json.dumps(c.request_body)]
    [record] = records  # a demonstration, the submission, then the storage step
    assert record.resources["llm_requests"] == 3 and record.resources["llm_output_tokens"] == 6
    [error] = [r.getMessage() for r in caplog.records if r.name == "continual_arc_baselines.tracking"]
    assert "1 model call(s) started for the instance still open" in error and storage[:8] in error


def test_only_an_instances_end_waits_and_only_for_the_calls_started_since_its_first_action(caplog):
    # a harness may leave a call running between actions (prime-agent compacts in the background); and a call open since
    # before the instance began (a harness's own work between instances, or one whose end never comes) would otherwise
    # hold up every later instance for the whole wait
    tracker = Tracker()
    tracker.emit(LLMCall(call_id="earlier", status="started", path="/v1/chat/completions"))

    class Harness(BaseLearner):
        acted = 0

        def act(self, obs):  # the first action leaves a call running, its end reported 0.5 s later
            self.acted += 1
            if self.acted > 1:
                return GiveUp()
            tracker.emit(LLMCall(call_id="mine", status="started", path="/v1/chat/completions"))
            ended = LLMCall(call_id="mine", status="completed", path="/v1/chat/completions", tokens={"output": 3})
            threading.Timer(0.5, tracker.emit, [ended]).start()
            return RequestDemos()

    learner = TrackedLearner(Harness(), tracker, calls_wait_s=5.0)
    obs = SimpleNamespace(step=0)
    started = time.monotonic()
    learner.observe(obs, learner.act(obs), DemosFeedback(pairs=[], refused=False, demo_requests_used=1))
    assert time.monotonic() - started < 0.4  # the instance goes on: no wait
    learner.observe(obs, learner.act(obs), GiveUpFeedback(charged=1.0))
    assert 0.5 <= time.monotonic() - started < 2.0 and learner.resources()["llm_output_tokens"] == 3
    assert not caplog.records
