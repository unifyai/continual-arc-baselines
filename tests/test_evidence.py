import json

import pytest
from continual_arc.env import EnvConfig
from continual_arc.runner import run
from continual_arc.schedule import stream_schedule
from continual_arc.testing import FakeSource, GiveUpLearner, ShiftMemoryLearner
from continual_workbench.context import create_context

from continual_arc_baselines.evidence import Evidence, adhoc_context, dependency_revisions
from continual_arc_baselines.testing import fake_registry


def events(context):
    return [json.loads(line) for line in (context.directory / "events.jsonl").read_text().splitlines()]


@pytest.fixture
def context(tmp_path):
    return create_context(tmp_path / "wb", "test-run", {"experiment_id": "t", "project_key": "t"})


def test_progress_and_finish_emit_metrics_and_artifacts(tmp_path, context):
    source = FakeSource(n_tasks=12)
    schedule = stream_schedule(source.task_ids(), 30, seed=1, source=source.name)
    config = EnvConfig()
    ev = Evidence(context, max_cost=config.max_cost)
    result = run(ShiftMemoryLearner(), schedule, config=config, out_dir=tmp_path / "run",
                 registry=fake_registry(), progress=ev.progress)
    emitted = ev.finish(result)

    metrics = [e["payload"] for e in events(context) if e["kind"] == "metric"]
    names = {m["name"] for m in metrics}
    assert {"instance/cost", "instance/solved", "running/mean_cost", "running/score",
            "instance/resource/calls", "summary/overall/score", "summary/stream/score"} <= names
    costs = [m for m in metrics if m["name"] == "instance/cost"]
    assert len(costs) == len(schedule) and [m["step"] for m in costs] == list(range(len(schedule)))
    # ShiftMemoryLearner solves everything after one demo request per task, so the
    # running score must climb and the final summary must match the run's own summary.
    running = [m["value"] for m in metrics if m["name"] == "running/score"]
    assert running[-1] > running[0]
    assert emitted["summary/overall/score"] == pytest.approx(result.summary["overall"]["score"])

    artifacts = [e["payload"] for e in events(context) if e["kind"] == "artifact"]
    assert {a["name"] for a in artifacts} == {"schedule.json", "config.json", "meta.json", "records.jsonl", "summary.json"}
    for a in artifacts:
        assert (context.directory / a["path"]).is_file()


def test_summary_metrics_skip_missing_phases(tmp_path, context):
    source = FakeSource(n_tasks=3)
    schedule = stream_schedule(source.task_ids(), 5, seed=0, source=source.name)  # no epilogue
    ev = Evidence(context)
    result = run(GiveUpLearner(), schedule, out_dir=tmp_path / "run", registry=fake_registry(),
                 progress=ev.progress)
    emitted = ev.finish(result)
    assert "summary/epilogue/score" not in emitted and "summary/stream/score" in emitted
    assert not any(m["name"] == "running/score" for m in
                   (e["payload"] for e in events(context) if e["kind"] == "metric"))


def test_adhoc_context_is_created_when_not_inside_workbench(tmp_path, monkeypatch):
    monkeypatch.delenv("WORKBENCH_RUN_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    ev = Evidence.from_environment(name="My Learner!")
    assert not ev.inside_workbench
    assert ev.directory.is_relative_to(tmp_path / ".workbench" / "runs")
    assert ev.context.run_id.startswith("adhoc-my-learner-")
    assert (ev.directory / "manifest.json").is_file()


def test_from_environment_uses_workbench_attempt(tmp_path, monkeypatch, context):
    monkeypatch.setenv("WORKBENCH_RUN_DIR", str(context.directory))
    ev = Evidence.from_environment()
    assert ev.inside_workbench and ev.directory == context.directory


def test_adhoc_slug_is_valid_for_workbench(tmp_path):
    ctx = adhoc_context("Stateless GPT-5 / v2", root=tmp_path)
    assert ctx.run_id.startswith("adhoc-stateless-gpt-5-v2-")


def test_dependency_revisions_name_the_installed_sources():
    revs = dependency_revisions()
    assert set(revs) == {"continual-arc"}
    for value in revs.values():
        assert value.startswith(("git+", "editable:")) or value[0].isdigit(), value
    assert dependency_revisions(("no-such-package",)) == {"no-such-package": "not-installed"}
