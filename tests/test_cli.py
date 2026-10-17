import json
from pathlib import Path

from continual_arc.learner import BaseLearner
from continual_arc.schedule import stream_schedule
from continual_arc.testing import FakeSource
from continual_workbench.context import create_context

from continual_arc_baselines.cli import main


def test_run_inside_workbench_attempt(tmp_path, monkeypatch, capsys):
    source = FakeSource(n_tasks=6)
    sched_path = tmp_path / "sched.json"
    stream_schedule(source.task_ids(), 20, seed=2, source=source.name).save(sched_path)
    context = create_context(tmp_path / "wb", "cli-run", {"experiment_id": "t", "project_key": "t"})
    monkeypatch.setenv("WORKBENCH_RUN_DIR", str(context.directory))

    code = main(["run", "--learner", "continual_arc.testing:ShiftMemoryLearner",
                 "--schedule", str(sched_path), "--no-proxy",
                 "--registry", "continual_arc_baselines.testing:fake_registry", "--quiet"])
    assert code == 0
    outcome = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert outcome["inside_workbench"] is True
    assert outcome["run_dir"] == str(context.directory / "continual-arc")
    assert (context.directory / "continual-arc" / "summary.json").is_file()
    assert outcome["overall_score"] > 0
    events = [json.loads(line) for line in (context.directory / "events.jsonl").read_text().splitlines()]
    assert {"parameters", "metric", "artifact", "continual_arc_summary"} <= {e["kind"] for e in events}
    params = next(e["payload"]["values"] for e in events if e["kind"] == "parameters")
    # Keys the workbench exports to MLflow must be present and scalar.
    assert params["method"] == "continual_arc.testing:ShiftMemoryLearner"
    assert params["protocol"] == "workstreams/workstreams" and params["seed"] == 2
    assert params["benchmark"] == "continual-arc" and params["model"] is None
    assert set(params["dependencies"]) == {"continual-arc"}
    assert params["protocol_version"].startswith("0.1.0+")
    assert params["proxy"] == {"enabled": False} and params["system"] == {}


def test_plain_learner_gets_proxy_counters_for_free(tmp_path, monkeypatch, capsys):
    """Any learner that builds a client from the environment is measured."""
    from .fake_upstream import FakeUpstream

    upstream = FakeUpstream().start()
    try:
        source = FakeSource(n_tasks=3)
        sched_path = tmp_path / "sched.json"
        stream_schedule(source.task_ids(), 6, seed=0, source=source.name).save(sched_path)
        monkeypatch.delenv("WORKBENCH_RUN_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        code = main(["run", "--learner", "tests.test_cli:EnvClientLearner",
                     "--schedule", str(sched_path), "--upstream", upstream.base_url,
                     "--registry", "continual_arc_baselines.testing:fake_registry", "--quiet"])
        assert code == 0
        outcome = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert outcome["inside_workbench"] is False and outcome["llm"]["llm_requests"] == 6
        records = [json.loads(line) for line in Path(outcome["run_dir"]).joinpath("records.jsonl").read_text().splitlines()]
        assert all(r["resources"]["llm_requests"] == 1 and r["resources"]["llm_cost_usd"] > 0 for r in records)
    finally:
        upstream.stop()


class EnvClientLearner(BaseLearner):
    """Calls the model once per instance through OPENAI_BASE_URL, then gives up."""

    def act(self, obs):
        import os

        import httpx
        from continual_arc.env import GiveUp

        httpx.post(os.environ["OPENAI_BASE_URL"] + "/chat/completions", timeout=10,
                   json={"model": "m", "messages": [{"role": "user", "content": obs.task_id}]})
        return GiveUp()



def test_run_requires_exactly_one_schedule_source(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        main(["run", "--learner", "continual_arc.testing:GiveUpLearner"])


def test_resume_continues_in_the_same_attempt(tmp_path, monkeypatch, capsys):
    import pytest

    source = FakeSource(n_tasks=6)
    sched_path = tmp_path / "sched.json"
    stream_schedule(source.task_ids(), 20, seed=2, source=source.name).save(sched_path)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "wb" / "stage1" / "cell"
    common = ["run", "--schedule", str(sched_path), "--no-proxy", "--registry",
              "continual_arc_baselines.testing:fake_registry", "--quiet", "--out", str(out), "--events"]
    with pytest.raises(RuntimeError, match="interrupted"):
        main(common + ["--learner", "continual_arc_baselines.testing:CrashAfterLearner", "--learner-kwargs", '{"n": 5}'])
    assert len((out / "records.jsonl").read_text().splitlines()) == 5
    attempt = json.loads((out / "attempt.json").read_text())["attempt_dir"]
    code = main(common + ["--learner", "continual_arc.testing:ShiftMemoryLearner", "--resume"])
    assert code == 0
    outcome = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert outcome["attempt_dir"] == attempt, "the resumed run keeps the attempt (and the system's state) of the first"
    assert len((out / "records.jsonl").read_text().splitlines()) == 20
    assert json.loads((out / "meta.json").read_text())["resumed_at"][0]["records"] == 5
    assert json.loads((out / "summary.json").read_text())["overall"]["n"] == 20
    # nothing to resume: a plain run in a fresh attempt
    code = main(common[:-3] + ["--out", str(tmp_path / "other"), "--learner", "continual_arc.testing:ShiftMemoryLearner", "--resume"])
    assert code == 0
