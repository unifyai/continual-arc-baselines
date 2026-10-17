"""The prompting floors against a scripted model and the data-free FakeSource."""

from __future__ import annotations

import json

import pytest
from continual_arc.configs import build_schedule
from continual_arc.runner import run
from continual_arc.tasks import TaskRegistry
from continual_arc.testing import FakeSource

from continual_arc_baselines.learners.chat import ChatClient
from continual_arc_baselines.learners.prompting import (
    DemoCacheLearner,
    ProgramLibraryLearner,
    StatelessLearner,
    extract_grid,
)


def shift_of(prompt: str) -> int | None:
    """Recover the FakeSource rule (output = input + s mod 10) from the first demo pair."""
    lines = prompt.splitlines()
    try:
        i = next(k for k, line in enumerate(lines) if line.startswith("Demo pair 1 input"))
        j = next(k for k, line in enumerate(lines) if line.startswith("Demo pair 1 output"))
    except StopIteration:
        return None
    a = int(lines[i + 1].split()[0])
    b = int(lines[j + 1].split()[0])
    return (b - a) % 10


def _test_input_of(prompt: str):
    lines = prompt.splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("Test input"))
    rows = []
    for line in lines[i + 1:]:
        if not line or not line[0].isdigit():
            break
        rows.append([int(c) for c in line.split()])
    return rows


class Oracle:
    """Answers correctly when demos are shown, wrongly otherwise; writes correct programs."""

    def __init__(self):
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, body: dict) -> dict:
        self.calls += 1
        system = body["messages"][0]["content"]
        user = body["messages"][-1]["content"]
        self.prompts.append(user)
        if "writing programs" in system:
            s = shift_of(user) or 0
            code = f"```python\ndef transform(grid):\n    return [[(c + {s}) % 10 for c in row] for row in grid]\n```"
            return {"choices": [{"message": {"content": code}}]}
        s = shift_of(user)
        grid = _test_input_of(user)
        if s is None:
            out = [[(c + 1) % 10 for c in row] for row in grid]  # a guess, usually wrong
        else:
            out = [[(c + s) % 10 for c in row] for row in grid]
        return {"choices": [{"message": {"content": "Sure.\n" + json.dumps({"grid": out})}}]}


@pytest.fixture
def world():
    source = FakeSource(n_tasks=16)
    reg = TaskRegistry({source.name: source})
    pool = source.task_ids()
    sched = build_schedule("arc1-smoke", "workstreams", 0, rearc_task_ids=pool[:8], eval_task_ids=pool[8:])
    for tid in sched.sources:
        sched.sources[tid] = source.name
    return reg, sched


def make(cls, oracle, **kw):
    return cls(model="fake", chat=ChatClient("fake", send=oracle), alias_tasks=True, **kw)


def test_extract_grid_variants():
    assert extract_grid('text {"grid": [[1, 2], [3, 4]]}') == [[1, 2], [3, 4]]
    assert extract_grid("```json\n{\"output\": [[5]]}\n```") == [[5]]
    assert extract_grid("here: [[1,2],[3,4]] done") == [[1, 2], [3, 4]]
    assert extract_grid("nothing") is None


def test_stateless_pays_for_demos_every_time(world):
    reg, sched = world
    oracle = Oracle()
    result = run(make(StatelessLearner, oracle), sched, registry=reg)
    stream = [r for r in result.records if r.phase != "epilogue"]
    # one pair per request since 26 Sep: every instance asks for all eight before answering
    assert all(r.demo_requests == 8 for r in stream)
    assert all(r.solved for r in stream)
    assert all(r.cost == 8.0 for r in stream)
    # the model saw all eight, numbered across the requests
    assert all("Demo pair 8 input" in p and "Demo pair 9" not in p for p in oracle.prompts)


def test_demo_cache_pays_once_per_task(world):
    reg, sched = world
    oracle = Oracle()
    learner = make(DemoCacheLearner, oracle)
    result = run(learner, sched, registry=reg)
    stream = [r for r in result.records if r.phase != "epilogue"]
    firsts = [r for r in stream if r.exposure_index == 0]
    later = [r for r in stream if r.exposure_index > 0]
    assert all(r.demo_requests == 8 for r in firsts)
    assert all(r.demo_requests == 0 and r.cost == 0.0 and r.solved for r in later)
    assert all(len(pairs) == 8 for pairs in learner.cache.values())
    assert learner.resources()["state_bytes"] > 0


def test_program_library_reuses_programs_without_calls(world):
    reg, sched = world
    oracle = Oracle()
    learner = make(ProgramLibraryLearner, oracle, sandbox="none")
    result = run(learner, sched, registry=reg)
    stream = [r for r in result.records if r.phase != "epilogue"]
    later = [r for r in stream if r.exposure_index > 0]
    assert all(r.solved and r.cost == 0.0 for r in later)
    n_tasks = len({r.task_id for r in result.records})  # stream and epilogue tasks alike get a program
    assert learner.stats["programs_kept"] == n_tasks
    # one induction call per task, no calls on recurrence
    assert oracle.calls == n_tasks
    assert all(r.demo_requests == 8 for r in stream if r.exposure_index == 0)
