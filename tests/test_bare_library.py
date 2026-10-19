"""Row M since 28 Sep (systems/bare.py, BareProgramLibraryLearner): the bare demo-cache floor with a program for every
submission, checked at no cost on the task's pairs, kept, and run first when the task returns. The model is scripted
(``ask`` answers from a list); programs run unsandboxed here, as in test_floors.py."""

from continual_arc.env import (
    DemosFeedback,
    LearnerObservation,
    RequestDemos,
    Submit,
    SubmitFeedback,
)
from continual_arc.grids import Grid
from continual_arc.tasks import Pair

from continual_arc_baselines.systems.bare import PROGRAM_NOTE, BareProgramLibraryLearner

FLIP = '```python\ndef transform(grid):\n    return [row[::-1] for row in grid]\n```\n{"action": "submit"}'
SAME = '```python\ndef transform(grid):\n    return grid\n```\n{"action": "submit"}'
REQUEST = '{"action": "request_demos"}'
PAIR = Pair(Grid([[3, 4]]), Grid([[4, 3]]))
RIGHT = SubmitFeedback(correct=True, valid=True, attempts_used=0, failed=False)
WRONG = SubmitFeedback(correct=False, valid=True, attempts_used=1, failed=False)


class Scripted(BareProgramLibraryLearner):
    def __init__(self, replies, **kwargs):
        super().__init__("vendor/model", sandbox="none", **kwargs)
        self.replies, self.sent = list(replies), []

    def start(self):
        self.messages = []

    def stop(self):
        self.messages = []

    def ask(self, message, *, reprompt=False):
        self.sent.append(message)
        return self.replies.pop(0)


def obs(step, task="t1", grid=((1, 2),), demos=0, attempts=0):
    return LearnerObservation(step=step, task_id=task, input=[list(r) for r in grid], attempts_used=attempts,
                              demo_requests_used=demos, attempt_cap=8, demo_cap=8)


def test_a_program_is_checked_free_kept_and_run_first_when_the_task_returns(tmp_path):
    s = Scripted([REQUEST, SAME, FLIP])
    s.attach(tmp_path / "system", {})
    o = obs(0)
    a = s.act(o)
    assert isinstance(a, RequestDemos)
    assert PROGRAM_NOTE.splitlines()[0] in s.sent[0]
    assert "finish is refused until the instance is over" in " ".join(s.sent[0].split())
    s.observe(o, a, DemosFeedback(pairs=[PAIR], refused=False, demo_requests_used=1))
    o = obs(0, demos=1)
    a = s.act(o)
    assert a == Submit([[2, 1]])   # the corrected program's output on the test input
    assert "checked on demo pair 1 at no cost and its output is wrong" in s.sent[2]
    assert "Failed checks in this instance: 1 of 3" in s.sent[2]
    s.observe(o, a, RIGHT)
    assert "[::-1]" in s.library["t1"] and s.stats["programs_kept"] == 1
    o = obs(1, grid=((5, 6),))   # the task returns: the kept program answers first, with no model call
    a = s.act(o)
    assert a == Submit([[6, 5]]) and len(s.sent) == 3
    s.observe(o, a, RIGHT)
    assert s.stats["library_runs"] == 1 and s.stats["library_hits"] == 1


def test_a_wrong_kept_program_costs_one_attempt_and_the_model_sees_it(tmp_path):
    s = Scripted([FLIP, SAME])
    s.attach(tmp_path / "system", {})
    o = obs(0)
    a = s.act(o)   # no pair requested: the program is kept on none, the model's choice
    s.observe(o, a, RIGHT)
    o = obs(1, grid=((7, 7, 8),))
    a = s.act(o)
    assert a == Submit([[8, 7, 7]]) and len(s.sent) == 1
    s.observe(o, a, WRONG)
    o = obs(1, grid=((7, 7, 8),), attempts=1)
    a = s.act(o)
    first = s.sent[1]
    assert "Your submission was INCORRECT" in first
    assert "The program kept for task" in first and "was run on this test input first" in first and "[::-1]" in first
    assert first.index("The program kept for task") < first.index("Same instance. Task id:")
    assert a == Submit([[7, 7, 8]]) and s.stats["library_misses"] == 1


def test_a_grid_alone_is_re_prompted_and_failed_checks_run_out(tmp_path):
    s = Scripted(['{"action": "submit", "grid": [[1]]}', FLIP])
    s.attach(tmp_path / "system", {})
    a = s.act(obs(0))
    assert "not a grid" in s.sent[1] and a == Submit([[2, 1]]) and s.stats["grid_refusals"] == 1

    s = Scripted([REQUEST] + [SAME] * 4)
    s.attach(tmp_path / "system2", {})
    o = obs(0)
    s.observe(o, s.act(o), DemosFeedback(pairs=[PAIR], refused=False, demo_requests_used=1))
    a = s.act(obs(0, demos=1))
    assert a == Submit([[1, 2]])   # after three failed checks, the output is submitted without one
    assert len(s.sent) == 5 and s.stats["checks_failed"] == 3 and "t1" not in s.library


def test_the_library_and_the_cache_are_wiped_with_the_state(tmp_path):
    s = Scripted([FLIP])
    s.attach(tmp_path / "system", {})
    o = obs(0)
    s.observe(o, s.act(o), RIGHT)
    s.cache["t1"] = [PAIR]
    s.wipe_state()
    assert s.library == {} and s.cache == {}


def test_the_stateless_library_keeps_nothing_and_does_not_say_it_keeps(tmp_path):
    """Row M0 (stateless=True, 29 Sep): every submission a program, checked as in M; before every instance the kept
    programs and the demo cache are wiped, so a returning task gets no kept program and no cached pairs, and the rules
    drop the sentence about keeping."""
    from continual_arc_baselines.systems.bare import PROGRAM_NOTE_STATELESS

    s = Scripted([REQUEST, FLIP, FLIP], stateless=True)
    s.attach(tmp_path / "system", {})
    o = obs(0)
    a = s.act(o)
    assert isinstance(a, RequestDemos)
    assert PROGRAM_NOTE_STATELESS in s.sent[0] and "is kept for the task" not in s.sent[0]
    s.observe(o, a, DemosFeedback(pairs=[PAIR], refused=False, demo_requests_used=1))
    o = obs(0, demos=1)
    a = s.act(o)
    assert a == Submit([[2, 1]])
    s.observe(o, a, RIGHT)
    o = obs(1, grid=((5, 6),))   # the task returns: nothing kept, so the model is asked again, with no cached pairs
    a = s.act(o)
    assert a == Submit([[6, 5]]) and len(s.sent) == 3
    assert "is kept for the task" not in s.sent[2] and "shown again at no cost" not in s.sent[2]
    assert s.stats["library_runs"] == 0 and s.stats["wipes"] >= 1 and not s.cache.get("t1")
    m = Scripted([FLIP])
    m.attach(tmp_path / "m", {})
    m.act(obs(0))
    assert PROGRAM_NOTE in m.sent[0]   # row M's rules are unchanged
