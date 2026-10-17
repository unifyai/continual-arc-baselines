"""The finish protocol (a system's end="finish", every harness row since 27 Sep 2026, Dan): the rules state facts only,
the verdict comes alone, and the harness ends each instance itself with the finish action, refused until the instance is
over. A scripted system plays one instance of a two-pair task through the environment."""

from continual_arc.env import ContinualArcEnv, EnvConfig
from continual_arc.schedule import RANDOM, stream_schedule
from continual_arc.tasks import FixedArcSource, TaskRegistry

from continual_arc_baselines.systems import arc3_protocol, protocol
from continual_arc_baselines.systems.base import SystemLearner

TASKS = {"f": {"train": [{"input": [[1, 0]], "output": [[0, 1]]}, {"input": [[2, 0]], "output": [[0, 2]]}],
               "test": [{"input": [[3, 0]], "output": [[0, 3]]}]}}
CORRECT = '{"action": "submit", "grid": [[0, 3]]}'
FINISH = 'Stored what I learned.\n{"action": "finish"}'


class Scripted(SystemLearner):
    """Answers each message with the next scripted reply; ``cut`` holds the indices of turns cut for budget."""

    name = "scripted"

    def __init__(self, replies, cut=(), **kwargs):
        super().__init__(**kwargs)
        self.replies, self.cut, self.sent = list(replies), set(cut), []

    def start(self):
        pass

    def stop(self):
        pass

    def new_session(self):
        pass

    def wipe_state(self):
        pass

    def ask(self, message, *, reprompt=False):
        self.sent.append(message)
        self.last_turn_cut = len(self.sent) - 1 in self.cut
        return self.replies.pop(0) if self.replies else ""


def play(system, tmp_path):
    source = FixedArcSource("evaluation", tasks=TASKS, dataset="arc2", own_pairs_only=True, name="own")
    sched = stream_schedule(["f"], 1, seed=0, source=source.name, params=RANDOM, mode="random")
    env = ContinualArcEnv(sched, registry=TaskRegistry({source.name: source}), config=EnvConfig())
    system.attach(tmp_path / "system", {})
    obs = env.reset()
    while obs is not None:
        action = system.act(obs)
        result = env.step(action)
        system.observe(obs, action, result.feedback)
        obs = result.observation
    return env


def test_the_verdict_comes_alone_and_finish_ends_the_instance(tmp_path):
    s = Scripted([CORRECT, FINISH], end="finish")
    env = play(s, tmp_path)
    assert env.records[0].solved and env.records[0].cost == 0
    assert '{"action": "finish"}' in s.sent[0] and "worth keeping" not in s.sent[0]
    assert s.sent[1] == "Your submission was CORRECT. Instance solved with 0 wrong attempt(s)."
    assert len(s.sent) == 2 and s.stats["finishes"] == 1 and s.stats["closings"] == s.stats["storage_turns"] == 0


def test_finish_while_the_instance_can_be_played_is_refused_and_costs_nothing(tmp_path):
    s = Scripted(['{"action": "finish"}', CORRECT, FINISH], end="finish")
    env = play(s, tmp_path)
    assert s.sent[1].startswith("I could not find a valid action in your reply (finish is refused until the instance is over)")
    assert '{"action": "finish"}' in s.sent[1]   # the re-prompt lists every action of the protocol
    assert env.records[0].cost == 0 and s.stats["finish_refusals"] == 1 and s.stats["finishes"] == 1


def test_other_actions_after_the_verdict_are_refused_and_the_instance_then_ends(tmp_path):
    s = Scripted([CORRECT, CORRECT, '{"action": "request_demos"}', CORRECT], end="finish", max_reprompts=2)
    env = play(s, tmp_path)
    assert s.sent[2] == protocol.OVER_REPROMPT.format(reason="a submission is refused")
    assert s.sent[3] == protocol.OVER_REPROMPT.format(reason="a demo request is refused")
    assert len(s.sent) == 4 and s.stats["finish_reprompts"] == 2 and s.stats["finish_forced"] == 1
    assert env.records[0].solved and env.records[0].cost == 0   # nothing after the verdict changes the record


def test_a_turn_spent_after_the_verdict_ends_the_instance_without_a_notice(tmp_path):
    s = Scripted([CORRECT, "still saving..."], cut={1}, end="finish")
    play(s, tmp_path)
    assert len(s.sent) == 2 and s.stats["finish_budget_hits"] == 1 and s.stats["finishes"] == 0


def test_the_stateless_row_gets_the_same_end(tmp_path):
    s = Scripted([CORRECT, FINISH], end="finish", stateless=True)
    play(s, tmp_path)
    assert s.sent[1] == "Your submission was CORRECT. Instance solved with 0 wrong attempt(s)." and s.stats["finishes"] == 1


def test_the_other_ends_are_unchanged(tmp_path):
    closing = Scripted([CORRECT, "done"])
    play(closing, tmp_path / "c")
    assert closing.sent[1].endswith("No action is needed for it; reply with the single word: done.")
    assert closing.stats["closings"] == 1 and closing.end == "closing"
    storage = Scripted([CORRECT, "done"], storage_turn=True)
    play(storage, tmp_path / "s")
    assert "you may save anything" in storage.sent[1] and storage.end == "storage"


def test_the_rules_state_facts_only():
    for rules in (protocol.RULES_FINISH, arc3_protocol.RULES_FINISH):
        rules = " ".join(rules.split())   # the rules wrap at the page width
        assert '{"action": "finish"}' in rules and "finish is refused until the instance is over" in rules
        assert "Each instance is a separate conversation" in rules and "scored on its own" in rules
        for advice in ("worth keeping", "prefer", "nothing is shared", "keep exploring", "transformed copy"):
            assert advice not in rules, advice


def test_finish_parses_only_under_the_finish_protocol():
    for text in ('{"action": "finish"}', "done", 'Saved.\n{"action": "done"}', '```json\n{"action": "end_instance"}\n```'):
        assert protocol.parse_action(text, finish=True) == protocol.Finish()
    assert arc3_protocol.parse_action('{"action": "finish"}', finish=True) == protocol.Finish()
    for parse in (protocol.parse_action, arc3_protocol.parse_action):
        try:
            parse('{"action": "finish"}')
            raise AssertionError("finish parsed without the finish protocol")
        except (protocol.ParseError, arc3_protocol.ParseError):
            pass


def test_the_arc3_verdict_says_when_the_cap_ends_the_instance():
    from continual_arc.arc3.env import Act, ActFeedback

    text = arc3_protocol.render_feedback(Act(1), ActFeedback([], "NOT_FINISHED", False, 40, True, False, failed=True))
    assert "The action cap is reached; the instance is over." in text
    assert "instance is over" not in arc3_protocol.render_feedback(Act(1), ActFeedback([], "NOT_FINISHED", False, 3, True, False))


def test_the_ask_to_store_follows_the_verdict_only_in_row_ha(tmp_path):
    """Row HA (ask_to_store, 29 Sep): the same finish protocol and facts-only rules, with the ask after the verdict and
    the harness's own note on what it keeps; a stateless row is never asked, and the ask needs the finish protocol."""
    import pytest

    class Noted(Scripted):
        storage_note = "Entries without global_=True are lost."

    s = Noted([CORRECT, FINISH], end="finish", ask_to_store=True)
    play(s, tmp_path / "ha")
    verdict = "Your submission was CORRECT. Instance solved with 0 wrong attempt(s)."
    assert s.sent[1].startswith(verdict + "\n\nBefore you finish, you should liberally save *anything*")
    assert "using your own tools. Entries without global_=True are lost. There is no risk" in s.sent[1]
    assert s.sent[1].endswith('reply with {"action": "finish"}.')
    assert "liberally" not in s.sent[0]                     # the rules are the finish protocol's, unchanged
    assert s.stats["store_asks"] == 1 and s.stats["finishes"] == 1 and s.describe()["ask_to_store"] is True
    h0 = Noted([CORRECT, FINISH], end="finish", ask_to_store=True, stateless=True)
    play(h0, tmp_path / "h0")
    assert h0.sent[1] == verdict and h0.stats["store_asks"] == 0
    plain = Scripted([CORRECT, FINISH], end="finish")
    play(plain, tmp_path / "h")
    assert plain.sent[1] == verdict and plain.describe()["ask_to_store"] is False
    with pytest.raises(ValueError):
        Scripted([], end="storage", ask_to_store=True)


def test_the_program_library_row_states_the_procedure_and_asks_for_the_program(tmp_path):
    """Row HPL (program_library, 29 Sep): the finish protocol's rules gain row M's method as a procedure, with the
    library's folder, and the verdict is followed by the ask to save the task's program to its file; a stateless row
    gets the same messages; the option needs the finish protocol and does not go beside ask_to_store."""
    import re

    import pytest

    s = Scripted([CORRECT, FINISH], end="finish", program_library=True)
    play(s, tmp_path / "hpl")
    assert "How to work, in every instance (follow this exactly): keep a program library in the folder programs," in s.sent[0]
    assert "4. After the verdict, save the program that reproduced all the pairs" in s.sent[0]
    label = re.search(r"Task id: (task-[0-9a-f]+)", s.sent[0]).group(1)
    assert s.sent[1].startswith("Your submission was CORRECT. Instance solved with 0 wrong attempt(s).\n\nNow save your "
                                f"program for this task to programs/{label}.py:")
    assert s.sent[1].endswith('reply with {"action": "finish"}.')
    assert s.stats["library_asks"] == 1 and s.stats["store_asks"] == 0 and s.describe()["program_library"] is True
    h0 = Scripted([CORRECT, FINISH], end="finish", program_library=True, stateless=True)
    play(h0, tmp_path / "hpl0")
    assert "keep a program library" in h0.sent[0] and "Now save your program" in h0.sent[1]
    with pytest.raises(ValueError):
        Scripted([], end="finish", program_library=True, ask_to_store=True)
    with pytest.raises(ValueError):
        Scripted([], end="closing", program_library=True)
