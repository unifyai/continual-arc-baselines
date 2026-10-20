import json

import pytest
from continual_arc.env import GiveUp, LearnerObservation, RequestDemos, Submit, SubmitFeedback
from continual_arc.tasks import Pair

from continual_arc_baselines.systems.protocol import (
    RULES,
    ParseError,
    Transcript,
    parse_action,
    render_feedback,
    render_observation,
    render_pairs,
)


def obs(**over):
    base = dict(step=3, task_id="task-0f1a2b3c", input=[[1, 2], [3, 4]], attempts_used=0, demo_requests_used=0,
                attempt_cap=8, demo_cap=8)
    return LearnerObservation(**{**base, **over})


def test_parse_finds_last_action_anywhere():
    text = 'Thinking... {"action": "request_demos"}\nActually:\n```json\n{"action": "submit", "grid": [[1]]}\n```\n'
    assert parse_action(text) == Submit([[1]])
    for text in ('blah {"action":"give_up"} done', '{"action": "skip"}'):
        try:
            parse_action(text)
        except ParseError:
            continue
        raise AssertionError(text)
    assert parse_action('{"action": "Request-Demos"}') == RequestDemos()


def test_parse_errors():
    with pytest.raises(ParseError):
        parse_action("no json here")
    with pytest.raises(ParseError):
        parse_action('{"action": "dance"}')
    # An invalid grid is still a submission: the environment charges it.
    assert parse_action('{"action": "submit", "grid": "nope"}') == Submit("nope")


def test_render_first_message_has_rules_then_instance():
    text = render_observation(obs(), pending=[], demo_cost=1.0, max_cost=16.0, first_in_session=True)
    assert "Continual-ARC" in text and "Task id: task-0f1a2b3c" in text
    assert "one that ends unsolved costs the whole budget, 16," in text
    assert "1 2\n3 4" in text and "New instance" in text
    later = render_observation(obs(attempts_used=1), pending=["Your submission was INCORRECT."],
                               demo_cost=1.0, max_cost=16.0, first_in_session=False)
    assert "Continual-ARC" not in later and "Same instance" in later
    assert "Feedback since your last action" in later


def test_feedback_and_pairs_render():
    fb = render_feedback(Submit([[1]]), SubmitFeedback(correct=False, valid=False, attempts_used=2, failed=True))
    assert "INCORRECT" in fb and "not a valid" in fb and "cap is reached" in fb
    pairs = render_pairs([Pair([[1]], [[2]])])
    assert "Demo pair 1 input (1x1):\n1\nDemo pair 1 output (1x1):\n2" == pairs
    t = Transcript()
    t.add(GiveUp(), SubmitFeedback(correct=True, valid=True, attempts_used=0, failed=False))
    assert t.take() and t.take() == []


def test_parse_accepts_synonyms_and_skips_unknown_candidates():
    assert parse_action('{"action": "answer", "answer": [[1, 2]]}') == Submit([[1, 2]])
    assert parse_action('{"action": "output", "output": [[3]]}') == Submit([[3]])
    # An unknown action earlier in the text does not hide a valid one.
    text = '{"action": "dance"} ... {"action": "request_demos"}'
    assert parse_action(text) == RequestDemos()
    with pytest.raises(ParseError, match="dance"):
        parse_action('{"action": "dance"}')


def test_parse_reads_bare_grid_and_unwraps_an_envelope():
    from continual_arc_baselines.systems.protocol import RequestDemos, Submit, parse_action

    assert parse_action('{"grid": [[1, 2], [3, 4]]}') == Submit([[1, 2], [3, 4]])
    inner = '{"action": "submit", "grid": [[7, 7], [6, 7]]}'
    assert parse_action(json.dumps({"response": inner})) == Submit([[7, 7], [6, 7]])
    assert parse_action("done\n" + json.dumps({"content": '{"action": "request_demos"}'})) == RequestDemos()
    # wrappers inside wrappers are followed a few levels down, and prose is still refused
    twice = json.dumps({"a": json.dumps({"b": inner})})
    assert parse_action(twice) == Submit([[7, 7], [6, 7]])
    with pytest.raises(ParseError):
        parse_action('{"note": "no action here"}')


def test_parse_reads_the_action_inside_a_malformed_or_nested_wrapper():
    from continual_arc_baselines.systems.protocol import RequestDemos, Submit, parse_action

    inner = '{"action":"submit","grid":[[7,7],[6,7]]}'
    # the runtime pasted the action between quotes without escaping: outer is not JSON
    assert parse_action('{"response": "' + inner + '"}') == Submit([[7, 7], [6, 7]])
    assert parse_action(inner + '"}') == Submit([[7, 7], [6, 7]])
    # dict wrappers
    assert parse_action('{"response": ' + inner + '}') == Submit([[7, 7], [6, 7]])
    assert parse_action('{"action": {"grid": [[1]]}}') == Submit([[1]])
    assert parse_action('{"action": {"action": "request_demos"}}') == RequestDemos()
    assert parse_action('{"output": {"action": "request_demos"}}') == RequestDemos()
    # a grid of nested lists is not mistaken for an object, and prose is still refused
    with pytest.raises(ParseError):
        parse_action("I stopped retrying delegation and will not call.")


def test_parse_survives_every_shape_seen_in_the_corpus_and_more():
    from continual_arc_baselines.systems.protocol import RequestDemos, Submit, parse_action

    g = [[0, 7, 7], [7, 0, 0]]
    ok = '{"action":"submit","grid":[[0,7,7],[7,0,0]]}'
    # stray characters after a complete action (192 of 281 real failures)
    for tail in ('"}', "]}", "}}", "}]", "]", ']"}'):
        assert parse_action(ok + tail) == Submit(g), tail
    # missing closers
    assert parse_action('{"action":"submit","grid":[[0,7,7],[7,0,0]') == Submit(g)
    assert parse_action('{"action":"submit","grid":[[0,7,7],[7,0,0]]') == Submit(g)
    assert parse_action('{"action":"submit","grid":[[0,7,7],[7,0,0]}') == Submit(g)
    # grid directly under the action key; action under other keys; argument objects
    assert parse_action('{"action":[[0,7,7],[7,0,0]]}') == Submit(g)
    assert parse_action('{"type": "Submit", "output": [[0,7,7],[7,0,0]]}') == Submit(g)
    assert parse_action('{"name": "submit", "arguments": {"grid": [[0,7,7],[7,0,0]]}}') == Submit(g)
    assert parse_action('{"tool": "request_demonstrations", "arguments": {}}') == RequestDemos()
    # a message that begins mid-grid keeps its complete rows
    assert parse_action(', 6, 6, 6],[0,7,7],[7,0,0]]}') == Submit(g)
    # Python repr, single quotes, trailing commas, smart quotes, booleans as cells
    assert parse_action("{'action': 'submit', 'grid': [[0, 7, 7], [7, 0, 0]]}") == Submit(g)
    assert parse_action('{"action": "submit", "grid": [[0,7,7],[7,0,0],],}') == Submit(g)
    assert parse_action('\u201caction\u201d: \u201crequest_demos\u201d'.encode().decode("unicode_escape") and
                        '{\u201caction\u201d: \u201crequest_demos\u201d}') == RequestDemos()
    # grid as a string, rows as strings, dict of rows, over-nested, floats
    assert parse_action('{"action":"submit","grid":"[[0,7,7],[7,0,0]]"}') == Submit(g)
    assert parse_action('{"action":"submit","grid":["077","700"]}') == Submit(g)
    assert parse_action('{"action":"submit","grid":["0 7 7","7 0 0"]}') == Submit(g)
    assert parse_action('{"action":"submit","grid":{"0":[0,7,7],"1":[7,0,0]}}') == Submit(g)
    assert parse_action('{"action":"submit","grid":[[[0,7,7],[7,0,0]]]}') == Submit(g)
    assert parse_action('{"action":"submit","grid":[[0.0,7.0,7.0],[7.0,0.0,0.0]]}') == Submit(g)
    # tags, bare lists, digit rows, fenced blocks with any tag, lists of actions
    assert parse_action("Here it is:\n<grid>[[0,7,7],[7,0,0]]</grid>") == Submit(g)
    assert parse_action("<action>request_demos</action>") == RequestDemos()
    assert parse_action("My answer:\n[[0,7,7],[7,0,0]]") == Submit(g)
    assert parse_action("Output grid:\n\n0 7 7\n7 0 0") == Submit(g)
    assert parse_action("```python\n" + ok + "\n```") == Submit(g)
    assert parse_action('[{"action":"request_demos"}, ' + ok + "]") == Submit(g)
    # a bare alias on the last line beats a quoted example above it
    assert parse_action('The format is {"action":"submit","grid":[[1]]} but I need more.\nrequest_demos') == RequestDemos()
    assert parse_action("REQUEST DEMOS") == RequestDemos()
    # a garbage grid under an explicit submit is a submission the environment charges
    assert parse_action('{"action": "submit", "grid": "nope"}') == Submit("nope")
    # refusals stay refusals
    for bad in ("done", "", "I stopped retrying delegation and will not call.", '{"action": "dance"}',
                '{"action": "give_up"}', "skip", "Let me think about rows 1 2 3 and 4 5 6."):
        with pytest.raises(ParseError):
            parse_action(bad)


def test_parse_does_not_read_prose_as_an_action():
    from continual_arc_baselines.systems.protocol import RequestDemos, parse_action

    for prose in ("Still exploring the demos; no conclusion yet.",
                  "The demos show a rotation. I will think more.",
                  "Cells [1, 2] and later [3, 4] are the anchors.",
                  "Examples: none yet.\nLet me reason about row 3 and row 4."):
        with pytest.raises(ParseError):
            parse_action(prose)
    for req in ("request_demos", "Request demos.", "I would like to request demos", "please show more examples",
                "get demonstrations", "REQUEST_DEMONSTRATIONS"):
        assert parse_action("Some reasoning first.\n" + req) == RequestDemos(), req


def test_demonstrations_are_numbered_across_requests_and_padding_says_what_it_is():
    """One pair per request since 26 Sep: the third request's pair is demo pair 3, and a padded pair (a transformed copy
    of one of the task's own pairs) is introduced by its note before its grids."""
    from continual_arc.env import DemosFeedback

    plain = Pair([[1]], [[2]])
    padded = Pair([[3]], [[4]], note="a transformed copy: rotated")
    third = render_feedback(RequestDemos(), DemosFeedback(pairs=[plain], refused=False, demo_requests_used=3))
    assert "Demo pair 3 input" in third and "Demo pair 1" not in third
    fourth = render_feedback(RequestDemos(), DemosFeedback(pairs=[padded], refused=False, demo_requests_used=4))
    assert fourth.index("Demo pair 4 is a transformed copy: rotated.") < fourth.index("Demo pair 4 input")
    assert "one more demonstration pair" in RULES and "transformed copy" in RULES
