from continual_arc.arc3.env import (
    Act,
    ActFeedback,
    Arc3LearnerObservation,
    ResetFeedback,
    ResetLevel,
)

from continual_arc_baselines.systems import arc3_protocol as p
from continual_arc_baselines.systems.base import protocol_for

FRAME = [[(r + c) % 16 for c in range(64)] for r in range(64)]


def obs(**over):
    base = dict(step=3, task_id="game-ab12", level=1, n_levels=7, frames=[FRAME], state="NOT_FINISHED",
                available_actions=[1, 2, 3, 4], actions_used=0, resets_used=0, action_cap=110, baseline_actions=22)
    base.update(over)
    return Arc3LearnerObservation(**base)


def test_protocol_selection_and_rendering():
    assert protocol_for(obs()) is p
    text = p.render_observation(obs(), pending=[], first_in_session=True)
    assert text.startswith(p.RULES.replace("{max_batch}", str(p.MAX_BATCH)))
    assert "Game id: game-ab12, level 2 of 7" in text
    assert "0123456789abcdef" in text  # one hex digit per cell
    assert text.count("\n") > 64
    again = p.render_observation(obs(actions_used=2), pending=["Action applied."], first_in_session=False)
    assert not again.startswith(p.RULES) and "Feedback since your last action" in again


def test_parse_actions():
    assert p.parse_action('ok\n{"action": "act", "id": 3}') == Act(3)
    assert p.parse_action('{"action": "act", "id": 6, "x": 10, "y": 12}') == Act(6, 10, 12)
    assert p.parse_action('{"action": "ACTION2"}') == Act(2)
    assert isinstance(p.parse_action('{"action": "reset"}'), ResetLevel)
    for bad in ("nothing here", '{"action": "act"}', '{"action": "dance"}', '{"action": "give_up"}'):
        try:
            p.parse_action(bad)
        except p.ParseError:
            continue
        raise AssertionError(bad)


def test_feedback_rendering():
    done = ActFeedback(frames=[FRAME], state="NOT_FINISHED", level_completed=True, actions_used=5, valid=True,
                       ignored=False)
    assert "COMPLETED" in p.render_feedback(Act(1), done)
    over = ActFeedback(frames=[FRAME, FRAME], state="GAME_OVER", level_completed=False, actions_used=9, valid=True,
                       ignored=False)
    text = p.render_feedback(Act(1), over)
    assert "GAME OVER" in text and "2 frames of animation" in text
    reset = ResetFeedback(frames=[FRAME], state="NOT_FINISHED", actions_used=9, resets_used=1)
    assert "Level reloaded (reset 1)" in p.render_feedback(ResetLevel(), reset)


def test_parse_sequences():
    seq = p.parse_action('{"action": "act", "ids": [1, 1, 2]}')
    assert seq == [Act(1), Act(1), Act(2)]
    seq = p.parse_action('{"actions": [{"id": 6, "x": 3, "y": 4}, {"id": 1}, "ACTION2"]}')
    assert seq == [Act(6, 3, 4), Act(1), Act(2)]
    assert len(p.parse_action('{"action": "act", "ids": ' + str(list(range(50))) + '}')) == p.MAX_BATCH


def test_sequence_is_fed_without_prompting_and_stops_at_game_over():
    from continual_arc_baselines.systems.base import SystemLearner

    class Scripted(SystemLearner):
        name = "scripted"
        replies = ['{"action": "act", "ids": [1, 2, 3, 4]}', '{"action": "reset"}']

        def __init__(self):
            super().__init__()
            self.asked = 0

        def start(self): pass
        def stop(self): pass
        def new_session(self): pass
        def wipe_state(self): pass

        def ask(self, message, *, reprompt=False):
            self.asked += 1
            return self.replies[min(self.asked - 1, len(self.replies) - 1)]

    s = Scripted()
    o = obs()
    assert s.act(o) == Act(1) and s.asked == 1
    s.observe(o, Act(1), ActFeedback(frames=[FRAME], state="NOT_FINISHED", level_completed=False, actions_used=1,
                                     valid=True, ignored=False))
    assert s.act(obs(actions_used=1)) == Act(2) and s.asked == 1  # queued, no prompt
    over = ActFeedback(frames=[FRAME], state="GAME_OVER", level_completed=False, actions_used=2, valid=True,
                       ignored=False)
    s.observe(obs(actions_used=1), Act(2), over)
    assert s._queue == []  # the rest of the sequence was dropped at game over
    # transcript: the compact line for action 1, then the full frame for action 2
    lines = list(s.transcript.pending)
    assert len(lines) == 2 and "Frame:" not in lines[0] and "Frame:" in lines[1]
    assert isinstance(s.act(obs(actions_used=2, state="GAME_OVER")), ResetLevel) and s.asked == 2
    assert s.stats["batched_actions"] == 1
