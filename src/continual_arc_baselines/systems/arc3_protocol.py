"""The text protocol between the benchmark and a system for ARC-AGI-3 games.

Same shape as the grid protocol (:mod:`continual_arc_baselines.systems.protocol`):
the rules once per session, then one message per action with the current
frame, the feedback since the last message, and a request for one JSON action.
There is no give-up action on this track: an instance ends when its level is
completed or its action cap is reached, so quitting can never be cheaper than
playing on (20 Sep: nearly every frontier-model instance ended in a voluntary
give-up after the agent read "fewer actions is better" as a reason to stop early).
Frames are 64 by 64 cells with colours 0 to 15, rendered one hexadecimal digit
per cell, one row per line. An action that animates returns several frames;
the last is shown in full and the count is stated.
"""

from __future__ import annotations

import json
import re
from typing import Any

from continual_arc.env import GiveUpFeedback

RULES = """\
You are being evaluated on Continual-ARC-3, a lifelong stream of ARC-AGI-3 games.
Each game has levels; an instance is one level of one game. You see the current
frame (a 64x64 grid of colours 0-15, one hex digit per cell, row by row, top row
first) and the actions the game accepts. You must discover what the game wants by
acting. Games recur across the stream, so anything you learn about a game id is
worth keeping.

On every message choose what to do and put it as a JSON object on the last line
of your reply:
  {"action": "act", "id": 1}                      one simple action (ids 1-5 and 7)
  {"action": "act", "id": 6, "x": 12, "y": 40}     one click at column x, row y (0-63)
  {"action": "act", "ids": [1, 1, 2, 5]}           a sequence of up to {max_batch} simple actions,
                                                  applied in order; it stops early if the
                                                  level completes or the game ends
  {"actions": [{"id": 6, "x": 3, "y": 4}, {"id": 1}]}   a sequence with clicks in it
  {"action": "reset"}                             reload the current level
Actions 1-4 usually move (up, down, left, right), 5 usually selects or interacts,
6 clicks, 7 undoes where supported; the meaning is the game's to define. Every
action in a sequence is charged; you see the frame after the sequence.

Scoring: an instance costs the actions you spend (resets included) divided by the
human baseline for its level, up to a cap of several times that baseline (the
observation states your cap). Completing the level ends the instance at that
cost; reaching the cap ends it at the maximum cost. There is no way to abandon an
instance and nothing is shared between instances: each is scored on its own, so
keep exploring until the level completes or the cap ends it. After a game over
you must reset to continue.
"""


# The rules under the finish protocol (end="finish", 27 Sep, Dan): facts only, and the harness ends each instance itself
# with finish, refused until the instance is over, as on the grid tracks (protocol.RULES_FINISH).
RULES_FINISH = """\
You are being evaluated on Continual-ARC-3, a lifelong stream of ARC-AGI-3 games.
Each game has levels; an instance is one level of one game. You see the current
frame (a 64x64 grid of colours 0-15, one hex digit per cell, row by row, top row
first) and the actions the game accepts. You must discover what the game wants by
acting. Games recur across the stream. Each instance is a separate conversation:
the next one starts without this one's messages.

On every message choose what to do and put it as a JSON object on the last line
of your reply:
  {"action": "act", "id": 1}                      one simple action (ids 1-5 and 7)
  {"action": "act", "id": 6, "x": 12, "y": 40}     one click at column x, row y (0-63)
  {"action": "act", "ids": [1, 1, 2, 5]}           a sequence of up to {max_batch} simple actions,
                                                  applied in order; it stops early if the
                                                  level completes or the game ends
  {"actions": [{"id": 6, "x": 3, "y": 4}, {"id": 1}]}   a sequence with clicks in it
  {"action": "reset"}                             reload the current level
  {"action": "finish"}                            end the instance, once it is over
Actions 1-4 usually move (up, down, left, right), 5 usually selects or interacts,
6 clicks, 7 undoes where supported; the meaning is the game's to define. Every
action in a sequence is charged; you see the frame after the sequence.

Scoring: an instance costs the actions you spend (resets included) divided by the
human baseline for its level, up to a cap of several times that baseline (the
observation states your cap). The instance is over when the level completes, at
that cost, or when the cap is reached, at the maximum cost. There is no way to
abandon an instance, and finish is refused until the instance is over. Each instance
is scored on its own, and lower cost is better. After a game over you must reset to
continue.
"""


def render_frame(frame: list[list[int]]) -> str:
    return "\n".join("".join(format(c, "x") for c in row) for row in frame)


def _state_line(state: str) -> str:
    if state == "GAME_OVER":
        return "State: GAME OVER. Reset to continue this level."
    if state == "WIN":
        return "State: WIN."
    return f"State: {state}."


def render_frames(frames: list[Any]) -> str:
    if not frames:
        return "(no frame returned)"
    head = "" if len(frames) == 1 else f"({len(frames)} frames of animation; the last is shown)\n"
    return head + render_frame(frames[-1])


MAX_BATCH = 20


def render_feedback(action: Any, feedback: Any, compact: bool = False) -> str:
    """``compact`` omits the frame: used for every action of a sequence except the
    last, so the transcript carries one frame per turn."""
    kind = type(feedback).__name__
    if kind == "ActFeedback":
        if feedback.level_completed:
            return f"Level COMPLETED after {feedback.actions_used} action(s). Instance solved."
        why = "" if feedback.valid else " (the action was not valid here and was charged)"
        ignored = " The game ignored it: it is over; reset to continue." if getattr(feedback, "ignored", False) else ""
        cap = " The action cap is reached; the instance is over." if getattr(feedback, "failed", False) else ""
        head = (f"Action {getattr(action, 'action', '?')} applied{why}.{ignored} Actions used: {feedback.actions_used}. "
                f"{_state_line(feedback.state)}{cap}")
        if compact:
            return head
        return f"{head}\nFrame:\n{render_frames(feedback.frames)}"
    if kind == "ResetFeedback":
        cap = " The action cap is reached; the instance is over." if getattr(feedback, "failed", False) else ""
        return (f"Level reloaded (reset {feedback.resets_used}). Actions used: {feedback.actions_used}. "
                f"{_state_line(feedback.state)}{cap}\nFrame:\n{render_frames(feedback.frames)}")
    if isinstance(feedback, GiveUpFeedback):
        return f"You gave up. Charged {feedback.charged:g}."
    return f"Feedback: {feedback!r}"


def render_observation(obs: Any, *, pending: list[str], first_in_session: bool,
                       finish: bool = False, rules_note: str = "", **_: Any) -> str:
    lines: list[str] = []
    if first_in_session:
        lines.append((RULES_FINISH if finish else RULES).replace("{max_batch}", str(MAX_BATCH)))
        if finish and rules_note:
            lines.append(rules_note)
        lines.append("")
    if pending:
        lines.append("Feedback since your last action:")
        lines.extend(pending)
        lines.append("")
    fresh = obs.is_fresh
    # Game id and level (the game shows its level), nothing about the stream (20 Sep).
    lines.append(f"{'New instance' if fresh else 'Same instance'}. Game id: {obs.task_id}, level {obs.level + 1} of {obs.n_levels}.")
    lines.append(f"Available actions: {', '.join(str(a) for a in obs.available_actions)}. "
                 f"Actions used: {obs.actions_used + obs.resets_used}/{obs.action_cap} "
                 f"(human baseline for this level: {obs.baseline_actions}). {_state_line(obs.state)}")
    if fresh or not pending:
        lines.append("Frame:")
        lines.append(render_frames(obs.frames))
    lines.append("Reply with exactly one action JSON on the last line.")
    return "\n".join(lines)


class ParseError(ValueError):
    pass


ACT_ALIASES = frozenset({"act", "action", "step", "press", "move", "click"})
RESET_ALIASES = frozenset({"reset", "reset_level", "restart"})
GIVE_UP_ALIASES = frozenset({"give_up", "giveup", "abandon", "skip"})
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _json_objects_from_end(text: str) -> list[str]:
    from continual_arc_baselines.systems.protocol import _json_objects_from_end as inner

    return inner(text)


def _one(spec: Any):
    """An ``Act`` from an int id or an object with id and optional x, y."""
    from continual_arc.arc3.env import Act

    if isinstance(spec, bool):
        raise ParseError(f"not an action: {spec!r}")
    if isinstance(spec, str):
        name = spec.strip().lower()
        if name.startswith("action") and name[6:].isdigit():
            return Act(int(name[6:]))
        if name.isdigit():
            return Act(int(name))
        raise ParseError(f"not an action: {spec!r}")
    if isinstance(spec, int):
        return Act(spec)
    if isinstance(spec, dict):
        ident = spec.get("id", spec.get("action_id", spec.get("n", spec.get("action"))))
        if isinstance(ident, str) and ident.lower().startswith("action") and ident[6:].isdigit():
            ident = int(ident[6:])
        try:
            ident = int(ident)
        except (TypeError, ValueError):
            raise ParseError(f"act needs an integer id, got {ident!r}") from None
        x, y = spec.get("x"), spec.get("y")
        return Act(ident, None if x is None else int(x), None if y is None else int(y))
    raise ParseError(f"not an action: {spec!r}")


def parse_action(text: str, *, finish: bool = False):
    """One action, or a list of ``Act`` for a sequence (at most ``MAX_BATCH``); with ``finish`` (the finish protocol),
    a finish action in any usual spelling too, as ``protocol.Finish``."""
    from continual_arc.arc3.env import Act, ResetLevel

    from continual_arc_baselines.systems.protocol import FINISH_ALIASES, Finish

    if finish:
        last = next((ln for ln in reversed(text.splitlines()) if ln.strip()), "")
        if "{" not in last and last.strip().strip(".!`\"'").lower().replace(" ", "_") in FINISH_ALIASES:
            return Finish()

    candidates = [m.group(1) for m in reversed(list(_FENCE.finditer(text)))]
    candidates += _json_objects_from_end(text)
    unknown: list[str] = []
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        if "actions" in obj and isinstance(obj["actions"], list):
            seq = [_one(a) for a in obj["actions"]][:MAX_BATCH]
            if not seq:
                raise ParseError("empty action sequence")
            return seq
        if "action" not in obj:
            continue
        action = str(obj.get("action", "")).strip().lower().replace("-", "_").replace(" ", "_")
        if finish and action in FINISH_ALIASES:
            return Finish()
        if action.startswith("action") and action[6:].isdigit():  # {"action": "ACTION3"}
            return Act(int(action[6:]))
        if action in ACT_ALIASES:
            if isinstance(obj.get("ids"), list):
                seq = [_one(a) for a in obj["ids"]][:MAX_BATCH]
                if not seq:
                    raise ParseError("empty action sequence")
                return seq
            return _one(obj)
        if action in RESET_ALIASES:
            return ResetLevel()
        if action in GIVE_UP_ALIASES:
            raise ParseError("there is no give_up in this game: the instance ends when the level "
                             "completes or the action cap is reached, so keep acting")
        unknown.append(str(obj.get("action")))
    if unknown:
        raise ParseError(f"unknown action {unknown[0]!r}")
    raise ParseError("no action JSON object found in the reply")


REPROMPT = (
    "I could not find a valid action in your reply ({reason}). Reply again with exactly one of\n"
    '{{"action": "act", "id": N}} (with "x" and "y" for id 6), {{"action": "act", "ids": [...]}}, '
    'or {{"action": "reset"}}\nas a JSON object on the last line.'
)

REPROMPT_FINISH = (
    "I could not find a valid action in your reply ({reason}). Reply again with exactly one of\n"
    '{{"action": "act", "id": N}} (with "x" and "y" for id 6), {{"action": "act", "ids": [...]}}, '
    '{{"action": "reset"}} or {{"action": "finish"}}\nas a JSON object on the last line.'
)
