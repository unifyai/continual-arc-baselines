"""The text protocol between the benchmark and a system.

Every system gets the same words: the rules once per session, then one message
per action with the current instance, the feedback since the last message, and
a request for exactly one JSON action. Parsing is deliberately forgiving about
where the JSON sits (fenced or bare, anywhere in the reply) and strict about
what it says.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from continual_arc.env import (
    Action,
    DemosFeedback,
    Feedback,
    GiveUpFeedback,
    LearnerObservation,
    RequestDemos,
    Submit,
    SubmitFeedback,
)
from continual_arc.grids import Grid

RULES = """\
You are being evaluated on Continual-ARC, a lifelong stream of ARC-style grid puzzles.
Each puzzle instance has a task id and a test input grid. Tasks recur across the
stream with fresh inputs, so anything you learn about a task id is worth keeping.

On every message you must choose exactly one action and put it as a JSON object
on the last line of your reply:
  {"action": "submit", "grid": [[0, 1, 2], [3, 4, 5]]}   submit an output grid
  {"action": "request_demos"}                            ask for one more demonstration pair for this task
Grids are lists of rows; cells are integers 0-9; dimensions are 1-30.

Each demo request returns one more input-output pair of the task, up to the demo cap shown
with the input. A pair marked as a transformed copy shows the task's rule under a different
rotation, reflection or recolouring from the test input's.

Scoring: an instance you solve costs 1 for each wrong submission and {demo_cost} for
each demo request. An instance ends when a submission is correct or when the attempt
cap is reached, and one that ends unsolved costs the whole budget, {max_cost}, however
many submissions and demo requests it used. Feedback on a submission is only
correct/incorrect. There is no way to abandon an instance and nothing is shared
between instances: each is scored on its own. Lower cost is better, so prefer
remembering a task's rule over re-requesting demos, and prefer requesting demos
over guessing blindly; solving at any cost is cheaper than not solving.
"""


# The rules under the finish protocol (a system's end="finish", every harness row since 27 Sep, Dan): facts only. The
# stream's recurrence and the conversation per instance are stated, with no advice on what to remember or when to ask for
# demonstrations, and the harness ends each instance itself with finish, refused until the instance is over, so that no
# message of ours tells it what to do after the verdict ("the harnesses should be free to do what they want").
RULES_FINISH = """\
You are being evaluated on Continual-ARC, a lifelong stream of ARC-style grid puzzles.
Each puzzle instance has a task id and a test input grid. Tasks recur across the
stream with fresh inputs. Each instance is a separate conversation: the next one
starts without this one's messages.

On every message you must choose exactly one action and put it as a JSON object
on the last line of your reply:
  {"action": "submit", "grid": [[0, 1, 2], [3, 4, 5]]}   submit an output grid
  {"action": "request_demos"}                            ask for one more demonstration pair for this task
  {"action": "finish"}                                   end the instance, once it is over
Grids are lists of rows; cells are integers 0-9; dimensions are 1-30.

Each demo request returns one more input-output pair of the task, up to the demo cap shown
with the input.

Scoring: an instance you solve costs 1 for each wrong submission and {demo_cost} for
each demo request. An instance is over when a submission is correct or when the attempt
cap is reached, and one that ends unsolved costs the whole budget, {max_cost}, however
many submissions and demo requests it used. Feedback on a submission is only
correct/incorrect. There is no way to abandon an instance, and finish is refused until
the instance is over. Each instance is scored on its own, and lower cost is better.
"""



def rules(finish: bool, demo_cost: float, max_cost: float) -> str:
    """The rules a learner reads at the start of a conversation (``RULES_FINISH`` under the finish protocol), with the
    configuration's demo cost and whole budget filled in."""
    return (RULES_FINISH if finish else RULES).replace("{demo_cost}", f"{demo_cost:g}").replace("{max_cost}", f"{max_cost:g}")


class Finish:
    """The finish action of the finish protocol: the harness ends an instance itself, once the instance is over."""

    def __repr__(self) -> str:
        return "Finish()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Finish)

    def __hash__(self) -> int:
        return hash("Finish")


FINISH_ALIASES = frozenset({"finish", "finished", "done", "end", "end_instance", "finish_instance", "next_instance"})


def render_grid(grid: Grid) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def _shape(grid: Grid) -> str:
    return f"{len(grid)}x{len(grid[0]) if grid else 0}"


def render_pairs(pairs: list[Any], start: int = 1) -> str:
    """Demonstration pairs numbered from ``start``; a padded pair says what it is first (26 Sep)."""
    parts = []
    for k, pair in enumerate(pairs, start):
        note = getattr(pair, "note", None)
        parts.append(
            (f"Demo pair {k} is {note}.\n" if note else "")
            + f"Demo pair {k} input ({_shape(pair.input)}):\n{render_grid(pair.input)}\n"
            f"Demo pair {k} output ({_shape(pair.output)}):\n{render_grid(pair.output)}"
        )
    return "\n\n".join(parts)


def render_feedback(action: Action, feedback: Feedback) -> str:
    if isinstance(feedback, SubmitFeedback):
        if feedback.correct:
            return f"Your submission was CORRECT. Instance solved with {feedback.attempts_used} wrong attempt(s)."
        why = "" if feedback.valid else " (the grid was not a valid ARC grid)"
        tail = " The attempt cap is reached; the instance is over." if feedback.failed else ""
        return f"Your submission was INCORRECT{why}. Wrong attempts so far: {feedback.attempts_used}.{tail}"
    if isinstance(feedback, DemosFeedback):
        if feedback.refused:
            return "Demo request REFUSED: the demo cap for this instance is reached."
        # pairs are numbered across the instance's requests (one pair per request since 26 Sep)
        start = (feedback.demo_requests_used - 1) * len(feedback.pairs) + 1
        return (f"Demos received (request {feedback.demo_requests_used}):\n\n"
                f"{render_pairs(feedback.pairs, start=max(1, start))}")
    if isinstance(feedback, GiveUpFeedback):
        return f"You gave up. Charged {feedback.charged:g}, the rest of the instance's budget."
    return f"Feedback: {feedback!r}"


def render_observation(obs: LearnerObservation, *, pending: list[str], demo_cost: float, max_cost: float,
                       first_in_session: bool, finish: bool = False, rules_note: str = "", preface: str = "") -> str:
    """``preface`` is what a row puts before the instance in the first message of its conversation: nothing, but for
    the demo-cache floor's cached pairs (systems/bare.py, 28 Sep)."""
    lines: list[str] = []
    if first_in_session:
        lines.append(rules(finish, demo_cost, max_cost))
        if finish and rules_note:
            lines.append(rules_note)
        lines.append("")
    if pending:
        lines.append("Feedback since your last action:")
        lines.extend(pending)
        lines.append("")
    if preface:
        lines.append(preface)
        lines.append("")
    fresh = obs.attempts_used == 0 and obs.demo_requests_used == 0
    # The learner is told the task's name for the run and the instance's own counters, nothing
    # about the stream: no position, no length, no phase, no exposure count (20 Sep). Whether it
    # has met this task before is for its own memory to say, as for a worker in an office.
    lines.append(f"{'New instance' if fresh else 'Same instance'}. Task id: {obs.task_id}")
    lines.append(f"Test input ({_shape(obs.input)}):")
    lines.append(render_grid(obs.input))
    lines.append(f"Wrong attempts used: {obs.attempts_used}/{obs.attempt_cap}. "
                 f"Demo requests used: {obs.demo_requests_used}/{obs.demo_cap}.")
    lines.append("Reply with exactly one action JSON on the last line.")
    return "\n".join(lines)


class ParseError(ValueError):
    pass


# Systems are not being tested on JSON discipline, so obvious synonyms are accepted.
SUBMIT_ALIASES = frozenset({"submit", "answer", "output", "solution", "submit_grid", "respond", "submission",
                            "final_answer", "final", "predict", "prediction", "guess", "attempt", "submit_answer",
                            "submit_output", "submit_solution", "output_grid", "solve", "response"})
GRID_KEYS = ("grid", "answer", "output", "solution", "test_output", "output_grid", "prediction", "predicted_output",
             "result", "matrix", "submission", "final_grid", "final_answer", "guess", "attempt", "response",
             "content", "data", "value", "cells", "rows")
DEMO_ALIASES = frozenset({"request_demos", "request_demo", "demos", "demo", "get_demos", "request_demonstrations",
                          "demonstrations", "demonstration", "examples", "request_examples", "get_examples",
                          "show_demos", "more_demos", "ask_for_demos", "requestdemos", "request_demonstration",
                          "get_demonstrations", "demo_request", "request_pairs", "more_examples"})
GIVE_UP_ALIASES = frozenset({"give_up", "giveup", "abandon", "skip", "pass", "quit", "forfeit", "surrender"})
# Keys a runtime or a model may use for the action name, the grid, and a nested argument object.
ACTION_KEYS = ("action", "type", "tool", "name", "command", "op", "operation", "intent", "move", "decision",
               "choice", "tool_name", "function", "action_type", "kind", "actions", "step")
ARG_KEYS = ("arguments", "args", "params", "parameters", "input", "inputs", "payload", "body", "tool_input")

_FENCE = re.compile(r"```[a-zA-Z0-9_-]*[ \t]*\n?(.*?)```", re.DOTALL)
_TAG = re.compile(r"<(grid|answer|output|solution|submit|final_answer|final|prediction|result)>(.*?)</\1>",
                  re.DOTALL | re.IGNORECASE)
_ACTION_TAG = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_ROW = re.compile(r"\[\s*\d\s*(?:,\s*\d\s*)*\]")
_SMART = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'",
                        " ": " ", "﻿": "", "​": ""})
_MAX_CANDIDATES = 200
_MAX_DEPTH = 3


def _clean(text: str) -> str:
    return (text or "").translate(_SMART).strip()


def _normalise_name(name: str) -> str:
    name = name.strip().strip("`'\"*_.:;!,()[]{}<> ").lower()
    return re.sub(r"[\s\-]+", "_", name)


_BARE_DEMO = re.compile(r"^(please_)?(i_|we_)?(would_like_to_|want_to_|will_|d_like_to_|ll_)?"
                        r"(request|get|ask_for|need|show|see|fetch)_(some_|more_|the_|a_)?(demos?|demonstrations?|examples?)"
                        r"(_please)?$")


def _classify(name: str, *, strict: bool = False, finish: bool = False) -> str | None:
    """'submit', 'demos', 'give_up', 'finish' (only with ``finish``) or None for an action name: exact aliases first,
    then words. ``strict`` is for free text (a bare last line, a string value): only an alias or a plain
    request phrase counts, so prose that mentions demonstrations is not a request for them."""
    n = _normalise_name(name)
    if not n:
        return None
    if finish and n in FINISH_ALIASES:
        return "finish"
    if n in DEMO_ALIASES or _BARE_DEMO.match(n):
        return "demos"
    if n in GIVE_UP_ALIASES:
        return "give_up"
    if n in SUBMIT_ALIASES:
        return "submit"
    if strict:
        return None
    if "demo" in n or "example" in n or "demonstration" in n:
        return "demos"
    if any(w in n for w in ("give_up", "giveup", "abandon", "skip", "forfeit", "surrender")):
        return "give_up"
    if any(w in n for w in ("submit", "answer", "solution", "output", "predict", "final", "guess", "attempt", "grid", "solve")):
        return "submit"
    return None


def _repair(s: str) -> str:
    """Balance the brackets of a JSON-like string: drop unmatched closers, append missing ones.
    String literals are skipped so braces inside them do not count."""
    out: list[str] = []
    stack: list[str] = []
    in_str = False
    esc = False
    pairs = {"{": "}", "[": "]"}
    for ch in s:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in pairs:
            stack.append(pairs[ch])
            out.append(ch)
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()
                out.append(ch)
            # an unmatched or mismatched closer is dropped
        else:
            out.append(ch)
    if in_str:
        out.append('"')
    while stack:
        out.append(stack.pop())
    return "".join(out).rstrip().rstrip(",")


def _loads(s: str) -> Any:
    """Parse JSON leniently: as is, without trailing commas, with escaped quotes unescaped, as a
    Python literal, and after bracket repair. None when nothing works."""
    s = s.strip()
    if not s or len(s) > 4_000_000:
        return None
    attempts = [s, re.sub(r",\s*([}\]])", r"\1", s)]
    if "\\" in s:
        attempts.append(s.replace('\\"', '"').replace("\\n", "\n"))
    for a in attempts:
        try:
            return json.loads(a)
        except ValueError:
            pass
    try:
        import ast

        v = ast.literal_eval(s)
        if isinstance(v, (dict, list, tuple)):
            return v
    except Exception:
        pass
    repaired = _repair(s)
    if repaired != s:
        for a in (repaired, re.sub(r",\s*([}\]])", r"\1", repaired)):
            try:
                return json.loads(a)
            except ValueError:
                pass
    return None


def _cell(v: Any) -> int | None:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v  # any integer: the environment judges colours, and a typo like 11 costs an attempt
    if isinstance(v, float):
        return int(v) if v.is_integer() else None
    if isinstance(v, str):
        t = v.strip()
        return int(t) if t.isdigit() and len(t) == 1 else None
    return None


def _digit_rows(text: str) -> list[list[int]] | None:
    """Rows of single digits written as text: '1 2 3', '123', '1,2,3', '| 1 | 2 |'."""
    rows: list[list[int]] = []
    for line in text.strip().splitlines():
        line = line.strip().strip("|").replace("|", " ").replace(",", " ").replace("[", " ").replace("]", " ")
        if not line.strip():
            if rows:
                break
            continue
        toks = line.split()
        if all(t.isdigit() for t in toks):
            cells = [int(c) for t in toks for c in t] if any(len(t) > 1 for t in toks) else [int(t) for t in toks]
            if cells and all(0 <= c <= 9 for c in cells):
                rows.append(cells)
                continue
        return None
    return rows or None


def _to_grid(v: Any, depth: int = 0) -> list[list[int]] | None:
    """A grid from anything a model might hand back: nested lists, rows as strings, a string
    holding a list, a dict of rows, an over-nested list. None when it is not a grid."""
    if depth > 4 or v is None or isinstance(v, bool):
        return None
    if isinstance(v, str):
        t = v.strip()
        if not t:
            return None
        if t[0] in "[{(":
            obj = _loads(t)
            if obj is not None:
                return _to_grid(obj, depth + 1)
        return _digit_rows(t)
    if isinstance(v, dict):
        keys = list(v.keys())
        if keys and all(str(k).strip().lstrip("-").isdigit() for k in keys):
            return _to_grid([v[k] for k in sorted(keys, key=lambda k: int(str(k).strip()))], depth + 1)
        for k in GRID_KEYS:
            for kk in v:
                if str(kk).lower() == k:
                    g = _to_grid(v[kk], depth + 1)
                    if g:
                        return g
        return None
    if isinstance(v, tuple):
        v = list(v)
    if not isinstance(v, list) or not v:
        return None
    # unwrap [[[...]]] to a list of rows
    while len(v) == 1 and isinstance(v[0], list) and v[0] and isinstance(v[0][0], list):
        v = v[0]
    if all(isinstance(r, (list, tuple, str)) for r in v):
        rows: list[list[int]] = []
        for r in v:
            if isinstance(r, str):
                row = _digit_rows(r)
                if not row or len(row) != 1:
                    return None
                rows.append(row[0])
            else:
                cells = [_cell(c) for c in r]
                if not cells or any(c is None for c in cells):
                    return None
                rows.append(cells)  # type: ignore[arg-type]
        return rows
    cells = [_cell(c) for c in v]
    if cells and all(c is not None for c in cells):
        return [cells]  # a flat list is one row
    return None


def _lower_keys(d: dict) -> dict:
    out: dict = {}
    for k, val in d.items():
        out.setdefault(str(k).strip().lower(), val)
    return out


def _grid_in(d: dict, depth: int = 0) -> list[list[int]] | None:
    if depth > _MAX_DEPTH:
        return None
    for k in GRID_KEYS:
        if k in d:
            g = _to_grid(d[k])
            if g:
                return g
    for k in ARG_KEYS:
        if isinstance(d.get(k), dict):
            g = _grid_in(_lower_keys(d[k]), depth + 1)
            if g:
                return g
        elif k in d:
            g = _to_grid(d[k])
            if g:
                return g
    # any value that is a grid of at least two rows
    for val in d.values():
        if isinstance(val, (list, str, dict)):
            g = _to_grid(val)
            if g and len(g) >= 2:
                return g
    return None


class _NoGiveUp(Exception):
    pass


def _action_from(obj: Any, depth: int = 0, unknown: list[str] | None = None, finish: bool = False) -> Action | None:
    """The action an already-parsed value describes, or None."""
    if depth > _MAX_DEPTH:
        return None
    if isinstance(obj, (list, tuple)):
        g = _to_grid(obj)
        if g and len(g) >= 2:
            return Submit(g)
        for item in reversed(list(obj)):
            act = _action_from(item, depth + 1, unknown, finish)
            if act:
                return act
        return None
    if isinstance(obj, str):
        kind = _classify(obj, strict=True, finish=finish) if len(obj) < 80 else None
        if kind == "finish":
            return Finish()
        if kind == "demos":
            return RequestDemos()
        if kind == "give_up":
            raise _NoGiveUp()
        if "{" in obj or "[" in obj:
            try:
                return parse_action(obj, _depth=depth + 1, finish=finish)
            except ParseError:
                return None
        return None
    if not isinstance(obj, dict):
        return None
    d = _lower_keys(obj)
    name = None
    for k in ACTION_KEYS:
        val = d.get(k)
        if isinstance(val, str):
            name = val
            break
        if isinstance(val, dict):
            act = _action_from(val, depth + 1, unknown, finish)
            if act:
                return act
        elif isinstance(val, (list, tuple)):
            g = _to_grid(val)
            if g:
                return Submit(g)
    kind = _classify(name, finish=finish) if name else None
    if kind == "finish":
        return Finish()
    if kind == "give_up":
        raise _NoGiveUp()
    if kind == "demos":
        return RequestDemos()
    grid = _grid_in(d)
    if kind == "submit" and grid:
        return Submit(grid)
    if grid and (kind is None or kind == "submit"):
        return Submit(grid)
    if kind == "submit":
        # a submission that names no readable grid: look deeper; then, if a grid key is present
        # but its value is not a grid, it is still a submission and the environment charges it
        # (a system that submits garbage pays an attempt, not a free reprompt)
        for val in d.values():
            if isinstance(val, (dict, list, str)):
                g = _to_grid(val)
                if g:
                    return Submit(g)
        for k in GRID_KEYS:
            if k in d:
                return Submit(d[k])
        if unknown is not None:
            unknown.append("submit without a grid")
        return None
    if name and unknown is not None:
        unknown.append(name)
    # no readable action here: an action nested under any key, as a dict or a string
    for val in d.values():
        if isinstance(val, dict) or (isinstance(val, str) and ("{" in val or "[" in val)):
            act = _action_from(val, depth + 1, unknown, finish)
            if act:
                return act
    return None


def _objects_from_end(text: str, open_ch: str = "{", close_ch: str = "}") -> list[str]:
    """Candidate substrings, last first, one per closing bracket by matching, so an object nested
    inside another is a candidate of its own (the outer one may not be valid JSON: a runtime that
    pastes the action between quotes without escaping it, or a model that adds a stray brace)."""
    found: list[str] = []
    close = len(text)
    while len(found) < _MAX_CANDIDATES:
        close = text.rfind(close_ch, 0, close)
        if close < 0:
            break
        depth = 0
        start = -1
        for i in range(close, -1, -1):
            ch = text[i]
            if ch == close_ch:
                depth += 1
            elif ch == open_ch:
                depth -= 1
                if depth == 0:
                    start = i
                    break
        if start >= 0:
            found.append(text[start:close + 1])
    return found


_ROW_RUN = re.compile(r"(?:\[\s*\d\s*(?:,\s*\d\s*)*\]\s*,?\s*){2,}")


def _rows_from_text(text: str) -> list[list[int]] | None:
    """The last contiguous run of digit rows written as lists, ``[1,2,3],[4,5,6]``, when the
    brackets around them are broken or the message begins mid-grid. Rows must follow one another
    with nothing but commas between them, so coordinates mentioned in prose are not a grid, and a
    partial leading row is not a row."""
    runs = list(_ROW_RUN.finditer(text))
    if not runs:
        return None
    rows = [[int(c) for c in re.findall(r"\d", m.group(0))] for m in _ROW.finditer(runs[-1].group(0))]
    if len(rows) < 2 or len({len(r) for r in rows}) != 1:
        return None
    return rows


def parse_action(text: str, _depth: int = 0, *, finish: bool = False) -> Action:
    """Find the action in a reply. Raises :class:`ParseError` if there is none.

    Systems are not being tested on JSON discipline (23 Sep: a tenth of low-effort instances were
    lost to replies that carried a correct action in a shape the first parser refused), so this
    reads, in order: a bare demo request on the last line; every brace-matched JSON object, last
    first and nested ones included, parsed leniently (trailing commas, single quotes, escaped
    quotes, missing or stray brackets), with the action name under any usual key and the grid
    under any usual key or nested argument object, a grid placed directly under the action key,
    and a wrapper whose value is itself an action (as a dict or a string); a grid in an XML-style
    tag; a bare nested list; rows written as ``[1,2],[3,4]`` when the brackets around them are
    broken; and rows of digits as text. A give-up in any spelling is refused, since there is no
    such action. With ``finish`` (the finish protocol) a finish action in any usual spelling is read too,
    as :class:`Finish`.
    """
    text = _clean(text)
    if not text:
        raise ParseError("empty reply")
    unknown: list[str] = []
    try:
        # 1. a bare demo request on the last line beats any earlier quoted example
        last = next((ln for ln in reversed(text.splitlines()) if ln.strip()), "")
        if "{" not in last and "[" not in last and len(last) < 80:
            bare = _classify(last, strict=True, finish=finish)
            if bare == "demos":
                return RequestDemos()
            if bare == "finish":
                return Finish()
        m = _ACTION_TAG.search(text)
        if m:
            kind = _classify(m.group(1), strict=True, finish=finish)
            if kind == "finish":
                return Finish()
            if kind == "demos":
                return RequestDemos()
            if kind == "give_up":
                raise _NoGiveUp()
        # 2. JSON objects: fenced blocks first, then every brace-matched object from the end
        candidates: list[str] = []
        for fm in reversed(list(_FENCE.finditer(text))):
            candidates.append(fm.group(1))
        candidates += _objects_from_end(text)
        if text[0] in "{[":
            candidates.append(text)
        seen: set[str] = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            obj = _loads(cand)
            if obj is None:
                continue
            act = _action_from(obj, _depth, unknown, finish)
            if act:
                return act
        # 3. a grid in a tag
        for tm in reversed(list(_TAG.finditer(text))):
            g = _to_grid(tm.group(2))
            if g:
                return Submit(g)
        # 4. a bare list of rows anywhere, last first
        for cand in _objects_from_end(text, "[", "]"):
            obj = _loads(cand)
            g = _to_grid(obj) if obj is not None else None
            if g and len(g) >= 2:
                return Submit(g)
        # 5. rows as [1,2],[3,4] with broken brackets around them, or a message starting mid-grid
        g = _rows_from_text(text)
        if g:
            return Submit(g)
        # 6. rows of digits as text at the end of the reply
        tail = text.split("\n\n")[-1]
        g = _digit_rows(tail)
        if g and len(g) >= 2 and len({len(r) for r in g}) == 1 and all(len(r) >= 2 for r in g):
            return Submit(g)
    except _NoGiveUp:
        raise ParseError("there is no give_up: the instance ends when a submission is correct "
                         "or the attempt cap is reached, so submit or request demos") from None
    if unknown:
        raise ParseError(f"unknown action {unknown[0]!r}")
    raise ParseError("no action JSON object found in the reply")


CLOSING = (
    "{verdict}\n\nThis instance is finished. No action is needed for it; reply with the single word: done."
)


def render_closing(pending: list[str]) -> str:
    """The message that closes an instance: the verdict on its final submission, which would
    otherwise reach the system only at the top of the next instance's first message, after its
    storage step has run (found 23 Sep: a harness's review wrote 'the submitted result received
    no correctness feedback, so it adds no verified knowledge')."""
    return CLOSING.format(verdict="\n".join(pending) if pending else "The instance is over.")


STORAGE = (
    "{verdict}\n\nThis instance is over: no more actions can be taken in it.{ends} Before the next instance, you may save "
    "anything that will help you with future instances, using your own tools.{note} When you are done, reply with the "
    "single word: done."
)


def render_storage(pending: list[str], *, conversation_ends: bool, note: str = "") -> str:
    """The message that closes an instance with a turn of its own for storage (a system's ``storage_turn``): the verdict,
    then the agent may save what will help later with whatever tools its harness has, and the reply is not read (Dan,
    26 Sep: a harness's storage functions are called at will, and the verdict comes only after the turn that submitted,
    so without this turn nothing could be stored knowing the outcome). ``note``: what the harness keeps, in its terms."""
    return STORAGE.format(verdict="\n".join(pending) if pending else "The instance is over.",
                          ends=" This conversation ends with it; only what you save outlasts it." if conversation_ends else "",
                          note=f" {note}" if note else "")


REPROMPT = (
    "I could not find a valid action in your reply ({reason}). Reply again with exactly one of\n"
    '{{"action": "submit", "grid": [[...]]}}, or {{"action": "request_demos"}}\n'
    "as a JSON object on the last line."
)

REPROMPT_FINISH = (
    "I could not find a valid action in your reply ({reason}). Reply again with exactly one of\n"
    '{{"action": "submit", "grid": [[...]]}}, {{"action": "request_demos"}} or {{"action": "finish"}}\n'
    "as a JSON object on the last line."
)

OVER_REPROMPT = (
    "The instance is over ({reason}). The only action left is {{\"action\": \"finish\"}}, as a JSON object on the last "
    "line."
)

# The ablation row HA (a system's ask_to_store, 29 Sep, Dan): the finish protocol and its facts-only rules, with this ask
# after the verdict, as liberal as we could make it, so that the row shows the most that asking a harness to store can
# give (Dan's wording; "only demonstrations and wrong attempts are scored" is ours, the fact that makes "no risk" true).
# {note}: what the harness keeps, in its own terms (storage_note; prime-agent's global_=True).
ASK_TO_STORE = (
    "Before you finish, you should liberally save *anything* that *could* potentially help you with future instances of "
    "this task, using your own tools.{note} There is no risk in being too verbose: only demonstrations and wrong attempts "
    "are scored. Details can always be pruned later if they're surplus to requirements. When you are done, reply with "
    '{{"action": "finish"}}.'
)

# The ablation row HPL, a harness-driven program library (a system's program_library, 29 Sep, Dan: "literally instruct the
# model exactly how to behave, in a similar manner to the program library"): the finish protocol's facts-only rules, then
# row M's method as a procedure the harness carries out itself with its own tools (M's scaffold writes, checks, keeps and
# replays the program; here the harness does). {programs}: the library's folder, in the harness's workspace, which only a
# stateless row wipes.
PROGRAM_LIBRARY_RULES = (
    "How to work, in every instance (follow this exactly): keep a program library in the folder {programs}, one Python "
    "file per task, named for its task id ({programs}/<task id>.py), defining transform(grid), which takes an input grid "
    "as a list of lists of integers and returns the output grid.\n"
    "1. When an instance starts, look for the task's file. If it exists, run its transform on the test input and submit "
    "the result before anything else.\n"
    "2. If there is no file, or its answer was wrong, request demonstration pairs as you judge necessary, write or fix "
    "transform, and run it on every pair you have received until it reproduces all of them; then submit its output for "
    "the test input. Checking a program on the pairs is free.\n"
    "3. Submit only grids that your program produced.\n"
    "4. After the verdict, save the program that reproduced all the pairs to the task's file, replacing any older "
    "version, then finish."
)

PROGRAM_LIBRARY_ASK = (
    "Now save your program for this task to {path}: the transform(grid) that reproduced every demonstration pair you "
    "received, replacing any older version. If no program of yours reproduced them all, leave the file as it was. Then "
    'reply with {{"action": "finish"}}.'
)

BUDGET_NOTICE = (
    "Your previous turn was stopped because it exceeded the time or model-call budget for a "
    "single action. Decide now from what you already know, without further exploration."
)


@dataclass
class Transcript:
    """Feedback lines waiting to be shown to the system on its next turn."""

    pending: list[str] = field(default_factory=list)

    def add(self, action: Action, feedback: Feedback) -> None:
        self.pending.append(render_feedback(action, feedback))

    def add_text(self, line: str) -> None:
        self.pending.append(line)

    def take(self) -> list[str]:
        lines, self.pending = self.pending, []
        return lines
