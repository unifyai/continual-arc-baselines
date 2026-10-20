"""The bare model as a system: the harness protocol with no harness (25 Sep 2026, Dan: each candidate model with and
without prime-agent, the model alone having only the two actions, request demonstrations or submit).

Each observation, rendered exactly as every harness receives it (:mod:`continual_arc_baselines.systems.base`), goes to
the model as the next user message of one conversation per instance, and its reply is parsed for the action as a
harness's is, with the same re-prompts and give-up. There is nothing else: no tools, no code, no memory, no files, one
model call per message. The instance's closing verdict is not sent: it exists for a harness's storage step, a bare
model stores nothing, and at a high reasoning effort it would cost a whole reasoning call.

Calls go to the run's proxy (``ARC_LLM_BASE_URL``), streamed, so a reply that reasons for half an hour is never silent
long enough to meet the proxy's read timeout. Earlier replies stay in the conversation as their text, without their
reasoning, as chat templates render a past turn.

Since 28 Sep the two floors of row F are built on it (Dan: the stateless floor should be "the model itself, allowed to
choose whatever actions it wants", not a policy of ours that buys all eight pairs): :class:`BareModelLearner` is the
stateless floor, and :class:`BareDemoCacheLearner` the demo-cache floor, the same model with a cache of the pairs it
bought for each task, shown again at no cost when the task returns. Both take ``end="finish"`` in the grid, so their
rules are the harness rows' (protocol.RULES_FINISH); the verdict is still not sent.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import httpx
from continual_arc.env import (
    Action,
    DemosFeedback,
    GiveUp,
    LearnerObservation,
    RequestDemos,
    Submit,
    SubmitFeedback,
)

from continual_arc_baselines.learners.prompting import _PY_FENCE, extract_program, run_transform
from continual_arc_baselines.proxy import PLACEHOLDER_KEY
from continual_arc_baselines.systems import protocol
from continual_arc_baselines.systems.base import SystemLearner, SystemUnavailable
from continual_arc_baselines.systems.protocol import _shape, render_grid


class BareModelLearner(SystemLearner):
    """The model alone behind the harness protocol. ``max_tokens`` caps each reply; ``timeout_s`` is the longest
    silence tolerated inside one streamed reply."""

    name = "bare"

    def __init__(self, model: str, *, max_tokens: int = 65_536, timeout_s: float = 3600.0,
                 context_window: int | None = None, **base: Any):
        super().__init__(**base)
        self.model, self.max_tokens, self.timeout_s = model, max_tokens, timeout_s
        self.context_window = context_window  # what a harness is told about its model; a bare model is told nothing
        self.messages: list[dict[str, str]] = []
        self._client: httpx.Client | None = None
        self.stats["truncations"] = 0

    def start(self) -> None:
        self._client = httpx.Client(timeout=httpx.Timeout(connect=30.0, read=self.timeout_s, write=60.0, pool=30.0))
        self.messages = []

    def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self.messages = []

    def new_session(self) -> None:
        self.messages = []
        self._fresh_session = True

    def wipe_state(self) -> None:
        self.messages = []

    def _close_instance(self, obs: Any) -> None:
        self.transcript.take()  # the verdict, with nothing to store it in

    def ask(self, message: str, *, reprompt: bool = False) -> str:
        assert self._client is not None
        if self.messages and self.messages[-1]["role"] == "user":
            # the last message got no reply (empty_reply="reprompt"): it and this one are one turn, as a conversation
            # holds no empty assistant message
            self.messages[-1]["content"] += "\n\n" + message
        else:
            self.messages.append({"role": "user", "content": message})
        url = self.environ["ARC_LLM_BASE_URL"].rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.environ.get('ARC_LLM_API_KEY', PLACEHOLDER_KEY)}"}
        body = {"model": self.model, "messages": self.messages, "max_tokens": self.max_tokens, "stream": True}
        parts: list[str] = []
        finish = None
        failure = None
        try:
            with self._client.stream("POST", url, json=body, headers=headers) as response:
                if response.status_code >= 400:
                    failure = f"HTTP {response.status_code}: {response.read()[:300]!r}"
                else:
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue  # blank lines and keep-alive comments
                        data = line[5:].strip()
                        if data == "[DONE]":
                            continue  # read on to the stream's end, so the connection goes back to the pool
                        chunk = json.loads(data)
                        if chunk.get("error"):
                            failure = f"error in the stream: {json.dumps(chunk['error'])[:300]}"
                            break
                        for choice in chunk.get("choices") or []:
                            parts.append((choice.get("delta") or {}).get("content") or "")
                            finish = choice.get("finish_reason") or finish
        except (httpx.TimeoutException, httpx.ConnectError) as error:
            # silent for timeout_s, or the proxy is gone: the system is down, whatever empty_reply says
            raise SystemUnavailable(f"{type(error).__name__}: {error}") from error
        except (httpx.HTTPError, ValueError) as error:
            failure = f"{type(error).__name__}: {error}"
        self.last_turn_outcome = {"finish_reason": finish, **({"error": failure} if failure else {})}
        if failure is not None:
            if self.empty_reply == "restart":
                raise SystemUnavailable(failure)
            parts = []   # a model call that failed is a turn with no text, re-prompted like one
        reply = "".join(parts)
        if finish == "length":
            self.stats["truncations"] += 1
        if reply:
            self.messages.append({"role": "assistant", "content": reply})
        return reply

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "system": self.name, "model": self.model, "max_tokens": self.max_tokens,
                "timeout_s": self.timeout_s}


class BareDemoCacheLearner(BareModelLearner):
    """The demo-cache floor (28 Sep): the bare model, plus a cache of the demonstration pairs it has bought for each
    task, up to ``cache_cap`` a task (the per-instance demo cap). When a task returns, the first message of the
    instance shows its cached pairs again at no cost, before the instance; the model may still request more, and they
    join the cache. Nothing else is kept: no answers, no verdicts, no conversation. On ARC-AGI-2 a cached pair is in the
    frame of the instance that bought it, which the task's rule survives (continual_arc.tasks, per-task labels)."""

    name = "bare-demo-cache"

    def __init__(self, model: str, *, cache_cap: int = 8, **kwargs: Any):
        super().__init__(model, **kwargs)
        self.cache_cap = cache_cap
        self.cache: dict[str, list] = {}

    def instance_preface(self, obs: LearnerObservation) -> str:
        pairs = self.cache.get(obs.task_id, [])
        if not pairs:
            return ""
        shown = "\n\n".join(f"Earlier pair {k} input ({_shape(p.input)}):\n{render_grid(p.input)}\n"
                             f"Earlier pair {k} output ({_shape(p.output)}):\n{render_grid(p.output)}"
                             for k, p in enumerate(pairs, 1))
        return f"Demonstration pairs of task {obs.task_id} that you received before, shown again at no cost:\n\n{shown}"

    def observe(self, obs: LearnerObservation, action: Any, feedback: Any) -> None:
        if isinstance(feedback, DemosFeedback) and not feedback.refused and feedback.pairs:
            kept = self.cache.setdefault(obs.task_id, [])
            kept.extend(feedback.pairs[: max(self.cache_cap - len(kept), 0)])
        super().observe(obs, action, feedback)

    def wipe_state(self) -> None:
        super().wipe_state()
        self.cache = {}

    def resources(self) -> dict[str, float]:
        out = super().resources()
        out["state_bytes"] = float(len(json.dumps({k: [p.to_json() for p in v] for k, v in self.cache.items()})))
        return out

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "system": self.name, "cache_cap": self.cache_cap}



PROGRAM_NOTE = """\
Here every submission is a program instead of a grid. To submit, write a Python function transform(grid) that
returns the output grid for an input of this task, in one ```python block using only the standard library, and end
your reply with {"action": "submit"}. Your program is run on the test input and its output is submitted. Before that
it is checked, at no cost, on every demonstration pair you have for this task, and a failure is reported to you;
after 3 failed checks in an instance, a program's output is submitted without a check. A program that reproduces all
the pairs you have is kept for the task: when the task returns, it is run on the new test input first and its output
is submitted."""

# Row M0, a stateless program library (29 Sep, Dan): every submission a program, checked as in M, with the demo cache and
# the kept programs wiped before every instance (stateless=True), so its rules drop the one sentence about keeping.
PROGRAM_NOTE_STATELESS = PROGRAM_NOTE[:PROGRAM_NOTE.index(" A program that reproduces all")]

PROGRAM_REPROMPT = ('I could not find a valid action in your reply ({reason}). Reply with your program in one '
                    '```python block and {{"action": "submit"}} on the last line, or with {{"action": "request_demos"}}.')


class BareProgramLibraryLearner(BareDemoCacheLearner):
    """Row M since 28 Sep (Dan: the program library as the demo-cache floor, "but the model *must* create one program for
    each task"; the model decides every demonstration request). The bare demo-cache floor, with one requirement: every
    submission is a program. The model writes ``transform(grid)`` in a Python block and ends with the submit action; the
    program runs on the test input in the harnesses' sandbox and its output is submitted, and a grid alone is re-prompted.
    Before that, the program is checked at no cost on every pair the model has for the task (the cached pairs shown at
    the start of the instance, then the pairs bought in it), and a failure goes back to it; after ``max_checks`` failed
    checks in an instance a program's output is submitted without a check, so no instance can loop. A program that
    reproduces every pair is kept for the task, the latest replacing the one before. When the task returns, the kept
    program runs on the new test input first and its output is submitted before the model is asked (no model call); if
    that is wrong, the model sees the program, the cached pairs and the failure. The rules are the harness rows' with
    :data:`PROGRAM_NOTE`; no verdict turn."""

    name = "bare-program-library"

    def __init__(self, model: str, *, max_checks: int = 3, program_timeout_s: float = 10.0, sandbox: str = "auto",
                 **kwargs: Any):
        kwargs.setdefault("end", "finish")
        super().__init__(model, **kwargs)
        self.max_checks = max_checks
        self.program_timeout_s = program_timeout_s
        self.sandbox = sandbox
        self.library: dict[str, str] = {}
        self._programs = Path(tempfile.mkdtemp(prefix="arc-bare-programs-"))
        self._instance: tuple | None = None
        self._earlier: list = []          # the cached pairs shown at the start of the instance
        self._bought: list = []           # the pairs bought in the instance
        self._checks = 0
        self._kept_run: tuple[str, str] | None = None   # (what happened, program) for the kept program's first run
        self._kept_pending = False
        self.stats.update({"programs_kept": 0, "checks_failed": 0, "program_errors": 0, "grid_refusals": 0,
                           "library_runs": 0, "library_hits": 0, "library_misses": 0})

    def run(self, program: str, grid: Any) -> tuple[Any, str | None]:
        return run_transform(program, grid, sandbox=self.sandbox, timeout_s=self.program_timeout_s,
                             workdir=self._programs)

    # -- what the model is shown ------------------------------------------------

    def instance_preface(self, obs: LearnerObservation) -> str:
        parts = []
        if self._earlier:
            shown = "\n\n".join(f"Earlier pair {k} input ({_shape(p.input)}):\n{render_grid(p.input)}\n"
                                 f"Earlier pair {k} output ({_shape(p.output)}):\n{render_grid(p.output)}"
                                 for k, p in enumerate(self._earlier, 1))
            parts.append(f"Demonstration pairs of task {obs.task_id} that you received before, shown again at no cost:\n\n"
                         f"{shown}")
        if self._kept_run is not None:
            what, program = self._kept_run
            parts.append(f"The program kept for task {obs.task_id} from an earlier instance {what}:\n"
                         f"```python\n{program.rstrip()}\n```")
        return "\n\n".join(parts)

    def _known_pairs(self) -> list[tuple[str, Any]]:
        return ([(f"earlier pair {k}", p) for k, p in enumerate(self._earlier, 1)]
                + [(f"demo pair {k}", p) for k, p in enumerate(self._bought, 1)])

    def _check(self, program: str) -> str | None:
        for name, pair in self._known_pairs():
            got, error = self.run(program, pair.input)
            if error is not None:
                return f"Your program was checked on {name} at no cost and failed ({error})."
            if got != pair.output:
                return (f"Your program was checked on {name} at no cost and its output is wrong. It gave\n"
                        f"{render_grid(got)}\nwhere the pair's output is\n{render_grid(pair.output)}")
        return None

    def _interpret(self, obs: LearnerObservation, reply: str) -> tuple[str, Any]:
        """('action', an action), ('check', a failed check to send back) or ('unusable', why)."""
        program = extract_program(reply)
        rest = _PY_FENCE.sub("", reply) if program else reply
        try:
            parsed: Any = protocol.parse_action(rest, finish=True)
            reason = ""
        except protocol.ParseError as error:
            parsed, reason = None, str(error)
        if isinstance(parsed, RequestDemos):
            return "action", parsed
        if isinstance(parsed, protocol.Finish):
            self.stats["finish_refusals"] += 1
            return "unusable", "finish is refused until the instance is over"
        if program is None or "def transform" not in program:
            if isinstance(parsed, Submit):
                self.stats["grid_refusals"] += 1
                return "unusable", "a submission here is a program in a ```python block, not a grid"
            return "unusable", reason or "no program and no demo request"
        problem = self._check(program)
        if problem is not None and self._checks < self.max_checks:
            self._checks += 1
            self.stats["checks_failed"] += 1
            return "check", (f"{problem}\nFailed checks in this instance: {self._checks} of {self.max_checks}. Reply "
                             'with a corrected program and {"action": "submit"}, or with {"action": "request_demos"}.')
        if problem is None:
            self.library[obs.task_id] = program
            self.stats["programs_kept"] += 1
        got, error = self.run(program, obs.input)
        if error is not None:
            self.stats["program_errors"] += 1
            if self._checks < self.max_checks:
                self._checks += 1
                return "check", (f"Your program failed on the test input ({error}), so nothing was submitted. Failed "
                                 f"checks in this instance: {self._checks} of {self.max_checks}.")
            return "action", Submit([[10]])   # no output to submit: charged like a wrong answer
        return "action", Submit(got)

    # -- the loop ---------------------------------------------------------------

    def act(self, obs: LearnerObservation) -> Action:
        if obs.is_fresh and self._session_pending:
            self._session_pending = False
            if self.stateless:
                # row M0: the kept programs and the demo cache are wiped before every instance, as a stateless row's
                # state is (SystemLearner.act)
                self.reset()
                self.stats["wipes"] += 1
                self.stats["sessions"] += 1
            elif self._started:
                try:
                    self.new_session()
                    self.stats["sessions"] += 1
                except SystemUnavailable as error:
                    self.note(f"step {obs.step}: new session failed ({error}); restarting the system")
                    self.close()
        if self._instance != (obs.task_id, obs.step):
            self._instance, self._checks, self._kept_run, self._bought = (obs.task_id, obs.step), 0, None, []
            self._earlier = list(self.cache.get(obs.task_id, []))
            program = self.library.get(obs.task_id)
            if program is not None:
                self.stats["library_runs"] += 1
                got, error = self.run(program, obs.input)
                if error is None:
                    self._kept_run = ("was run on this test input first, and its output was submitted", program)
                    self._kept_pending = True
                    return Submit(got)
                self._kept_run = (f"failed on this test input ({error}), so nothing was submitted", program)
        pending = self.transcript.take()

        def observation(fresh: bool) -> str:
            preface = self.instance_preface(obs) if fresh else ""
            return protocol.render_observation(obs, pending=pending, demo_cost=self.demo_cost,
                                               first_in_session=fresh, finish=True,
                                               rules_note=PROGRAM_NOTE_STATELESS if self.stateless else PROGRAM_NOTE,
                                               preface=preface)

        try:
            reply = self._ask_resilient(observation, kind="observation", step=obs.step, task=obs.task_id)
            unusable = 0
            while True:
                kind, value = self._interpret(obs, reply)
                if kind == "action":
                    return value
                if kind == "unusable":
                    self.stats["parse_failures"] += 1
                    self.note(f"step {obs.step}: unusable reply ({value}); tail: {reply[-300:]!r}")
                    if unusable == self.max_reprompts:
                        self.note(f"step {obs.step}: giving up after {unusable + 1} unusable replies")
                        return GiveUp()
                    unusable += 1
                    self.stats["reprompts"] += 1
                    message = PROGRAM_REPROMPT.format(reason=value)
                else:
                    message = value

                def follow_up(fresh: bool, message: str = message) -> str:
                    return observation(True) + "\n\n" + message if fresh else message

                reply = self._ask_resilient(follow_up, reprompt=True, kind="check" if kind == "check" else "reprompt",
                                            step=obs.step, task=obs.task_id)
        except SystemUnavailable as error:
            self.stats["forced_give_ups"] += 1
            self.note(f"step {obs.step}: system unavailable after {self.max_restarts} restart(s) "
                      f"({error}); this action is a give-up")
        return GiveUp()

    def observe(self, obs: LearnerObservation, action: Any, feedback: Any) -> None:
        if isinstance(feedback, DemosFeedback) and not feedback.refused and feedback.pairs:
            self._bought.extend(feedback.pairs)
        if self._kept_pending and isinstance(feedback, SubmitFeedback):
            self._kept_pending = False
            self.stats["library_hits" if feedback.correct else "library_misses"] += 1
        super().observe(obs, action, feedback)

    def wipe_state(self) -> None:
        super().wipe_state()
        self.library = {}

    def resources(self) -> dict[str, float]:
        out = super().resources()
        out["state_bytes"] += float(sum(len(p) for p in self.library.values()))
        return out

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "system": self.name, "max_checks": self.max_checks,
                "program_timeout_s": self.program_timeout_s, "sandbox": self.sandbox}
