"""Model-call tracking as hooks, so it can be attached to any system.

The benchmark tests *systems* (prime-agent, opencode, hermes-agent, ...), not
models, and a system calls its model however it likes. The only assumption we
make is OpenAI/OpenRouter compatibility at the HTTP boundary. Tracking therefore
lives at that boundary: :mod:`continual_arc_baselines.proxy` observes every
physical request and reports it here as an :class:`LLMCall`; anything can
subscribe to those calls through a :class:`Tracker`.

Two subscribers ship with the package:

* :class:`Usage` keeps cumulative counters shaped for
  :meth:`continual_arc.learner.Learner.resources`, so per-instance tokens and
  dollars land in the run records without the system counting anything.
* :class:`JournalSink` writes each call into the workbench's cost journal and
  event log in the same shape as the workbench's own OpenRouter client, so
  ``workbench costs`` and MLflow sync work unchanged.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from continual_arc.env import GiveUpFeedback, SubmitFeedback

log = logging.getLogger(__name__)

TOKEN_FIELDS = ("input", "output", "cache_read", "cache_write", "reasoning")


def instance_ended(feedback: Any) -> bool:
    """Whether ``feedback`` ends its instance: a give-up; a correct submission, or the one that
    reaches the attempt cap."""
    return isinstance(feedback, GiveUpFeedback) or (
        isinstance(feedback, SubmitFeedback) and (feedback.correct or feedback.failed)
    )


@dataclass
class LLMCall:
    """One physical model request as seen at the HTTP boundary.

    ``status`` is ``started`` when the request is forwarded, then exactly one of
    ``completed`` (a response with a body came back), ``failed`` (transport
    error or non-2xx) or ``cancelled`` (the client went away mid-stream).
    Token counts and cost are provider-reported or ``None``; nothing is estimated.
    """

    call_id: str
    status: str
    path: str
    requested_model: str | None = None
    model: str | None = None
    generation_id: str | None = None
    stream: bool = False
    tokens: dict[str, int | None] = field(default_factory=lambda: dict.fromkeys(TOKEN_FIELDS))
    cost: Decimal | None = None
    latency_ms: float | None = None
    status_code: int | None = None
    error: str | None = None
    finish_reason: str | None = None
    response_headers: dict[str, str] = field(default_factory=dict)
    request_body: dict[str, Any] | None = None
    response_body: dict[str, Any] | None = None

    @property
    def priced(self) -> bool:
        return self.cost is not None


@dataclass
class Usage:
    """Cumulative counters over physical requests, as ``resources()`` keys."""

    requests: int = 0
    failed_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: Decimal = Decimal("0")
    unpriced_requests: int = 0  # completed requests with no provider-reported charge

    def update(self, call: LLMCall) -> None:
        if call.status == "started":
            self.requests += 1
            return
        if call.status in ("failed", "cancelled"):
            self.failed_requests += 1
        self.input_tokens += call.tokens.get("input") or 0
        self.output_tokens += call.tokens.get("output") or 0
        self.reasoning_tokens += call.tokens.get("reasoning") or 0
        if call.cost is not None:
            self.cost_usd += call.cost
        elif call.status == "completed":
            self.unpriced_requests += 1

    def as_resources(self) -> dict[str, float]:
        return {
            "llm_requests": float(self.requests),
            "llm_failed_requests": float(self.failed_requests),
            "llm_input_tokens": float(self.input_tokens),
            "llm_output_tokens": float(self.output_tokens),
            "llm_reasoning_tokens": float(self.reasoning_tokens),
            "llm_cost_usd": float(self.cost_usd),
            "llm_unpriced_requests": float(self.unpriced_requests),
        }


Hook = Callable[[LLMCall], None]


class Tracker:
    """A thread-safe list of hooks; :attr:`usage` is always subscribed.

    :attr:`last_activity` is a monotonic timestamp of the last byte the proxy saw
    move in either direction. Adapters use it to tell a slow model call (bytes are
    still flowing) from a hung system (nothing is). The tracker also keeps the calls
    started and not yet ended, so a caller can wait for them (:meth:`wait_ended`).

    :attr:`credit_s` is the seconds the proxy's calls spent on work that is not the
    model's reply (a call it answers itself through ``LLMProxy.serve``), which an
    adapter leaves out of a turn's time budget (:meth:`credit`).
    """

    def __init__(self) -> None:
        self.usage = Usage()
        self._hooks: list[Hook] = []
        self._lock = threading.Lock()
        self._reported = threading.Condition(self._lock)  # notified as each call's end is reported
        self._open: dict[str, float] = {}  # the calls not yet ended: when each started (time.monotonic())
        self.last_activity = time.monotonic()
        self.credit_s = 0.0

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def credit(self, seconds: float) -> None:
        """Seconds of a call not to count against its turn's budget (added to :attr:`credit_s`)."""
        if seconds > 0:
            with self._lock:
                self.credit_s += float(seconds)

    def subscribe(self, hook: Hook) -> Callable[[], None]:
        with self._lock:
            self._hooks.append(hook)

        def unsubscribe() -> None:
            with self._lock:
                if hook in self._hooks:
                    self._hooks.remove(hook)

        return unsubscribe

    def emit(self, call: LLMCall) -> None:
        self.touch()
        with self._lock:
            self.usage.update(call)
            if call.status == "started":
                self._open[call.call_id] = time.monotonic()
            else:
                self._open.pop(call.call_id, None)
                self._reported.notify_all()
            hooks = list(self._hooks)
        for hook in hooks:
            try:
                hook(call)
            except Exception:  # a broken subscriber must not break the run
                log.exception("tracking hook %r failed on call %s", hook, call.call_id)

    def wait_ended(self, since: float, timeout: float) -> dict[str, float]:
        """Wait, ``timeout`` seconds at most, until every call started at or after ``since``
        (``time.monotonic()``) has reported its end, and so is in :attr:`usage`. Returns the
        calls still open then, with the seconds each has been open."""

        def still_open() -> dict[str, float]:
            return {cid: t0 for cid, t0 in self._open.items() if t0 >= since}

        with self._reported:
            self._reported.wait_for(lambda: not still_open(), timeout=max(0.0, timeout))
            now = time.monotonic()
            return {cid: now - t0 for cid, t0 in still_open().items()}


class JournalSink:
    """Mirror calls into a workbench :class:`RunContext` and its cost journal.

    Records follow the workbench's OpenRouter client: a ``request_started``
    observation at dispatch, then a ``response`` (or ``transport_error``)
    observation under the same ``request_attempt_id``, so the journal's summary
    resolves each physical request to one entity.
    """

    def __init__(self, context: Any, provider: str = "openrouter", journal: Any = None,
                 capture_content: bool = False):
        from continual_workbench.accounting import CostJournal

        self.context = context
        self.provider = provider
        self.journal = journal or CostJournal(context.directory)
        self.capture_content = capture_content

    def __call__(self, call: LLMCall) -> None:
        identity = {"run_id": self.context.run_id, "attempt_id": self.context.attempt_id,
                    "call_id": call.call_id, "request_attempt_id": call.call_id}
        base = {"provider": self.provider, "currency": "USD", "unit": "openrouter_credits",
                "upstream_unit": "USD", "requested_model": call.requested_model,
                "model": call.model or call.requested_model, "path": call.path, "stream": call.stream}
        if call.status == "started":
            record = self.journal.record({**identity, **base, "origin": "request_started",
                                          "measurement": "unknown", "status": "started",
                                          "tokens": dict.fromkeys(TOKEN_FIELDS)})
            self.context.emit("llm_request", record)
            if self.capture_content and call.request_body is not None:
                self.context.emit("llm_content", {**identity, "direction": "request",
                                                  "body": call.request_body})
            return
        completed = call.status == "completed"
        record = self.journal.record({
            **identity, **base,
            "origin": "response" if completed else "transport_error",
            "measurement": "provider_reported" if completed else "unknown",
            "status": call.status, "generation_id": call.generation_id,
            "finish_reason": call.finish_reason, "latency_ms": call.latency_ms,
            "status_code": call.status_code, "error_type": call.error,
            "tokens": {k: call.tokens.get(k) for k in TOKEN_FIELDS},
            "account_charge": format(call.cost, "f") if call.cost is not None else None,
            "upstream_inference_cost": None,
            "request_id": call.response_headers.get("x-request-id"),
        })
        self.context.emit("llm_usage" if completed else "llm_request", record)
        if self.capture_content and call.response_body is not None:
            self.context.emit("llm_content", {**identity, "direction": "response",
                                              "body": call.response_body})


class PilotStop(Exception):
    """The pilot gate's stop: ``stop_after`` instances are done and the next one has not begun (TrackedLearner)."""

    def __init__(self, done: int):
        super().__init__(f"stopped after {done} instances for the pilot gate")
        self.done = done


class TrackedLearner:
    """Wrap any learner so its ``resources()`` include the tracker's counters.

    The learner needs no knowledge of the proxy: the runner reads cumulative
    counters before every action and after its feedback, and charges the
    differences to the instance. A call's tokens and cost are counted when the
    proxy reports its end, which it does once it has written the reply, so the
    learner can take the reply and end the instance before the report comes. At
    an instance's end (:func:`instance_ended`, the rule by which the adapters
    close an instance), before the runner reads the counters, the wrapper waits
    until every call started since the instance's first action has reported its
    end, ``calls_wait_s`` at most: a call still open then is logged as an error,
    and the instance's record is charged without its tokens and cost. Only the
    end waits, since a harness may leave a call running between actions
    (prime-agent compacts in the background), and a call started before the
    instance began holds up no instance.
    """

    def __init__(self, learner: Any, tracker: Tracker, *, calls_wait_s: float = 600.0,
                 stop_after: int | None = None, done: int = 0):
        self.learner = learner
        self.tracker = tracker
        # The pilot gate (CLI --stop-after): once ``stop_after`` instances are done (``done`` of them before a resume),
        # the next instance's first action raises PilotStop, after the runner has written the last record and before
        # the new instance has taken any action, so a resume replays nothing.
        self.stop_after = stop_after
        self._done = done
        # as the collective learner waits: a call normally ends within milliseconds of the
        # reply, but one the harness left running (a compaction) can take minutes
        self.calls_wait_s = calls_wait_s
        self._since: float | None = None  # when the open instance's first action began

    def reset(self) -> None:
        self.learner.reset()

    def begin_round(self, n_instances: int) -> None:
        self.learner.begin_round(n_instances)

    def end_round(self) -> None:
        self.learner.end_round()

    def act(self, obs: Any) -> Any:
        if self._since is None:
            if self.stop_after is not None and self._done >= self.stop_after:
                raise PilotStop(self._done)
            self._since = time.monotonic()
        return self.learner.act(obs)

    def observe(self, obs: Any, action: Any, feedback: Any) -> None:
        if self._since is None:  # an instance the runner ended without asking the learner to act
            self._since = time.monotonic()
        self.learner.observe(obs, action, feedback)
        if not instance_ended(feedback):
            return
        self._done += 1
        since, self._since = self._since, None
        started = time.monotonic()
        still = self.tracker.wait_ended(since, self.calls_wait_s)
        if still:
            log.error("step %d: %d model call(s) started for the instance still open after waiting %.1f s for them "
                      "(%s); its record is charged without their tokens and cost", obs.step, len(still),
                      time.monotonic() - started,
                      ", ".join(f"{cid[:8]}: open {age:.1f} s" for cid, age in still.items()))

    def resources(self) -> dict[str, float]:
        return {**(self.learner.resources() or {}), **self.tracker.usage.as_resources()}
