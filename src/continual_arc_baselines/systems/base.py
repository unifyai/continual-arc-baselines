"""The adapter base: one conversation with one long-running system.

Subclasses implement how to start, prompt, reset and stop the system. The base
handles the benchmark side: rendering observations, delivering feedback,
parsing actions, re-prompting on protocol violations, restarting a system that
stops answering, and wiping state on ``reset``. It counts nothing; usage comes
from the proxy.

A system sees what the runner shows every learner: the instance alone, its task
under the run's name (``task-<hex>``, :mod:`continual_arc.aliases`), which the
adapter passes on as it is.

One policy lives here because it applies to every system: **a system failure
costs one action, not the run.** If the system cannot be brought back after
``max_restarts`` restarts, the action becomes a give-up, counted in
``system_forced_give_ups``, and the run continues.

Every turn the adapter logs (``turns.jsonl``) carries how it ended when the adapter knows it
(``outcome``: the harness's stop reason and its own error message), and a launch since 28 Sep
re-prompts an empty reply rather than restarting the system (``empty_reply``).
"""

from __future__ import annotations

import json
import logging
import os
import tarfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from continual_arc.env import Action, Feedback, GiveUp, LearnerObservation
from continual_arc.learner import BaseLearner

from continual_arc_baselines.systems import protocol
from continual_arc_baselines.systems.protocol import Transcript
from continual_arc_baselines.tracking import instance_ended

log = logging.getLogger(__name__)


END_MODES = ("closing", "storage", "finish")
"""How a harness row's instance ends (SystemLearner ``end``)."""

EMPTY_REPLY_MODES = ("restart", "reprompt")
"""What an empty reply to an observation or a re-prompt does (SystemLearner ``empty_reply``)."""

class SystemUnavailable(RuntimeError):
    """The system process died or stopped answering."""


class SystemLearner(BaseLearner):
    """A learner that forwards the benchmark conversation to an external system.

    Lifecycle: :meth:`attach` (where to keep state, what environment children
    inherit, the run's tracker) is called once by the CLI before the run;
    :meth:`start` lazily on the first action; :meth:`reset` stops, wipes and
    lazily restarts; :meth:`stop` at the end. ``demo_cost`` is only for the
    wording of the rules.
    """

    name = "system"
    # What the harness keeps between instances, in its own terms, for the storage turn's message (storage_turn): the
    # facts its agent needs to save anything that lasts, and cannot know otherwise.
    storage_note = ""

    def __init__(self, *, max_reprompts: int = 2, refine_every: int = 0, demo_cost: float = 1.0,
                 max_restarts: int = 2, session_per_instance: bool = True, instance_gap_s: float = 0.0,
                 stateless: bool = False, storage_turn: bool = False, end: str = "", empty_reply: str = "restart",
                 archive_wipes: bool = False, ask_to_store: bool = False, program_library: bool = False):
        self.max_reprompts = max_reprompts
        self.refine_every = refine_every
        # How an instance ends. Off: a closing turn with the verdict and "no action is needed" (the harness rows since 23
        # Sep). On: the verdict and a turn of the agent's own in which to save what will help later, with whatever tools
        # its harness has (protocol.render_storage; the team rows since 26 Sep, Dan: many harnesses store at will, and the
        # turn that submits ends before the verdict comes, so without it nothing could be stored knowing the outcome).
        # Never for a stateless row, whose state could not survive it.
        # ``end`` names how an instance ends: "closing" and "storage" are the two above (storage_turn picks between them
        # when ``end`` is not given); "finish" (the harness rows of the grid since 27 Sep, Dan: "the harnesses should be
        # free to do what they want") sends the verdict alone, with no instruction, and the harness ends the instance
        # itself with the finish action, which is refused until the instance is over. The rules state facts only
        # (protocol.RULES_FINISH). Rows H and H0 are treated alike: nothing is said to one that is not said to the other.
        self.end = end or ("storage" if storage_turn else "closing")
        if self.end not in END_MODES:
            raise ValueError(f"end must be one of {END_MODES}, got {end!r}")
        self.storage_turn = self.end == "storage"
        # The ablation row HA (29 Sep, Dan): the finish protocol with one change, an ask to save liberally after the
        # verdict (protocol.ASK_TO_STORE), so that the row shows the most that asking a harness to
        # store can give. Never for a stateless row, whose state could not survive it.
        if ask_to_store and self.end != "finish":
            raise ValueError("ask_to_store is an addition to the finish protocol (end='finish')")
        self.ask_to_store = ask_to_store
        # The ablation row HPL (29 Sep, Dan): a harness-driven program library. The finish protocol's rules gain row M's
        # method as a procedure (protocol.PROGRAM_LIBRARY_RULES: a program per task in a folder of the workspace, run
        # first when the task returns, checked on every pair before a submission, saved after the verdict) and the
        # verdict is followed by the ask to save the task's program (PROGRAM_LIBRARY_ASK). A
        # stateless row gets the same messages, so that HPL0 would differ from HPL only in what survives.
        if program_library and (self.end != "finish" or ask_to_store):
            raise ValueError("program_library is an addition to the finish protocol, and not beside ask_to_store")
        self.program_library = program_library
        # An empty reply to an observation or a re-prompt. "restart" (every cell launched before 28 Sep): the system is
        # taken for dead and restarted, state intact. "reprompt" (every launch since 28 Sep, Dan): it is an unusable reply
        # like any other, re-prompted in the same conversation within the same max_reprompts; the system is restarted
        # only when the adapter finds its process dead or silent (SystemUnavailable), or when it has answered nothing at
        # all for longer than one action's re-prompts (max_reprompts + 2 empty replies in a row), which is silence too.
        # openclaw's empty replies were a turn whose work had gone to subagents, a model call that failed, turns of tool
        # calls alone, and text that never reached the adapter (28 Sep): a restart killed what the
        # harness had in flight and sent the observation again, which a re-prompt does not.
        if empty_reply not in EMPTY_REPLY_MODES:
            raise ValueError(f"empty_reply must be one of {EMPTY_REPLY_MODES}, got {empty_reply!r}")
        self.empty_reply = empty_reply
        self._empty_in_a_row = 0
        # Before a stateless row's wipe (and a fresh cell's reset), copy the system's own transcripts (transcript_paths)
        # to workdir/wipe-archive/, outside the system's reach, so its turns can be read afterwards as row H's can
        # (since 28 Sep: H0 deleted openclaw's transcripts with its state, so its empty replies could not be diagnosed).
        self.archive_wipes = archive_wipes
        # Row H0: the same harness with its own persistence switched off. Before every instance the
        # system is stopped, everything it wrote is deleted (the fresh mode's wipe: home, memory,
        # skills, sessions, workspace, tmp; only the adapter's provider configuration survives) and
        # it is started again. Its end-of-instance storage step and idle gap are skipped, since
        # nothing could survive them. Paired against row H on the same schedule, the difference is
        # the value of persistence with the harness's within-instance skill held equal (23 Sep).
        self.stateless = stateless
        if stateless:
            instance_gap_s = 0.0
        # Each instance is a new piece of work: the system gets a fresh conversation for it and
        # keeps only what it persisted itself (files, memory, skills). The gap is idle wall-clock
        # after an instance, for systems whose housekeeping waits for quiet (openclaw).
        self.session_per_instance = session_per_instance
        self.instance_gap_s = instance_gap_s
        self._session_pending = False
        self.demo_cost = demo_cost
        self.max_restarts = max_restarts
        self.workdir: Path = Path(".workbench") / "system"
        self.environ: dict[str, str] = dict(os.environ)
        self.tracker: Any = None
        self.transcript = Transcript()
        self._started = False
        self._fresh_session = True
        self.last_turn_cut = False  # set by an adapter that aborted the last turn for budget
        # How the last turn ended, as the adapter knows it (stop reason, the harness's error message, ...): logged with
        # the turn and cleared (log_turn). Adapters set it in ask().
        self.last_turn_outcome: dict[str, Any] = {}
        self._instances_done = 0
        self.stats = {"reprompts": 0, "parse_failures": 0, "restarts": 0, "resets": 0,
                      "refines": 0, "forced_give_ups": 0, "budget_hits": 0, "stalls": 0,
                      "idle_timeouts": 0, "provider_retries": 0,
                      "sessions": 0, "wipes": 0, "closings": 0, "storage_turns": 0,
                      "finishes": 0, "finish_refusals": 0, "finish_reprompts": 0, "finish_forced": 0,
                      "finish_budget_hits": 0, "empty_replies": 0, "wipe_archives": 0, "store_asks": 0,
                      "library_asks": 0}
        self._turns = None      # turns.jsonl, opened on first use

    # -- to implement --------------------------------------------------------

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def ask(self, message: str, *, reprompt: bool = False) -> str:
        """Send one message, return the system's final reply text.

        ``reprompt`` marks the short follow-up after an unusable reply; systems
        with per-action budgets give it a smaller one.
        """
        raise NotImplementedError

    def instance_preface(self, obs: LearnerObservation) -> str:
        """Text put before the instance in the first message of its conversation (protocol.render_observation's
        ``preface``): nothing for a harness row; the demo-cache floor's cached pairs (systems/bare.py, 28 Sep)."""
        return ""

    def after_instance(self) -> None:
        """Hook at the end of an instance, before its conversation is closed: a system's own
        explicit storage step goes here (prime-agent's global refine)."""

    def new_session(self) -> None:
        """Start a fresh conversation without touching persistent state."""
        raise NotImplementedError

    def wipe_state(self) -> None:
        """Delete everything the system persisted. Called while stopped."""
        raise NotImplementedError

    def transcript_paths(self) -> list[Path]:
        """The system's own transcripts and logs (not its caches, environments or skills), copied out before a wipe
        when ``archive_wipes`` is on. Nothing by default; a harness adapter names its own."""
        return []

    def refine(self) -> None:
        """Optional self-improvement step between instances."""

    def describe(self) -> dict[str, Any]:
        """Identity for the run record: system name, version, model, settings."""
        return {"system": self.name, "max_restarts": self.max_restarts,
                "session_per_instance": self.session_per_instance, "instance_gap_s": self.instance_gap_s, "stateless": self.stateless, "storage_turn": self.storage_turn,
                "end": self.end, "empty_reply": self.empty_reply, "archive_wipes": self.archive_wipes,
                "ask_to_store": self.ask_to_store, "program_library": self.program_library}

    def system_usage(self) -> dict[str, float]:
        """What the system itself reports about its usage (secondary to the proxy)."""
        return {}

    # -- lifecycle -----------------------------------------------------------

    def attach(self, workdir: Path, environ: dict[str, str], tracker: Any = None) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.environ = {**os.environ, **environ}
        self.tracker = tracker

    def _ensure_started(self) -> None:
        if not self._started:
            self.start()
            self._started = True
            self._fresh_session = True

    def reset(self) -> None:
        if self._started:
            self.stop()
            self._started = False
        self._archive_before_wipe()
        self.wipe_state()
        self.transcript = Transcript()
        self.stats["resets"] += 1

    def _archive_before_wipe(self) -> None:
        """The last instance's transcripts, into ``workdir/wipe-archive/instance-NNNN.tar.gz`` (the system is stopped)."""
        if not self.archive_wipes or self._instances_done == 0:
            return
        paths = [p for p in self.transcript_paths() if p.exists()]
        if not paths:
            return
        dest = self.workdir / "wipe-archive" / f"instance-{self._instances_done - 1:04d}.tar.gz"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(dest, "w:gz") as tar:
                for path in paths:
                    tar.add(path, arcname=str(path.relative_to(self.workdir)) if path.is_relative_to(self.workdir)
                            else path.name)
            self.stats["wipe_archives"] += 1
        except OSError as error:
            self.note(f"archive before the wipe failed: {error}")

    def close(self) -> None:
        if self._started:
            self.stop()
            self._started = False
        if self._turns is not None:
            self._turns.close()
            self._turns = None

    # -- the benchmark loop --------------------------------------------------

    def note(self, message: str) -> None:
        """Diagnostics hook; systems may override to keep a log with the evidence."""
        log.warning(message)

    def log_turn(self, kind: str, message: str, reply: str, **meta: Any) -> None:
        """The protocol conversation, one line per exchange, in ``turns.jsonl`` beside the
        run: what the adapter sent, what came back, when, and which instance it belonged to.
        The harnesses' own event streams and stores are partial records by their design
        (prime-agent's daemon injects its digest only when the store changed and drops
        events under load; the ACP harnesses return one text per turn and keep the rest
        in their own databases), so the adapter, which sees every exchange for every
        harness, is the one place a complete record can be written (23 Sep)."""
        try:
            if self._turns is None:
                self._turns = (self.workdir / "turns.jsonl").open("a")
            outcome = {"outcome": self.last_turn_outcome} if self.last_turn_outcome else {}
            self._turns.write(json.dumps({"t": time.time(), "kind": kind, "instance": self._instances_done,
                                          "session": self.stats.get("sessions", 0), **meta, **outcome,
                                          "message": message, "reply": reply}, default=str) + "\n")
            self._turns.flush()
        except OSError as error:
            log.warning("turn log: %s", error)
        self.last_turn_outcome = {}

    def _ask_resilient(self, render: Callable[[bool], str], *, reprompt: bool = False, kind: str = "turn",
                       **meta: Any) -> str:
        """Ask, restarting the system up to ``max_restarts`` times if it fails.

        ``render(fresh)`` produces the message; ``fresh`` is True when the system
        has just (re)started and needs the rules again.
        """
        for attempt in range(self.max_restarts + 1):
            try:
                self._ensure_started()
                fresh, self._fresh_session = self._fresh_session, False
                message = render(fresh)
                started = time.monotonic()
                self.last_turn_outcome = {}
                reply = self.ask(message, reprompt=reprompt)
                self.log_turn("reprompt" if reprompt else kind, message, reply, seconds=round(time.monotonic() - started, 2),
                              fresh=fresh, **meta)
                if reply.strip() or self.last_turn_cut:
                    # a turn the adapter cut short for budget was the system working, and its follow-up carries the
                    # budget notice
                    self._empty_in_a_row = 0
                    return reply
                self.stats["empty_replies"] += 1
                self._empty_in_a_row += 1
                if self.empty_reply == "restart":
                    # Cells launched before 28 Sep: a system that answers with nothing is taken to have stopped
                    # functioning as surely as one that crashed (its context exhausted, or its model call failed
                    # inside it), and is restarted, state intact.
                    raise SystemUnavailable("empty reply")
                if self._empty_in_a_row > self.max_reprompts + 1:
                    # nothing at all for longer than one action's re-prompts: silence, as a dead process is
                    self._empty_in_a_row = 0
                    raise SystemUnavailable(f"no text in {self.max_reprompts + 2} replies in a row")
                # an unusable reply like any other: the caller's parse refuses it and re-prompts
                return reply
            except SystemUnavailable as error:
                self.close()
                if attempt == self.max_restarts:
                    raise
                self.stats["restarts"] += 1
                self.note(f"system unavailable ({error}); restart {attempt + 1}/{self.max_restarts}")
        raise SystemUnavailable("unreachable")

    def act(self, obs: LearnerObservation) -> Action:
        if obs.is_fresh and self._session_pending:
            self._session_pending = False
            if self.stateless:
                # stop, delete everything the system wrote, start again on the next ask
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
        pending = self.transcript.take()
        finish = self.end == "finish"
        note = self.storage_note
        if self.program_library:
            note = "\n".join(x for x in (note, protocol.PROGRAM_LIBRARY_RULES.format(programs=self.programs_dir())) if x)
        extra = {"finish": True, "rules_note": note} if finish else {}

        def observation(fresh: bool) -> str:
            preface = self.instance_preface(obs) if fresh else ""
            return protocol.render_observation(obs, pending=pending, demo_cost=self.demo_cost, first_in_session=fresh,
                                               **({**extra, "preface": preface} if preface else extra))

        try:
            reply = self._ask_resilient(observation, kind="observation", step=obs.step, task=obs.task_id)
            for attempt in range(self.max_reprompts + 1):
                try:
                    parsed = protocol.parse_action(reply, finish=finish)
                    if isinstance(parsed, protocol.Finish):
                        # finish while the instance can still be played: refused, like any action the protocol lacks
                        self.stats["finish_refusals"] += 1
                        raise protocol.ParseError("finish is refused until the instance is over")
                    return parsed
                except protocol.ParseError as error:
                    self.stats["parse_failures"] += 1
                    self.note(f"step {obs.step}: unusable reply ({error}); tail: {reply[-300:]!r}")
                    if attempt == self.max_reprompts:
                        self.note(f"step {obs.step}: giving up after {attempt + 1} unusable replies")
                        return GiveUp()
                    self.stats["reprompts"] += 1
                    hint = (protocol.REPROMPT_FINISH if finish else protocol.REPROMPT).format(reason=error)

                    # A system restarted mid-reprompt has lost the instance: resend it.
                    def follow_up(fresh: bool, hint: str = hint) -> str:
                        return observation(True) + "\n\n" + hint if fresh else hint

                    reply = self._ask_resilient(follow_up, reprompt=True, step=obs.step, task=obs.task_id)
        except SystemUnavailable as error:
            self.stats["forced_give_ups"] += 1
            self.note(f"step {obs.step}: system unavailable after {self.max_restarts} restart(s) "
                      f"({error}); this action is a give-up")
        return GiveUp()

    def observe(self, obs: LearnerObservation, action: Action, feedback: Feedback) -> None:
        ended = instance_ended(feedback)  # the rule by which TrackedLearner waits for the instance's calls
        self.transcript.add_text(protocol.render_feedback(action, feedback))
        if ended:
            self._instances_done += 1
            self._close_instance(obs)
            if self._refine_due():
                self.stats["refines"] += 1
                self.refine()
            if not self.stateless:
                try:
                    self.after_instance()
                except SystemUnavailable as error:
                    self.note(f"step {obs.step}: after-instance step failed ({error})")
            if self.instance_gap_s > 0:
                time.sleep(self.instance_gap_s)
            if self.session_per_instance:
                self._session_pending = True

    def programs_dir(self) -> str:
        """Row HPL's program library: a folder of the harness's workspace (every harness adapter has one, kept between
        instances except in a stateless row), named by its absolute path so that no harness has to guess its cwd."""
        workspace = getattr(self, "workspace", None)
        return str(Path(workspace) / "programs") if workspace else "programs"

    def _refine_due(self) -> bool:
        """Whether the instance that has just ended is followed by the system's refinement (every ``refine_every``
        instances; never for a stateless row, whose state could not survive it)."""
        return bool(self.refine_every) and self._instances_done % self.refine_every == 0 and not self.stateless

    def _close_instance(self, obs: LearnerObservation) -> None:
        """Deliver the verdict on the instance's final submission to the system, in the same
        conversation, before any storage step runs and before the conversation is closed. Without
        this the verdict sat in the transcript until the next instance's first message, in a new
        session about another task, so every harness's storage step reviewed a trajectory whose
        outcome it did not know (23 Sep). One turn per instance; the reply is not read. With
        ``storage_turn`` the turn is the agent's own to save what will help later
        (:func:`protocol.render_storage`), logged as kind ``storage``; else a short closing."""
        pending = self.transcript.take()
        if not self._started:
            return
        if self.end == "finish":
            self._finish_instance(obs, pending)
            return
        storage = self.storage_turn and not self.stateless
        try:
            if storage:
                message = protocol.render_storage(pending, conversation_ends=self.session_per_instance,
                                                  note=self.storage_note)
            else:
                message = protocol.render_closing(pending)
            started = time.monotonic()
            self.last_turn_outcome = {}
            reply = self.ask(message)   # the reply is not read
            self.log_turn("storage" if storage else "closing", message, reply,
                          seconds=round(time.monotonic() - started, 2), step=obs.step, task=obs.task_id)
            self.stats["storage_turns" if storage else "closings"] += 1
        except SystemUnavailable as error:
            # a system that cannot take the closing is restarted for the next instance; the
            # verdict is lost to it, as it was to every system before this message existed
            self.note(f"step {obs.step}: closing message failed ({error}); restarting the system")
            self.close()
            self.stats["restarts"] += 1

    def _finish_instance(self, obs: LearnerObservation, pending: list[str]) -> None:
        """The end of an instance under the finish protocol: the verdict alone, in the same conversation, and the
        harness does what it wants (store, or nothing) until it replies with finish. Any other action is refused with
        the fact that the instance is over (a submission cannot change the scored record); after ``max_reprompts``
        such replies, or when the harness spends its turn's budget, the instance ends here as if it had finished."""
        message = "\n".join(pending) if pending else "The instance is over."
        if self.ask_to_store and not self.stateless:
            message += "\n\n" + protocol.ASK_TO_STORE.format(note=f" {self.storage_note}" if self.storage_note else "")
            self.stats["store_asks"] += 1
        if self.program_library:
            message += "\n\n" + protocol.PROGRAM_LIBRARY_ASK.format(path=f"{self.programs_dir()}/{obs.task_id}.py")
            self.stats["library_asks"] += 1
        try:
            for attempt in range(self.max_reprompts + 1):
                started = time.monotonic()
                self.last_turn_outcome = {}
                reply = self.ask(message, reprompt=attempt > 0)
                self.log_turn("verdict" if attempt == 0 else "finish_reprompt", message, reply,
                              seconds=round(time.monotonic() - started, 2), step=obs.step, task=obs.task_id)
                if self.last_turn_cut:
                    # the harness used its whole turn after the verdict: the instance ends here, with no notice
                    self.stats["finish_budget_hits"] += 1
                    if hasattr(self, "_budget_pending"):
                        self._budget_pending = False
                    return
                try:
                    parsed = protocol.parse_action(reply, finish=True)
                    if isinstance(parsed, protocol.Finish):
                        self.stats["finishes"] += 1
                        return
                    name = {"Submit": "a submission", "RequestDemos": "a demo request"}[type(parsed).__name__]
                    reason = f"{name} is refused"
                except protocol.ParseError as error:
                    reason = str(error)
                if attempt == self.max_reprompts:
                    self.stats["finish_forced"] += 1
                    self.note(f"step {obs.step}: no finish after {attempt + 1} replies; the instance ends here")
                    return
                self.stats["finish_reprompts"] += 1
                message = protocol.OVER_REPROMPT.format(reason=reason)
        except SystemUnavailable as error:
            # a system that cannot take the verdict is restarted for the next instance
            self.note(f"step {obs.step}: verdict turn failed ({error}); restarting the system")
            self.close()
            self.stats["restarts"] += 1

    def resources(self) -> dict[str, float]:
        out = {f"system_{k}": float(v) for k, v in self.stats.items()}
        out.update(self.system_usage())
        return out
