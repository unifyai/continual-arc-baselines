"""The three prompting floors of row F.

* :class:`StatelessLearner` requests demonstrations on every instance, solves
  from them and the input, and on a wrong answer re-prompts with the feedback
  until the cap. It carries nothing between instances: ARC with a per-instance
  charge for the demonstrations, and the floor.
* :class:`DemoCacheLearner` requests demonstrations once per task id, keeps
  them, and replays them to the same solver on every later instance. Near the
  ceiling of what the model can solve on the score; flat on everything else.
* :class:`ProgramLibraryLearner` asks the model on first exposure for a Python
  program mapping input to output, keeps it if it reproduces every
  demonstration, and runs it on later instances first; on a wrong answer it asks
  for a repair given the failure and the demonstrations.

All three see the task only under the run's name (the runner's view), render
grids the way the harness protocol does, and count nothing themselves: model calls go through the
proxy and are journaled there. Programs run in a subprocess under the same
sandbox the harnesses get, with no network and no access to the benchmark.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from continual_arc.env import (
    Action,
    DemosFeedback,
    Feedback,
    LearnerObservation,
    RequestDemos,
    Submit,
    SubmitFeedback,
)
from continual_arc.grids import InvalidGrid, normalize_grid
from continual_arc.learner import BaseLearner

from continual_arc_baselines.learners.chat import ChatClient
from continual_arc_baselines.systems.protocol import render_grid, render_pairs
from continual_arc_baselines.systems.sandbox import (
    SandboxSpec,
    darwin_user_dirs,
    python_runtime_root,
    resolve_backend,
    wrap_command,
)

log = logging.getLogger(__name__)

SYSTEM = """\
You solve ARC-style grid puzzles. A grid is a list of rows; cells are integers 0-9; each
dimension is 1-30. You are shown demonstration input/output pairs that share one hidden
transformation rule, then a test input. Work out the rule and produce the output grid for
the test input. Your reply must end with exactly one JSON object on its own line:
{"grid": [[0, 1, 2], [3, 4, 5]]}
Only the last JSON object in your reply is read; it must be the complete output grid."""

PROGRAM_SYSTEM = """\
You solve ARC-style grid puzzles by writing programs. A grid is a list of rows of integers
0-9. You are shown demonstration input/output pairs that share one hidden transformation
rule. Write a Python function `transform(grid)` that returns the output grid for any input
of this task. Use only the standard library. Reply with exactly one fenced ```python block
containing the complete function and nothing else that needs to run."""

STYLES = {
    # the model may reason at length in its reply (or in its reasoning channel)
    "reasoned": "",
    # the reply is the grid: cheap and fast, at the price of visible working
    "brief": "\n\nReply with the JSON grid line only, no explanation.",
}

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_PY_FENCE = re.compile(r"```python\s*(.*?)```", re.DOTALL)


def extract_program(text: str) -> str | None:
    """The program from a reply: the ``program`` string of the last JSON object, else the
    last fenced Python block, else the whole text when it defines ``transform``."""
    from continual_arc_baselines.systems.protocol import _json_objects_from_end

    for candidate in [m.group(1) for m in reversed(list(_FENCE.finditer(text)))] + _json_objects_from_end(text):
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("program"), str) and "def transform" in obj["program"]:
            return obj["program"]
    m = _PY_FENCE.search(text)
    if m:
        return m.group(1)
    return text if "def transform" in text else None


def extract_grid(text: str):
    """The last JSON object with a grid under a known key, else the last bare
    list of lists; None if nothing parses as a grid."""
    from continual_arc_baselines.systems.protocol import GRID_KEYS, _json_objects_from_end

    candidates = [m.group(1) for m in reversed(list(_FENCE.finditer(text)))]
    candidates += _json_objects_from_end(text)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            for key in ("grid", *GRID_KEYS):
                if key in obj:
                    try:
                        return normalize_grid(obj[key])
                    except InvalidGrid:
                        break
    # a bare list of lists, last one in the text
    for m in reversed(list(re.finditer(r"\[\s*\[[\d,\s\[\]]*\]\s*\]", text))):
        try:
            return normalize_grid(json.loads(m.group(0)))
        except (ValueError, InvalidGrid):
            continue
    return None


def _prompt(label: str, pairs: list, test_input, feedback: list[str]) -> str:
    parts = [f"Task {label}."]  # the task id only; no exposure count (20 Sep)
    if pairs:
        parts.append(f"{len(pairs)} demonstration pair(s) follow, then the test input.\n")
        parts.append(render_pairs(pairs))
    else:
        parts.append("No demonstrations are shown. Use what you know about this task.")
    parts.append(f"\nTest input ({len(test_input)}x{len(test_input[0])}):\n{render_grid(test_input)}")
    if feedback:
        parts.append("\nFeedback on your previous answers for this test input:")
        parts.extend(feedback)
    parts.append("\nReply with the output grid for the test input as the final JSON line.")
    return "\n".join(parts)


def run_transform(code: str, grid, *, sandbox: str, timeout_s: float, workdir: Path) -> tuple[Any, str | None]:
    """Run a model-written ``transform(grid)`` on one grid in a sandboxed subprocess, with no network and no access
    to the benchmark (the harnesses' sandbox). Returns (grid, None), or (None, what went wrong). Shared by the
    program-library learners, prompting (23 Sep) and bare (28 Sep)."""
    script = workdir / "program.py"
    script.write_text(code + textwrap.dedent('''

        if __name__ == "__main__":
            import json, sys
            _g = json.loads(sys.stdin.read())
            print(json.dumps(transform(_g)))
    '''))
    backend = resolve_backend(sandbox)
    python = Path(sys.executable)
    reads = [python.parent.parent]
    runtime = python_runtime_root(python)
    if runtime:
        reads.append(runtime)
    spec = SandboxSpec(read_paths=reads + [workdir], write_paths=[workdir, *darwin_user_dirs()],
                       tmp_dir=workdir, home=workdir)
    argv = wrap_command(backend, [str(python), "-I", "-S", str(script)], spec, workdir / "sandbox.sb")
    try:
        out = subprocess.run(argv, input=json.dumps(grid), capture_output=True, text=True,
                             timeout=timeout_s, check=False, cwd=workdir,
                             env={"HOME": str(workdir), "PATH": "/usr/bin:/bin"})
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if out.returncode != 0:
        return None, (out.stderr or "").strip()[-400:] or f"exit {out.returncode}"
    try:
        return normalize_grid(json.loads(out.stdout.strip().splitlines()[-1])), None
    except (ValueError, IndexError, InvalidGrid) as error:
        return None, f"bad output: {error}"


class _Floor(BaseLearner):
    """Shared plumbing: the client and per-instance state."""

    name = "floor"

    def __init__(self, model: str = "IFM/K2-Horizon-7B",
                 max_calls_per_instance: int = 6, chat: ChatClient | None = None,
                 defaults: dict[str, Any] | None = None, timeout_s: float = 600.0,
                 style: str = "reasoned", demos_wanted: int = 8):
        if style not in STYLES:
            raise ValueError(f"style must be one of {sorted(STYLES)}, got {style!r}")
        self.style = style
        self.model = model
        self.chat = chat or ChatClient(model, defaults=defaults, timeout_s=timeout_s)
        self.max_calls_per_instance = max_calls_per_instance
        # Demonstrations arrive one pair per request since 26 Sep; a floor asks for this many (all eight, Dan)
        # before its first answer, and stops early only if a request is refused or returns nothing.
        self.demos_wanted = demos_wanted
        self._task: str | None = None
        self._feedback: list[str] = []
        self._calls_this_instance = 0
        self.stats = {"parse_failures": 0, "chat_errors": 0, "truncations": 0}

    def _begin(self, obs: LearnerObservation) -> None:
        if obs.attempts_used == 0 and obs.demo_requests_used == 0 and (self._task != (obs.task_id, obs.step)):
            self._task = (obs.task_id, obs.step)
            self._feedback = []
            self._calls_this_instance = 0

    def _solve(self, obs: LearnerObservation, pairs: list) -> Action:
        if self._calls_this_instance >= self.max_calls_per_instance:
            # Out of model calls: no learner may abandon an instance (one protocol for every
            # row, 20 Sep), so submit an invalid grid, charged like a wrong answer, until the
            # attempt cap ends the instance.
            return Submit([[10]])
        self._calls_this_instance += 1
        try:
            text = self.chat.complete([
                {"role": "system", "content": SYSTEM + STYLES[self.style]},
                {"role": "user", "content": _prompt(obs.task_id, pairs, obs.input, self._feedback)},
            ])
        except Exception as error:  # a provider failure costs the action, not the run
            self.stats["chat_errors"] += 1
            log.warning("chat failed: %s", error)
            return Submit([[10]])  # charged like a wrong answer; the instance is not abandoned
        grid = extract_grid(text)
        if grid is None:
            self.stats["parse_failures"] += 1
            if getattr(self.chat, "last_finish_reason", None) == "length":
                self.stats["truncations"] += 1
                log.warning("reply cut at max_tokens before a grid appeared")
            self._feedback.append("Your previous reply contained no readable grid.")
            return Submit([[10]])  # invalid on purpose: charged like a wrong answer, keeps accounting honest
        return Submit(grid)

    def observe(self, obs: LearnerObservation, action: Action, feedback: Feedback) -> None:
        if isinstance(feedback, SubmitFeedback) and not feedback.correct and isinstance(action, Submit):
            self._feedback.append(f"Attempt {feedback.attempts_used}: INCORRECT. You submitted:\n"
                                  + (render_grid(action.grid) if action.grid != [[10]] else "(unreadable)"))

    def resources(self) -> dict[str, float]:
        return {"floor_calls": float(self.chat.calls), "floor_failures": float(self.chat.failures),
                "floor_parse_failures": float(self.stats["parse_failures"])}

    def describe(self) -> dict[str, Any]:
        return {"system": self.name, "model": self.model,
                "max_calls_per_instance": self.max_calls_per_instance, "style": self.style,
                "demos_wanted": self.demos_wanted}

    def close(self) -> None:
        self.chat.close()


class StatelessLearner(_Floor):
    name = "stateless"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._pairs: list = []
        self._instance: tuple | None = None
        self._refused = False

    def reset(self) -> None:
        self._pairs = []

    def act(self, obs: LearnerObservation) -> Action:
        self._begin(obs)
        if self._instance != (obs.task_id, obs.step):  # a new instance: nothing carries over
            self._instance, self._pairs, self._refused = (obs.task_id, obs.step), [], False
        if len(self._pairs) < self.demos_wanted and obs.demo_requests_used < obs.demo_cap and not self._refused:
            return RequestDemos()
        return self._solve(obs, self._pairs)

    def observe(self, obs, action, feedback) -> None:
        if isinstance(feedback, DemosFeedback):
            if feedback.refused or not feedback.pairs:
                self._refused = True
            else:
                self._pairs.extend(feedback.pairs)
        super().observe(obs, action, feedback)


class DemoCacheLearner(_Floor):
    name = "demo-cache"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.cache: dict[str, list] = {}
        self._complete: set[str] = set()   # tasks whose demonstrations are all cached: never requested again
        self._instance: tuple | None = None
        self._refused = False

    def reset(self) -> None:
        self.cache = {}
        self._complete = set()

    def act(self, obs: LearnerObservation) -> Action:
        self._begin(obs)
        if self._instance != (obs.task_id, obs.step):
            self._instance, self._refused = (obs.task_id, obs.step), False
        tid = obs.task_id
        if tid not in self._complete:
            if (not self._refused and obs.demo_requests_used < obs.demo_cap
                    and len(self.cache.get(tid, [])) < self.demos_wanted):
                return RequestDemos()
            self._complete.add(tid)   # the first meeting has what it will get: never asked for again
        return self._solve(obs, self.cache.get(tid, []))

    def observe(self, obs, action, feedback) -> None:
        if isinstance(feedback, DemosFeedback):
            if feedback.refused or not feedback.pairs:
                self._refused = True
            else:
                self.cache.setdefault(obs.task_id, []).extend(feedback.pairs)
        super().observe(obs, action, feedback)

    def resources(self) -> dict[str, float]:
        out = super().resources()
        out["state_bytes"] = float(len(json.dumps({k: [p.to_json() for p in v] for k, v in self.cache.items()})))
        return out


class ProgramLibraryLearner(_Floor):
    name = "program-library"

    def __init__(self, sandbox: str = "auto", program_timeout_s: float = 10.0, max_repairs: int = 2,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.sandbox = sandbox
        self.program_timeout_s = program_timeout_s
        self.max_repairs = max_repairs
        self.library: dict[str, str] = {}
        self.demos: dict[str, list] = {}
        self._complete: set[str] = set()
        self._refused = False
        self._repairs_this_instance = 0
        self._tried_program = False
        self._workdir = Path(tempfile.mkdtemp(prefix="arc-programs-"))
        self.stats.update({"programs_kept": 0, "programs_rejected": 0, "repairs": 0, "program_errors": 0})

    def reset(self) -> None:
        self.library = {}
        self.demos = {}
        self._complete = set()

    # -- running model-written code ------------------------------------------

    def run_program(self, code: str, grid) -> tuple[Any, str | None]:
        """Run ``transform(grid)`` in a sandboxed subprocess. Returns (grid, error)."""
        return run_transform(code, grid, sandbox=self.sandbox, timeout_s=self.program_timeout_s, workdir=self._workdir)

    def _verify(self, code: str, pairs: list) -> str | None:
        for k, pair in enumerate(pairs):
            got, error = self.run_program(code, pair.input)
            if error:
                return f"demonstration {k + 1}: program failed ({error})"
            if got != pair.output:
                return f"demonstration {k + 1}: program output\n{render_grid(got)}\nexpected\n{render_grid(pair.output)}"
        return None

    def _induce(self, task_id: str, pairs: list, previous: str | None, why: str | None) -> str | None:
        user = [f"Task {task_id}.", render_pairs(pairs)]
        if previous:
            user.append("\nYour previous program:\n```python\n" + previous + "\n```")
            user.append("It is wrong: " + (why or "unknown") + "\nWrite a corrected program.")
        try:
            # a program is Python in a fence, not JSON: the grid floors' JSON output mode is
            # switched off for this call (small models cannot write a program as a JSON string)
            text = self.chat.complete([{"role": "system", "content": PROGRAM_SYSTEM},
                                       {"role": "user", "content": "\n".join(user)}], response_format=None)
        except Exception as error:
            self.stats["chat_errors"] += 1
            log.warning("chat failed: %s", error)
            return None
        return extract_program(text)

    # -- the learner ----------------------------------------------------------

    def act(self, obs: LearnerObservation) -> Action:
        fresh = obs.attempts_used == 0 and obs.demo_requests_used == 0
        if fresh and self._task != (obs.task_id, obs.step):
            self._repairs_this_instance = 0
            self._tried_program = False
            self._refused = False
        self._begin(obs)
        tid = obs.task_id
        if tid not in self._complete:
            if (not self._refused and obs.demo_requests_used < obs.demo_cap
                    and len(self.demos.get(tid, [])) < self.demos_wanted):
                return RequestDemos()
            self._complete.add(tid)
        pairs = self.demos.get(tid, [])
        # a program is checked against the unmarked pairs only, which share one frame: a padded pair (a transformed
        # copy, marked with a note) can show the rule rotated or recoloured, which a correct program need not match
        framed = [p for p in pairs if not getattr(p, "note", None)] or pairs
        if tid in self.library and not self._tried_program:
            self._tried_program = True
            got, error = self.run_program(self.library[tid], obs.input)
            if error is None:
                return Submit(got)
            self.stats["program_errors"] += 1
        # No usable program: induce or repair, up to the budget, then fall back to a direct answer.
        if self._calls_this_instance < self.max_calls_per_instance and self._repairs_this_instance <= self.max_repairs:
            self._calls_this_instance += 1
            previous = self.library.get(tid)
            why = self._feedback[-1] if previous and self._feedback else None
            code = self._induce(tid, pairs, previous, why)
            if previous:
                self.stats["repairs"] += 1
            self._repairs_this_instance += 1
            if code:
                problem = self._verify(code, framed) if framed else None
                if problem is None:
                    self.library[tid] = code
                    self.stats["programs_kept"] += 1
                    got, error = self.run_program(code, obs.input)
                    if error is None:
                        return Submit(got)
                else:
                    self.stats["programs_rejected"] += 1
                    self._feedback.append(problem)
        return self._solve(obs, pairs)

    def observe(self, obs, action, feedback) -> None:
        if isinstance(feedback, DemosFeedback):
            if feedback.refused or not feedback.pairs:
                self._refused = True
            else:
                self.demos.setdefault(obs.task_id, []).extend(feedback.pairs)
        if isinstance(feedback, SubmitFeedback) and not feedback.correct and obs.task_id in self.library:
            # the stored program was wrong on this input: the next call repairs it
            self._feedback.append(f"the stored program produced a wrong answer on a test input:\n{render_grid(obs.input)}")
        super().observe(obs, action, feedback)

    def resources(self) -> dict[str, float]:
        out = super().resources()
        out["state_bytes"] = float(sum(len(c) for c in self.library.values()))
        out["programs_kept"] = float(self.stats["programs_kept"])
        out["programs_rejected"] = float(self.stats["programs_rejected"])
        out["repairs"] = float(self.stats["repairs"])
        return out
