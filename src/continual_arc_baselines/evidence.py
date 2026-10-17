"""Bridge a continual-arc run to a workbench attempt directory.

The continual-arc runner already writes a complete run directory (schedule,
config, per-instance records, summary). This module mirrors the parts a person
wants to compare across runs into workbench events, so ``workbench sync`` can put
them in MLflow, and registers the run directory's files as artifacts so the
evidence travels with the attempt.

Metric names (``step`` is the schedule step, zero-based):

* ``instance/cost``, ``instance/solved``, ``instance/wrong_attempts``,
  ``instance/demo_requests``, ``instance/gave_up``, ``instance/exposure_index``
* ``instance/resource/<key>`` for every counter the learner reports
* ``running/mean_cost`` and ``running/score`` over the stream so far
* ``summary/<phase>/<stat>`` once, at the end, from the run summary
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from continual_workbench.context import RunContext, create_context

ADHOC_ROOT = ".workbench"

_INSTANCE_FIELDS = ("cost", "solved", "wrong_attempts", "demo_requests", "gave_up", "exposure_index")
_SUMMARY_PHASES = ("overall", "stream", "epilogue")
_SUMMARY_STATS = (
    "n", "mean_cost", "score", "solve_rate", "mean_wrong_attempts",
    "mean_demo_requests", "give_up_rate", "invalid_rate",
)
_RUN_FILES = ("schedule.json", "config.json", "meta.json", "records.jsonl", "summary.json", "events.jsonl")


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not value or not value[0].isalpha():
        value = "run-" + value
    return value


def adhoc_context(name: str, root: str | os.PathLike[str] = ADHOC_ROOT) -> RunContext:
    """Create an attempt directory outside any ``workbench run``.

    Every run leaves evidence, even a quick one from the terminal. The run id is
    ``adhoc-<name>-<unix seconds>`` so it never collides with workbench runs.
    """
    run_id = _slug(f"adhoc-{name}-{int(time.time())}")
    manifest = {"experiment_id": "adhoc", "project_key": "continual-arc-baselines",
                "recovery": "restart-only", "resume_performed": False}
    return create_context(root, run_id, manifest)


def dependency_revisions(names: tuple[str, ...] = ("continual-arc",)) -> dict[str, str]:
    """The exact source each dependency was installed from, for the run record.

    The benchmark is not pinned, so this is what makes a run reproducible: a git
    install yields ``git+<url>@<commit>``; an editable checkout yields
    ``editable:<path>`` (honestly unpinned); anything else yields the version.
    """
    import importlib.metadata as md
    import json

    out: dict[str, str] = {}
    for name in names:
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            out[name] = "not-installed"
            continue
        raw = dist.read_text("direct_url.json")
        info = json.loads(raw) if raw else {}
        url, vcs, dir_info = info.get("url", ""), info.get("vcs_info") or {}, info.get("dir_info") or {}
        if vcs.get("commit_id"):
            out[name] = f"git+{url}@{vcs['commit_id']}"
        elif dir_info.get("editable"):
            out[name] = f"editable:{url.removeprefix('file://')}"
        else:
            out[name] = f"{dist.version}"
    return out


class Evidence:
    """Emit continual-arc progress and results into a workbench :class:`RunContext`."""

    def __init__(self, context: RunContext, max_cost: float | None = None):
        self.context = context
        self.max_cost = max_cost
        self._n = 0
        self._total_cost = 0.0

    @classmethod
    def from_environment(cls, name: str = "run", max_cost: float | None = None) -> Evidence:
        """Use the attempt ``workbench run`` created, or make an ad-hoc one."""
        if os.environ.get("WORKBENCH_RUN_DIR"):
            return cls(RunContext.from_environment(), max_cost=max_cost)
        return cls(adhoc_context(name), max_cost=max_cost)

    @classmethod
    def reopen(cls, attempt_dir: str | os.PathLike[str], max_cost: float | None = None) -> Evidence:
        """Continue writing into an attempt an interrupted run created (``--resume``)."""
        return cls(RunContext(str(attempt_dir)), max_cost=max_cost)

    def prime(self, records) -> None:
        """Seed the running mean with the stream records a resumed run already holds."""
        for record in records:
            if record.phase == "stream":
                self._n += 1
                self._total_cost += float(record.cost)

    @property
    def directory(self) -> Path:
        return self.context.directory

    @property
    def inside_workbench(self) -> bool:
        return bool(os.environ.get("WORKBENCH_RUN_DIR"))

    def parameters(self, values: dict[str, Any]) -> None:
        self.context.parameters(values)

    # -- per instance --------------------------------------------------------

    def progress(self, record, n_steps: int) -> None:
        """A ``progress`` callback for :func:`continual_arc.runner.run`."""
        step = int(record.step)
        for name in _INSTANCE_FIELDS:
            self.context.metric(f"instance/{name}", float(getattr(record, name)), step=step)
        for key, value in (record.resources or {}).items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            self.context.metric(f"instance/resource/{_slug(key)}", value, step=step)
        if record.phase == "stream":
            self._n += 1
            self._total_cost += float(record.cost)
            mean = self._total_cost / self._n
            self.context.metric("running/mean_cost", mean, step=step)
            if self.max_cost:
                self.context.metric("running/score", 100.0 * (1.0 - mean / self.max_cost), step=step)

    # -- at the end ----------------------------------------------------------

    def finish(self, result) -> dict[str, float]:
        """Emit summary metrics and register the run directory's files as artifacts."""
        emitted: dict[str, float] = {}
        summary = result.summary
        for phase in _SUMMARY_PHASES:
            block = summary.get(phase) or {}
            for stat in _SUMMARY_STATS:
                value = block.get(stat)
                if isinstance(value, (int, float)) and value == value:
                    name = f"summary/{phase}/{stat}"
                    self.context.metric(name, float(value))
                    emitted[name] = float(value)
        for key, block in (summary.get("resources") or {}).items():
            total = block.get("total")
            if isinstance(total, (int, float)):
                name = f"summary/resource/{_slug(key)}/total"
                self.context.metric(name, float(total))
                emitted[name] = float(total)
        if result.out_dir is not None:
            for filename in _RUN_FILES:
                path = Path(result.out_dir) / filename
                if path.is_file():
                    self.context.artifact(path)
        self.context.emit("continual_arc_summary", {
            "out_dir": str(result.out_dir) if result.out_dir is not None else None,
            "n_records": len(result.records),
            "overall_score": (summary.get("overall") or {}).get("score"),
        })
        return emitted
