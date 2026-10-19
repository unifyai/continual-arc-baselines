"""Baselines and experiment harness for Continual-ARC.

``continual-arc`` is the benchmark: environment, schedules, metrics, with no model, no network and no tracking. This
package holds what the paper evaluates on it: the bare-model floors and the program library
(:mod:`continual_arc_baselines.systems.bare`), the adapters that drive agent harnesses through the benchmark's protocol
(:mod:`continual_arc_baselines.systems`), the proxy through which every model call passes
(:mod:`continual_arc_baselines.proxy`), and :mod:`continual_arc_baselines.cli`, which runs a learner over a schedule
and writes the run directory with its evidence (:mod:`continual_arc_baselines.evidence`).
"""

import inspect

from continual_arc import runner as _runner

__version__ = "0.1.0"

# The adapters pass a system each task's name as the benchmark's runner gives it, so under a runner that still shows a
# learner the whole slot, the real task id (a lookup key into the public solutions) would reach the system as it is.
# A lockfile made before the restricted learner view pins such a benchmark, so loading against one is refused rather
# than left to leak the id.
if "view" in inspect.signature(_runner.run).parameters:
    raise ImportError("continual-arc predates the restricted learner view: update it "
                      "(uv lock --upgrade-package continual-arc && uv sync)")
