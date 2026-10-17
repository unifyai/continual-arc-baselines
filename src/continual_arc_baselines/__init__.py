"""Baselines and experiment harness for Continual-ARC.

``continual-arc`` is the benchmark: environment, schedules, metrics, with no model, no network and no tracking. This
package holds what the paper evaluates on it: the bare-model floors and the program library
(:mod:`continual_arc_baselines.systems.bare`), the adapters that drive agent harnesses through the benchmark's protocol
(:mod:`continual_arc_baselines.systems`), the proxy through which every model call passes
(:mod:`continual_arc_baselines.proxy`), and :mod:`continual_arc_baselines.cli`, which runs a learner over a schedule
and writes the run directory with its evidence (:mod:`continual_arc_baselines.evidence`).
"""

__version__ = "0.1.0"
