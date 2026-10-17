"""Adapters that put a third-party agent *system* behind the learner interface.

The benchmark scores systems, not models. A system owns its agent loop, its
tools, its memory and its model calls; the adapter's job is only to hand it
each observation, read back one action, and wipe its state when the schedule
says so. Model usage is measured at the HTTP boundary by
:mod:`continual_arc_baselines.proxy`, never by the adapter.
"""

from continual_arc_baselines.systems.base import SystemLearner, SystemUnavailable

__all__ = ["SystemLearner", "SystemUnavailable"]
