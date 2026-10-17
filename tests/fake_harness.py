"""A stand-in for a harness adapter, for the tracking tests: one model call per action through its proxy, and a
memory of its own in its state directory."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import httpx
from continual_arc.env import DemosFeedback, RequestDemos, Submit

from continual_arc_baselines.proxy import PLACEHOLDER_KEY
from continual_arc_baselines.systems.protocol import TaskAliases
from continual_arc_baselines.tracking import Tracker


class FakeHarness:
    """Stands in for a harness adapter: one model call per action through its proxy (asking for a
    reasoning effort the server would refuse), and a memory of its own, a file in its state
    directory, from which it solves a task it has met before; otherwise it asks for demonstrations."""

    inside = 0
    max_inside = 0
    lock = threading.Lock()

    def __init__(self, user: int) -> None:
        self.user = user
        self.aliases = TaskAliases()
        self.environ: dict[str, str] = {}
        self.demos: list = []
        self.calls = 0

    def attach(self, workdir: Path, environ: dict[str, str], tracker=None) -> None:
        self.workdir, self.environ = Path(workdir), dict(environ)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)

    @property
    def memory_file(self) -> Path:
        return self.workdir / "state" / "memory.json"

    def memory(self) -> dict:
        return json.loads(self.memory_file.read_text()) if self.memory_file.exists() else {}

    def task_label(self, task_id: str) -> str:
        return self.aliases.alias(task_id)

    def ask(self, content: str) -> None:
        httpx.post(self.environ["ARC_LLM_BASE_URL"] + "/chat/completions", timeout=10,
                   headers={"Authorization": f"Bearer {PLACEHOLDER_KEY}"},
                   json={"model": "base", "reasoning_effort": "xhigh", "messages": [{"role": "user", "content": content}]})
        self.calls += 1

    def act(self, obs):
        with FakeHarness.lock:
            FakeHarness.inside += 1
            FakeHarness.max_inside = max(FakeHarness.max_inside, FakeHarness.inside)
        try:
            time.sleep(0.02)
            self.ask(f"{self.task_label(obs.task_id)} step {obs.step}")
            shift = self.memory().get(self.task_label(obs.task_id))
            if shift is None and self.demos:
                d = self.demos[0]
                shift = (d.output[0][0] - d.input[0][0]) % 10
            if shift is None:
                return RequestDemos()
            return Submit([[(c + shift) % 10 for c in row] for row in obs.input])
        finally:
            with FakeHarness.lock:
                FakeHarness.inside -= 1

    def observe(self, obs, action, feedback) -> None:
        if isinstance(feedback, DemosFeedback) and feedback.pairs:
            self.demos = list(feedback.pairs)
        if getattr(feedback, "correct", False):
            d = self.demos[0] if self.demos else None
            if d is not None:  # the storage step: write the rule down under the task's name
                mem = self.memory()
                mem[self.task_label(obs.task_id)] = (d.output[0][0] - d.input[0][0]) % 10
                self.memory_file.write_text(json.dumps(mem))
            self.demos = []

    def reset(self) -> None:
        self.memory_file.unlink(missing_ok=True)

    def resources(self) -> dict[str, float]:
        return {"harness_calls": float(self.calls)}

    def close(self) -> None:
        pass


class StoringHarness(FakeHarness):
    """A FakeHarness whose storage step asks the model too: the last call of every instance it solves."""

    def observe(self, obs, action, feedback) -> None:
        super().observe(obs, action, feedback)
        if getattr(feedback, "correct", False):
            self.ask(f"{self.task_label(obs.task_id)} storage")


def late_reports(monkeypatch, delay: float, which=lambda call: True) -> None:
    """The proxy reports the end of a call (``which`` of them) ``delay`` seconds after the harness has the reply."""
    emit = Tracker.emit

    def late(self, call):
        if call.status != "started" and which(call):
            time.sleep(delay)
        emit(self, call)

    monkeypatch.setattr(Tracker, "emit", late)
