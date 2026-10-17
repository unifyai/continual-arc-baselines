"""hermes-agent (Nous Research) as a learner, over its native ACP server.

``hermes-acp`` is one long-running process with one session. Everything the
system keeps between instances lives under a private ``HERMES_HOME``: its
memories (``MEMORY.md``, ``USER.md``), skills, sessions and caches. That
directory is the continual-learning method under test.

Two rows of the paper map onto one switch. hermes-agent's own self-improvement
is a post-turn background review that rewrites memory and skills (the same
machinery its ``/refine`` command triggers by hand). ``self_improve=False``
turns it off (row H0: the harness keeps what it writes during turns and nothing
more); ``self_improve=True`` leaves it on (row H1: the harness evolves itself).
The ACP server does not dispatch ``/refine`` (unknown slash commands go to the
model as text), so ``refine_every`` has no effect here and is recorded as such.

The system starts as a blank slate: the ``.no-bundled-skills`` marker stops hermes
seeding its bundled skill library into ``HERMES_HOME``, so the skills directory
is empty until the system writes to it, and no messaging channel is configured.

Model calls go through the proxy: ``config.yaml`` names an ``openai`` provider
whose endpoint is the proxy and whose key is the placeholder. Install: the
checkout's virtual environment provides ``hermes-acp`` (``uv pip install -e
".[acp]"`` in the hermes-agent repository); pass ``command`` if it is elsewhere.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from continual_arc_baselines.systems.acp import KEY_VARIABLE, ACPSystemLearner
from continual_arc_baselines.systems.base import SystemUnavailable
from continual_arc_baselines.systems.sandbox import python_runtime_root

CHECKOUT = Path.home() / "hermes-agent"


class HermesAgentLearner(ACPSystemLearner):
    name = "hermes-agent"

    def __init__(self, model: str = "IFM/K2-Horizon-7B", **kwargs: Any):
        super().__init__(model=model, **kwargs)
        if self.refine_every:
            self.refine_every = 0  # not applicable; see the module docstring

    def default_command(self) -> list[str]:
        local = CHECKOUT / ".venv" / "bin" / "hermes-acp"
        if local.exists():
            return [str(local)]
        found = shutil.which("hermes-acp")
        return [found or "hermes-acp"]

    @property
    def hermes_home(self) -> Path:
        return self.state_dir / "hermes"

    def transcript_paths(self) -> list[Path]:
        return [self.hermes_home]   # its sessions database (state.db) and logs, about 1.6 MB an instance in H0

    def argv(self) -> list[str]:
        return list(self.command)

    DEAD_REPLY = "Provider has been unresponsive"

    def ask(self, message: str, *, reprompt: bool = False) -> str:
        reply = super().ask(message, reprompt=reprompt)
        if self.DEAD_REPLY in reply:
            # hermes latches this after five calls with no response (a credit lapse held by the
            # proxy did it on 22 Sep) and refuses every call thereafter; only a new session
            # clears it. The system is down, not answering: restart it, state intact.
            raise SystemUnavailable("hermes latched its provider as unresponsive")
        return reply

    def prepare(self) -> None:
        self.hermes_home.mkdir(parents=True, exist_ok=True)
        # Blank slate: hermes seeds its bundled skill library into a fresh HERMES_HOME
        # unless this marker exists. The system starts with no skills and no memories;
        # whatever it has later, it wrote itself. Messaging channels are never
        # configured (the ACP server is the only surface).
        (self.hermes_home / ".no-bundled-skills").touch()
        config = {
            "model": {"provider": "openai", "default": self.model, "max_tokens": self.max_tokens},
            "providers": {"openai": {"api": self.base_url(), "api_key": self.api_key(),
                                     "discover_models": False, "context_length": self.context_window}},
            "approvals": {"mode": "off"},
            "auxiliary": {
                "title_generation": {"enabled": False},
            },
        }
        # JSON is valid YAML, and hermes reads config.yaml.
        (self.hermes_home / "config.yaml").write_text(json.dumps(config, indent=1))

    def extra_env(self) -> dict[str, str]:
        return {
            "HERMES_HOME": str(self.hermes_home),
            "HERMES_INFERENCE_MODEL": self.model,
            "HERMES_YOLO_MODE": "1",
            "OPENAI_BASE_URL": self.base_url(),
            "OPENAI_API_KEY": self.environ.get(KEY_VARIABLE) or self.api_key(),
        }

    def extra_read_paths(self) -> list[Path]:
        reads: list[Path] = []
        exe = Path(self.command[0]).expanduser()
        venv = exe.resolve().parent.parent if exe.exists() else None
        if venv is not None:
            reads.append(venv)
            checkout = venv.parent
            if (checkout / "pyproject.toml").exists():
                reads.append(checkout)  # an editable install imports from the checkout
            runtime = python_runtime_root(venv / "bin" / "python")
            if runtime:
                reads.append(runtime)
        return reads

    def version(self) -> str:
        if self._version is None:
            try:
                out = subprocess.run([*self.command, "--version"], capture_output=True, text=True,
                                     timeout=60, check=False, env={**os.environ, "HERMES_HOME": str(self.hermes_home)})
                text = (out.stdout or out.stderr).strip()
                self._version = text.splitlines()[0] if text else "unknown"
            except (OSError, subprocess.TimeoutExpired):
                self._version = "unknown"
        return self._version

    def refine(self) -> None:
        self.note("refine requested but hermes-agent evolves through its background review "
                  "(self_improve); nothing to trigger")

    def describe(self) -> dict[str, Any]:
        return super().describe()
