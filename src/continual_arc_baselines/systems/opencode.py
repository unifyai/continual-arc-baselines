"""opencode (SST) as a learner, over ``opencode acp``.

opencode keeps its state under the XDG directories (data, state, cache, config),
all pointed at a private directory here. It has no self-improvement mechanism
(no memory review, no skill capture), so it appears only in the frozen row
without evolution; ``refine_every`` and ``self_improve`` are refused so the
record cannot claim otherwise.

The model is a custom OpenAI-compatible provider in ``opencode.json`` whose
base URL is the proxy. Install: ``brew install opencode`` or the upstream
installer; pass ``command`` if the binary is elsewhere.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from continual_arc_baselines.systems.acp import ACPSystemLearner, free_port

PROVIDER = "arc-proxy"


class OpenCodeLearner(ACPSystemLearner):
    name = "opencode"

    def __init__(self, model: str = "IFM/K2-Horizon-7B", self_improve: bool = False, **kwargs: Any):
        if self_improve or kwargs.get("refine_every"):
            raise ValueError("opencode has no self-improvement mechanism; run it with "
                             "self_improve=False and refine_every=0")
        self._port: int | None = None
        super().__init__(model=model, **kwargs)

    def default_command(self) -> list[str]:
        return [shutil.which("opencode") or "opencode"]

    def listen_ports(self) -> list[int]:
        return [self._port] if self._port else []

    @property
    def xdg(self) -> dict[str, Path]:
        root = self.state_dir / "xdg"
        return {name: root / name for name in ("data", "state", "cache", "config")}

    @property
    def config_path(self) -> Path:
        return self.xdg["config"] / "opencode" / "opencode.json"

    def argv(self) -> list[str]:
        # The ACP server runs opencode's internal HTTP server too; give it a
        # loopback port the sandbox lets it bind.
        return [*self.command, "acp", "--cwd", str(self.workspace), "--hostname", "127.0.0.1",
                "--port", str(self._port)]

    def prepare(self) -> None:
        self._port = free_port()
        for path in self.xdg.values():
            path.mkdir(parents=True, exist_ok=True)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {PROVIDER: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Continual-ARC proxy",
                "options": {"baseURL": self.base_url(), "apiKey": "{env:ARC_LLM_API_KEY}",
                            "timeout": int(self.timeout_s * 1000)},
                "models": {self.model: {
                    "name": self.model, "reasoning": True, "tool_call": True, "temperature": True,
                    "cost": {"input": 0, "output": 0},
                    "limit": {"context": self.context_window, "output": self.max_tokens},
                    "modalities": {"input": ["text"], "output": ["text"]},
                }},
            }},
            "model": f"{PROVIDER}/{self.model}",
            "small_model": f"{PROVIDER}/{self.model}",
            "share": "disabled",
        }
        self.config_path.write_text(json.dumps(config, indent=1))

    def extra_env(self) -> dict[str, str]:
        xdg = self.xdg
        return {
            "XDG_DATA_HOME": str(xdg["data"]),
            "XDG_STATE_HOME": str(xdg["state"]),
            "XDG_CACHE_HOME": str(xdg["cache"]),
            "XDG_CONFIG_HOME": str(xdg["config"]),
            "OPENCODE_CONFIG": str(self.config_path),
            # Project config discovery makes opencode write a .gitignore into its own install
            # directory, read-only in the sandbox, and die at start (smoke, 22 Sep): it stays off.
            # Its persistent carrier is then its private global config directory.
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_SHARE": "1",
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
            # External skills are the ~/.agents and ~/.claude directories, not the workspace; enabling
            # them makes opencode write a .gitignore into its own install directory, which the sandbox
            # keeps read-only, and it dies at start (smoke, 22 Sep). Workspace skills stay discoverable.
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        }

    def version(self) -> str:
        if self._version is None:
            try:
                out = subprocess.run([*self.command, "--version"], capture_output=True, text=True,
                                     timeout=60, check=False)
                text = (out.stdout or out.stderr).strip()
                self._version = text.splitlines()[0] if text else "unknown"
            except (OSError, subprocess.TimeoutExpired):
                self._version = "unknown"
        return self._version
