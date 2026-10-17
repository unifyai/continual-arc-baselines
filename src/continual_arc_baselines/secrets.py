"""Credentials, from the environment.

The one credential a run needs is the upstream model provider's key (``OPENROUTER_API_KEY`` by default), and it lives
only in the tracking proxy's process, never in a system's configuration or logs.
"""

from __future__ import annotations

import os
from typing import Any

_MISSING = object()


def get(name: str, default: Any = _MISSING) -> str:
    """The value of ``name`` in the environment."""
    value = os.environ.get(name)
    if value:
        return value
    if default is _MISSING:
        raise RuntimeError(f"{name} is not in the environment")
    return default
