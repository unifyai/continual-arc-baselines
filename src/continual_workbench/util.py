"""Stable identities and crash-safe local writes."""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


def plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        return {"tensor": value.detach().cpu().tolist(), "dtype": str(value.dtype)}
    if hasattr(value, "tolist"):
        return {"array": value.tolist(), "dtype": str(value.dtype)}
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(value: Any) -> str:
    def compact(item):
        if dataclasses.is_dataclass(item):
            return compact(dataclasses.asdict(item))
        if isinstance(item, dict):
            return {str(k): compact(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [compact(v) for v in item]
        if hasattr(item, "detach"):
            tensor = item.detach().cpu().contiguous()
            data = tensor.reshape(-1).view(__import__('torch').uint8).numpy().tobytes()
            return {"tensor_sha256": byte_digest(data), "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype)}
        if hasattr(item, "tobytes") and hasattr(item, "dtype"):
            return {"array_sha256": byte_digest(item.tobytes()), "shape": list(item.shape),
                    "dtype": str(item.dtype)}
        return item
    return hashlib.sha256(canonical(compact(value))).hexdigest()


def byte_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def slug(value: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", value):
        raise ValueError(f"Expected a lowercase kebab-case identifier: {value!r}")
    return value


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def locked(path: Path, *, blocking: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
