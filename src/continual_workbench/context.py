"""Small SDK writing durable metadata without requiring a tracking server."""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import uuid

from .util import atomic_write, canonical, locked, slug


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class RunContext:
    def __init__(self, directory, *, run_id=None, attempt_id=None):
        self.directory = Path(directory).resolve()
        manifest = json.loads((self.directory / "manifest.json").read_text())
        self.run_id = run_id or manifest["run_id"]
        self.attempt_id = attempt_id or manifest["attempt_id"]
        names = json.loads(os.environ.get("WORKBENCH_REDACT_NAMES", "[]"))
        self.secret_values = {name: os.environ[name] for name in names if name in os.environ}
        if (self.run_id, self.attempt_id) != (manifest["run_id"], manifest["attempt_id"]):
            raise ValueError("Context identity disagrees with manifest")

    @classmethod
    def from_environment(cls):
        value = os.environ.get("WORKBENCH_RUN_DIR")
        if not value:
            raise ValueError("Run with workbench or set WORKBENCH_RUN_DIR explicitly")
        return cls(value)

    def emit(self, kind, payload):
        if not isinstance(kind, str) or not isinstance(payload, dict):
            raise ValueError("Events need a kind and object payload")
        def clean(value):
            if isinstance(value, str):
                for secret in sorted(self.secret_values.values(), key=len, reverse=True):
                    if secret:
                        value = value.replace(secret, "[REDACTED]")
                return value
            if isinstance(value, dict):
                return {clean(str(k)): clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            return value
        payload = clean(payload)
        event = {"event_id": uuid.uuid4().hex, "run_id": self.run_id,
                 "attempt_id": self.attempt_id, "kind": kind, "at": timestamp(),
                 "payload": payload}
        data = canonical(event) + b"\n"
        with locked(self.directory / ".events.lock"):
            with (self.directory / "events.jsonl").open("a+b") as handle:
                handle.seek(0, 2)
                if handle.tell():
                    handle.seek(-1, 2)
                    if handle.read(1) != b"\n":
                        raise ValueError("Incomplete event tail; recover evidence before appending")
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        return event

    def metric(self, name, value, step=0):
        if not isinstance(name, str) or not name or not math.isfinite(value):
            raise ValueError("Metrics require a name and finite numeric value")
        if type(step) is not int or step < 0:
            raise ValueError("Metric step must be a nonnegative integer")
        return self.emit("metric", {"name": name, "value": value, "step": step})

    def parameters(self, values):
        return self.emit("parameters", {"values": values})

    def artifact(self, path, name=None):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError("Artifact must be a regular file")
        name = name or path.name
        if Path(name).name != name or name in {".", ".."} or "\\" in name:
            raise ValueError("Artifact names must be a single filename")
        checksum = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(chunk)
        sha = checksum.hexdigest()
        dest = self.directory / "artifacts" / sha / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            pending = dest.parent / (".pending-" + uuid.uuid4().hex)
            shutil.copyfile(path, pending)
            # Verify bytes after copy: source files may be changing during execution.
            copied = hashlib.sha256()
            with pending.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    copied.update(chunk)
            if copied.hexdigest() != sha:
                pending.unlink()
                raise ValueError("Artifact changed while snapshotting")
            os.replace(pending, dest)
        payload = {"name": name, "path": str(dest.relative_to(self.directory)),
                   "sha256": sha, "size": dest.stat().st_size}
        self.emit("artifact", payload)
        return payload


def create_context(root, run_id, manifest):
    slug(run_id)
    attempt = uuid.uuid4().hex
    directory = Path(root).resolve() / "runs" / run_id / "attempts" / attempt
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {**manifest, "run_id": run_id, "attempt_id": attempt,
                "created_at": timestamp(), "schema_version": 1}
    atomic_write(directory / "manifest.json", canonical(manifest))
    return RunContext(directory)
