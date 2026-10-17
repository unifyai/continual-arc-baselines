"""Append-only provider accounting; costs are decimal strings, unknowns stay null.

Sources: https://openrouter.ai/docs/cookbook/administration/usage-accounting
USD denomination: OpenRouterTeam/terraform-provider-openrouter docs/data-sources/api_key.md.
Provider credits are not an invoice including purchases, fees, or taxes.
"""

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
import os
from pathlib import Path

TOKEN_FIELDS = ("input", "output", "cache_read", "cache_write", "reasoning")
COST_FIELDS = ("account_charge", "upstream_inference_cost")


class JournalCorruptionError(ValueError):
    """Evidence needs explicit recovery; no automatic truncation is attempted."""


def _money(value):
    if value is None:
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("Cost must be a finite decimal") from None
    if not result.is_finite() or result < 0:
        raise ValueError("Cost must be a finite nonnegative decimal")
    return format(result, "f")


def _count(value):
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError("Token count must be a nonnegative integer or null")
    return value


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def normalize_openrouter(response: dict, headers: dict | None = None) -> dict:
    """Extract only accounting metadata from completion or generation lookup JSON.

    Cache/reasoning counts are subsets, not extra tokens to add to input/output.
    """
    if not isinstance(response, dict):
        raise TypeError("Response must be an object")
    lookup = isinstance(response.get("data"), dict)
    data = response["data"] if lookup else response
    usage = data if lookup else _mapping(data.get("usage"))
    prompt = _mapping(usage.get("prompt_tokens_details"))
    completion = _mapping(usage.get("completion_tokens_details"))
    headers = {str(k).lower(): v for k, v in (headers or {}).items()}
    values = (
        [usage.get("native_tokens_prompt"), usage.get("native_tokens_completion"),
         usage.get("native_tokens_cached"), None, usage.get("native_tokens_reasoning")]
        if lookup else
        [usage.get("prompt_tokens"), usage.get("completion_tokens"),
         prompt.get("cached_tokens"), prompt.get("cache_write_tokens"),
         completion.get("reasoning_tokens")]
    )
    result = {
        "provider": "openrouter", "origin": "reconciliation" if lookup else "response",
        "generation_id": data.get("id"),
        "request_id": data.get("request_id") or headers.get("x-request-id")
        or headers.get("x-openrouter-request-id"),
        "model": data.get("model"), "resolved_provider": data.get("provider_name") or data.get("provider"),
        "response_cache_status": headers.get("x-openrouter-cache-status"),
        "response_cache_age": headers.get("x-openrouter-cache-age"),
        "finish_reason": data.get("finish_reason") or next((c.get("finish_reason")
            for c in data.get("choices", []) if c.get("finish_reason")), None),
        "currency": "USD", "unit": "openrouter_credits",
        "upstream_unit": "USD",
        "tokens": dict(zip(TOKEN_FIELDS, map(_count, values))),
        "account_charge": _money(usage.get("total_cost") if lookup else usage.get("cost")),
        "upstream_inference_cost": _money(
            usage.get("upstream_inference_cost") if lookup else
            _mapping(usage.get("cost_details")).get("upstream_inference_cost")),
    }
    if lookup:
        result["normalized_tokens"] = {
            "input": _count(usage.get("tokens_prompt")),
            "output": _count(usage.get("tokens_completion")),
        }
        result["is_byok"] = data.get("is_byok")
    return result


class CostJournal:
    """A process-safe local journal. Share one directory to aggregate run attempts.

    Corrections are appended, never edits. The summary resolves observations by
    generation identity; reconciled non-null values supersede response values.
    POSIX advisory locking requires all writers to use this API on a local filesystem.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "costs.jsonl"

    @contextmanager
    def _locked(self):
        fd = os.open(self.directory / ".costs.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _read(self):
        if not self.path.exists():
            return []
        records = []
        with self.path.open("rb") as handle:
            for number, line in enumerate(handle, 1):
                if not line.endswith(b"\n"):
                    raise JournalCorruptionError(f"Incomplete journal tail at line {number}")
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict) or not record.get("source_id"):
                        raise ValueError()
                except (ValueError, UnicodeDecodeError):
                    raise JournalCorruptionError(f"Invalid journal record at line {number}") from None
                records.append(record)
        return records

    def record(self, record: dict) -> dict:
        value = dict(record)
        for field in COST_FIELDS:
            value[field] = _money(value.get(field))
        value["tokens"] = {k: _count(value.get("tokens", {}).get(k)) for k in TOKEN_FIELDS}
        if value.get("origin") == "local_replay":
            if any(value[k] is None or Decimal(value[k]) != 0 for k in COST_FIELDS):
                raise ValueError("Local replay must explicitly declare zero charges")
            if value.get("generation_id"):
                raise ValueError("Replay needs a separate identity, not the paid generation identity")
        if not value.get("provider") or not value.get("origin"):
            raise ValueError("Provider and origin are required")
        if not value.get("source_id"):
            value["source_id"] = hashlib.sha256(_canonical(value).encode()).hexdigest()
        _canonical(value)
        with self._locked():
            records = self._read()
            for old in records:
                if old["source_id"] == value["source_id"]:
                    if old != value:
                        raise ValueError("Source identity already exists with different data")
                    return old
            fd = os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(_canonical(value).encode() + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        return value

    def summary(self) -> dict:
        with self._locked():
            records = self._read()
        return summarize_records(records)


def summarize_records(records: list[dict]) -> dict:
    """Resolve normalized journal observations, deduplicating replayed source events."""
    unique = {}
    for record in records:
        source = record["source_id"]
        if source in unique and unique[source] != record:
            raise ValueError("Conflicting observations for one source identity")
        unique[source] = record
    records = list(unique.values())
    aliases = {}
    for row in records:
        if row.get("generation_id") and row.get("request_attempt_id"):
            alias = (row["provider"], row["request_attempt_id"])
            identity = (row["provider"], "generation", row["generation_id"])
            if alias in aliases and aliases[alias] != identity:
                raise ValueError("One physical request has conflicting generation identities")
            aliases[alias] = identity
    groups = {}
    for row in records:
        provider = row["provider"]
        if row["origin"] == "local_replay":
            key = (provider, "replay", row.get("request_attempt_id") or row["source_id"])
        elif row.get("generation_id"):
            key = (provider, "generation", row["generation_id"])
        elif row.get("request_attempt_id"):
            key = aliases.get((provider, row["request_attempt_id"]),
                              (provider, "attempt", row["request_attempt_id"]))
        else:
            key = (provider, "source", row["source_id"])
        groups.setdefault(key, []).append(row)
    effective = []
    ranks = {"request_started": 0, "transport_error": 1, "response": 2,
             "stream": 2, "local_replay": 2, "reconciliation": 3}
    for key, rows in groups.items():
        owner = next((r for r in rows if r["origin"] != "reconciliation"), rows[0])
        currencies = {(r.get("currency"), r.get("unit")) for r in rows
                      if r.get("currency") or r.get("unit")}
        if len(currencies) != 1:
            raise ValueError("Generation currency/unit missing or inconsistent")
        currency, unit = next(iter(currencies))
        entity = {"identity": list(key), "run_id": owner.get("run_id"),
                  "attempt_id": owner.get("attempt_id"), "currency": currency, "unit": unit,
                  "account_charge": None, "upstream_inference_cost": None,
                  "tokens": dict.fromkeys(TOKEN_FIELDS),
                  "origins": sorted({r["origin"] for r in rows})}
        for row in sorted(rows, key=lambda r: ranks.get(r["origin"], 1)):
            for field in COST_FIELDS:
                if row.get(field) is not None:
                    entity[field] = row[field]
            for field in TOKEN_FIELDS:
                if row["tokens"].get(field) is not None:
                    entity["tokens"][field] = row["tokens"][field]
        effective.append(entity)

    def totals(items):
        result = {"entities": len(items), "tokens": {}, "charges": []}
        for field in TOKEN_FIELDS:
            known = [r["tokens"][field] for r in items if r["tokens"][field] is not None]
            result["tokens"][field] = {"known_total": sum(known),
                                        "unknown_entities": len(items) - len(known)}
        units = sorted({(r["currency"], r["unit"]) for r in items})
        for currency, unit in units:
            subset = [r for r in items if (r["currency"], r["unit"]) == (currency, unit)]
            charge = {"currency": currency, "unit": unit, "upstream_unit": "USD"
                      if all(r["identity"][0] == "openrouter" for r in subset) else None}
            for field in COST_FIELDS:
                known = [Decimal(r[field]) for r in subset if r[field] is not None]
                charge[field] = {"known_total": format(sum(known, Decimal(0)), "f"),
                                 "unknown_entities": len(subset) - len(known)}
            result["charges"].append(charge)
        return result

    attempts = []
    identities = {(r["run_id"], r["attempt_id"]) for r in effective}
    for run_id, attempt_id in sorted(identities, key=str):
        subset = [r for r in effective if (r["run_id"], r["attempt_id"]) == (run_id, attempt_id)]
        attempts.append({"run_id": run_id, "attempt_id": attempt_id, **totals(subset)})
    return {"schema_version": 1, "records": len(records), **totals(effective),
            "attempts": attempts, "effective": effective}
