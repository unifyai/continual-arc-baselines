#!/usr/bin/env python3
"""The ARC Prize leaderboard, read from its own data files, as a table.

The live leaderboard is https://arcprize.org/leaderboard. The page is rendered from four public
JSON files under https://arcprize.org/media/data/ (datasets, models, providers, evaluations), and
this script reads those directly, so what it prints is exactly what the site plots: verified
semi-private scores with cost per task, one row per (model, reasoning effort). Use it to pick and
sanity-check every model the harness rows run on: a candidate for a track needs a verified score
in the band where a harness has room to learn, and nothing about a model should be taken from
memory when this table is one command away.

Usage:
  arc-leaderboard.py [--dataset arc1|arc2] [--min-score S] [--max-score S] [--max-cost C]
                     [--grep PATTERN] [--band] [--openrouter] [--refresh] [--json]
  --band       the calibration band for ARC-AGI-1: score 0.30 to 0.65 at up to 0.10 USD a task,
               the region where a model solves enough to drive a harness and misses enough to learn
  --openrouter also list the OpenRouter model ids whose name carries the same tokens (needs
               OPENROUTER_API_KEY in the environment); a hint, not an identity
  --refresh    fetch the files again (they are cached for an hour under .workbench/arc-leaderboard/)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

SITE = "https://arcprize.org"
LEADERBOARD = f"{SITE}/leaderboard"
FILES = ("datasets", "models", "providers", "evaluations")
DATASETS = {"arc1": "v1_Semi_Private", "arc2": "v2_Semi_Private"}
CACHE = Path(__file__).resolve().parent.parent / ".workbench" / "arc-leaderboard"
CACHE_TTL_S = 3600


def fetch(name: str, refresh: bool) -> list:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.json"
    if refresh or not path.exists() or time.time() - path.stat().st_mtime > CACHE_TTL_S:
        req = urllib.request.Request(f"{SITE}/media/data/{name}.json", headers={"User-Agent": "continual-arc-baselines"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            path.write_bytes(resp.read())
    return json.loads(path.read_text())


def tokens(s: str) -> set[str]:
    s = re.sub(r"\(.*?\)", "", s.lower())
    s = re.sub(r"\b(openai|anthropic|google|thinky|thinkingmachines|deepseek|reasoning|thinking|preview)\b", " ", s)
    return {t for t in re.split(r"[^a-z0-9]+", s) if t and not re.fullmatch(r"\d{4}|\d{2}", t)}


def openrouter_catalogue() -> list[dict]:
    key = os.environ["OPENROUTER_API_KEY"]
    req = urllib.request.Request("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["data"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="arc1")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-score", type=float, default=1.0)
    ap.add_argument("--max-cost", type=float, default=float("inf"), help="USD per task")
    ap.add_argument("--grep", help="case-insensitive pattern on the model id or display name")
    ap.add_argument("--band", action="store_true")
    ap.add_argument("--openrouter", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the rows as JSON instead of a table")
    args = ap.parse_args()
    if args.band:
        args.dataset, args.min_score, args.max_score, args.max_cost = "arc1", 0.30, 0.65, 0.10

    models = {m["id"]: m for m in fetch("models", args.refresh)}
    providers = {p["id"]: p for p in fetch("providers", args.refresh)}
    evaluations = fetch("evaluations", args.refresh)
    dataset = DATASETS[args.dataset]
    rows = []
    for e in evaluations:
        if e.get("datasetId") != dataset or not e.get("display", True):
            continue
        m = models.get(e["modelId"], {})
        score = float(e.get("score") or 0.0)
        cost = e.get("costPerTask")
        if not (args.min_score <= score <= args.max_score):
            continue
        if cost is not None and cost > args.max_cost:
            continue
        name = m.get("displayName") or e["modelId"]
        if args.grep and not re.search(args.grep, e["modelId"] + " " + name, re.I):
            continue
        rows.append({
            "model_id": e["modelId"], "name": name, "provider": providers.get(m.get("providerId"), {}).get("displayName", m.get("providerId")),
            "released": (m.get("modelReleaseDate") or "")[:10], "score": score, "cost_per_task": cost, "results_url": e.get("resultsUrl") or "",
        })
    rows.sort(key=lambda r: (-r["score"], r["cost_per_task"] if r["cost_per_task"] is not None else 1e9))

    if args.openrouter and rows:
        catalogue = [(c["id"], tokens(c["id"] + " " + c.get("name", ""))) for c in openrouter_catalogue() if ":free" not in c["id"]]
        for r in rows:
            want = tokens(models.get(r["model_id"], {}).get("modelGroup") or r["model_id"])
            r["openrouter"] = sorted(cid for cid, have in catalogue if want and want <= have)[:3]

    if args.json:
        print(json.dumps(rows, indent=1))
        return 0
    print(f"ARC Prize leaderboard, {dataset} ({args.dataset}), {len(rows)} rows; live at {LEADERBOARD}")
    print(f"{'score':>6s}  {'$/task':>8s}  {'released':10s}  {'provider':14s}  {'model':44s}  name" + ("  openrouter" if args.openrouter else ""))
    for r in rows:
        cost = f"{r['cost_per_task']:8.3f}" if r["cost_per_task"] is not None else "       -"
        line = f"{r['score']*100:5.1f}%  {cost}  {r['released']:10s}  {str(r['provider'])[:14]:14s}  {r['model_id'][:44]:44s}  {r['name'][:36]}"
        if args.openrouter:
            line += "  " + (", ".join(r.get("openrouter") or []) or "-")
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
