"""``arc-baselines``: run a system over a Continual-ARC schedule with evidence.

This is the command a workbench experiment spec invokes. It also works from a
plain terminal, in which case an ad-hoc attempt directory is created under
``.workbench/`` so evidence is never lost.

Unless ``--no-proxy`` is given, a local OpenAI-compatible proxy is started for
the run and its address is exported as ``OPENAI_BASE_URL`` / ``OPENROUTER_BASE_URL``
/ ``ARC_LLM_BASE_URL`` (with a placeholder key) before the learner is built, so
any system the learner spawns, and any client the learner builds from the
environment, is measured at the HTTP boundary without further wiring.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from continual_arc.aliases import ALIAS_STYLES

from continual_arc_baselines import secrets


def _import(spec: str, what: str):
    if ":" not in spec:
        raise SystemExit(f"--{what} must look like package.module:Name, got {spec!r}")
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise SystemExit(f"{module_name} has no attribute {attr!r}") from None


def _dataset_label(schedule) -> str:
    """Which task data the schedule draws on, from its provenance."""
    dataset = (schedule.meta or {}).get("dataset") or (schedule.meta or {}).get("config", {}).get("dataset")
    if dataset == "arc2":
        return "arc-agi-2+augmented"
    if dataset in (None, "arc1"):
        return "arc-agi-1+re-arc"
    return str(dataset)


def source_digest() -> str:
    """sha256 over this package's Python sources (first 16 hex digits): which code ran, where the runner's copy of
    the repo is an rsync with no git history."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:16]


def run_parameters(*, learner, learner_spec, learner_kwargs, schedule, config_name, mode,
                   env_config, proxy_settings, alias_style) -> dict:
    """Everything worth recording about a run, in two layers.

    The workbench keeps the whole dict in local evidence but exports to MLflow
    only a fixed allowlist of scalar keys (``SAFE_PARAMS`` in its tracking module:
    programme, benchmark, protocol, method, model, seed, dataset, ...). The first
    block maps onto those names so runs are comparable in the UI; the second block
    is the full detail for the record.
    """
    from continual_arc import __version__ as benchmark_version

    from continual_arc_baselines.evidence import dependency_revisions

    revisions = dependency_revisions()
    arc_rev = revisions["continual-arc"]
    arc_commit = arc_rev.rsplit("@", 1)[-1][:12] if arc_rev.startswith("git+") else arc_rev
    system = dict(learner.describe()) if hasattr(learner, "describe") else {}
    model = system.get("model") or getattr(learner, "model", None)
    from continual_arc.configs import protocol_of

    proto = protocol_of(schedule, env_config)
    return {
        "programme": "continual-arc",
        "benchmark": "continual-arc",
        "track": proto["track"],
        "canonical": proto["canonical"],
        "protocol": f"{config_name or schedule.name}/{mode}",
        "protocol_version": f"{benchmark_version}+{arc_commit}",
        "method": learner_spec,
        "model": model,
        "seed": schedule.seed,
        "dataset": _dataset_label(schedule),
        # full detail, local evidence only
        "learner_kwargs": learner_kwargs,
        "system": system,
        "alias_style": alias_style,
        "schedule": schedule.name,
        "n_slots": len(schedule),
        "env_config": env_config.to_json(),
        "proxy": proxy_settings,
        "dependencies": revisions,
        "protocol_detail": proto,
    }


def cmd_run(args) -> int:
    from continual_arc.configs import build_schedule
    from continual_arc.report import text_report
    from continual_arc.runner import build_env, default_config, load_config, prior_records, run
    from continual_arc.schedule import Schedule

    from continual_arc_baselines.evidence import Evidence
    from continual_arc_baselines.proxy import LLMProxy
    from continual_arc_baselines.tracking import JournalSink, PilotStop, TrackedLearner, Tracker

    if bool(args.schedule) == bool(args.config):
        raise SystemExit("give exactly one of --schedule or --config")
    if args.schedule:
        schedule = Schedule.load(args.schedule)
    else:
        schedule = build_schedule(args.config, args.mode, seed=args.seed)
    # The environment config follows the schedule's dataset: grids for ARC-AGI-1
    # and 2, games for ARC-AGI-3.
    config = load_config(json.loads(Path(args.env_config).read_text())) if args.env_config else default_config(schedule)
    kwargs = json.loads(args.learner_kwargs) if args.learner_kwargs else {}
    registry = _import(args.registry, "registry")() if args.registry else None
    name = args.name or args.learner.rsplit(":", 1)[-1]

    if args.resume and not args.out:
        raise SystemExit("--resume needs --out: the run directory to continue")
    prior = prior_records(args.out) if args.resume else []
    pointer = Path(args.out) / "attempt.json" if args.out else None
    evidence = None
    if prior and pointer is not None and pointer.is_file():
        # Continue in the attempt the interrupted launch used: the system's persisted
        # state (memories, skills, files) lives there and carries on.
        attempt = Path(json.loads(pointer.read_text())["attempt_dir"])
        if attempt.is_dir():
            evidence = Evidence.reopen(attempt, max_cost=config.max_cost)
    if evidence is None:
        evidence = Evidence.from_environment(name=name, max_cost=config.max_cost)
    out_dir = Path(args.out) if args.out else evidence.directory / "continual-arc"
    if args.out:
        out_dir.mkdir(parents=True, exist_ok=True)
        pointer.write_text(json.dumps({"attempt_dir": str(evidence.directory)}))
    if prior:
        evidence.prime(prior)
        print(f"resuming {name}: {len(prior)} instances done, continuing from {len(prior) + 1}", file=sys.stderr)

    tracker = Tracker()
    proxy = None
    proxy_settings: dict = {"enabled": False}
    if not args.no_proxy:
        defaults = dict(json.loads(args.llm_defaults) if args.llm_defaults else {})
        # OpenRouter's sticky routing key: after a cached request it keeps the conversation on
        # the provider that served it, keyed by a hash of the messages unless a session_id is
        # given. Cells of one family open with identical prompts, so the hash would treat them
        # as one session; the attempt id keeps each cell's routing and cache affinity its own.
        # Applied with setdefault, so a system that sets its own wins.
        defaults.setdefault("session_id", evidence.directory.name)
        tracker.subscribe(JournalSink(evidence.context, provider=args.provider,
                                      capture_content=args.capture_content))
        proxy = LLMProxy(args.upstream, api_key=secrets.get(args.upstream_key_env, default=None),
                         declared_context_length=args.declared_context_length or None,
                         tracker=tracker, body_defaults=defaults,
                         capture_content=args.capture_content).start()
        os.environ.update(proxy.environ())
        proxy_settings = {"enabled": True, "upstream": args.upstream, "provider": args.provider,
                          "body_defaults": defaults, "base_url": proxy.base_url,
                          "upstream_key_present": bool(proxy.api_key)}

    learner = _import(args.learner, "learner")(**kwargs)
    if hasattr(learner, "attach"):
        learner.attach(evidence.directory / "system", proxy.environ() if proxy else {},
                       tracker=tracker)
    params = run_parameters(
        learner=learner, learner_spec=args.learner, learner_kwargs=kwargs, schedule=schedule,
        config_name=args.config, mode=args.mode, env_config=config, proxy_settings=proxy_settings,
        alias_style=args.alias_style,
    )
    evidence.parameters(params)
    hold = out_dir / "pilot-hold.json"
    if args.out:
        # What ran, in the run directory itself, which the watchers mirror to the results repo (the attempt's
        # evidence stays on the runner): the system's description (describe(): its version and settings) and this
        # launch's command line, one line per launch or resume (28 Sep).
        with (out_dir / "launches.jsonl").open("a") as fh:
            fh.write(json.dumps({"at": datetime.now(UTC).isoformat(timespec="seconds"),
                                 "resume": bool(args.resume), "prior_records": len(prior), "argv": args.argv,
                                 "system": params["system"], "learner_kwargs": kwargs,
                                 "dependencies": params["dependencies"], "baselines_src": source_digest()},
                                default=str) + "\n")
        if hold.exists() and not args.stop_after:
            hold.unlink()   # launched past its pilot: released

    def progress(record, n_steps):
        evidence.progress(record, n_steps)
        if not args.quiet:
            flag = "ok " if record.solved else ("GU " if record.gave_up else "X  ")
            cost = record.resources.get("llm_cost_usd", 0.0)
            print(
                f"[{record.step + 1:>6}/{n_steps}] {flag} {record.phase:9s} {record.task_id} "
                f"exp={record.exposure_index} cost={record.cost:.1f} usd={cost:.4f}",
                file=sys.stderr,
            )

    try:
        result = run(
            TrackedLearner(learner, tracker, stop_after=args.stop_after, done=len(prior)), schedule, config=config,
            out_dir=out_dir, registry=registry, log_events=args.events, progress=progress, learner_name=name,
            env=build_env(schedule, config, registry), resume=args.resume, alias_style=args.alias_style,
        )
    except PilotStop as stop:
        # stopped by --stop-after: held, and continued by a launch with --resume and without --stop-after
        hold.write_text(json.dumps({"stopped_after": stop.done, "at": datetime.now(UTC).isoformat(
            timespec="seconds"), "argv": args.argv}))
        print(f"pilot: {stop}; held ({hold})", file=sys.stderr)
        return 0
    finally:
        if hasattr(learner, "close"):
            learner.close()
        if proxy is not None:
            proxy.stop()
    emitted = evidence.finish(result)
    print(text_report(result.summary, title=f"{name} on {schedule.name}"), file=sys.stderr)
    print(json.dumps({
        "run_dir": str(out_dir),
        "attempt_dir": str(evidence.directory),
        "inside_workbench": evidence.inside_workbench,
        "overall_score": result.summary["overall"]["score"],
        "stream_score": result.summary["stream"]["score"],
        "epilogue_score": result.summary["epilogue"]["score"],
        "n_metrics_emitted": len(emitted),
        "llm": tracker.usage.as_resources(),
    }))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arc-baselines")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run a learner or system over a schedule, with workbench evidence")
    r.add_argument("--learner", required=True, help="import path, e.g. continual_arc_baselines.systems.prime_agent:PrimeAgentLearner")
    r.add_argument("--learner-kwargs", help="JSON object of constructor kwargs")
    r.add_argument("--schedule", help="schedule JSON from `continual-arc schedule make`")
    r.add_argument("--config", help="named config to build a schedule from: <track>-<size>, e.g. arc1-small, arc2-small, arc3-small (no default track)")
    r.add_argument("--mode", default="workstreams", help="workstreams | sequential | random | fresh (the transfer reference)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--env-config", help="EnvConfig JSON file (default: benchmark defaults)")
    r.add_argument("--registry", help="import path of a zero-arg TaskRegistry factory (testing)")
    r.add_argument("--out", help="run directory (default: <attempt dir>/continual-arc)")
    r.add_argument("--name", help="learner display name")
    r.add_argument("--events", action="store_true", help="also log every action and feedback")
    r.add_argument("--alias-style", choices=ALIAS_STYLES, default="hex",
                   help="the names the system sees tasks under, drawn for the run by the runner: task-3f2a9c1e (hex) "
                        "or a codename such as amber-heron (names)")
    r.add_argument("--resume", action="store_true",
                   help="continue an interrupted run in --out: finished instances are kept, the interrupted one is "
                        "replayed, and the system's persisted state (its attempt directory) is reused")
    r.add_argument("--quiet", action="store_true")
    r.add_argument("--stop-after", type=int, metavar="N",
                   help="stop cleanly once N instances are done, leaving pilot-hold.json in the run directory; a "
                        "launch with --resume and without --stop-after continues it")
    g = r.add_argument_group("model-call tracking (the proxy)")
    g.add_argument("--no-proxy", action="store_true", help="do not start the tracking proxy")
    g.add_argument("--upstream", default="https://openrouter.ai/api/v1",
                   help="OpenAI-compatible base URL the proxy forwards to")
    g.add_argument("--upstream-key-env", default="OPENROUTER_API_KEY",
                   help="name of the environment variable that holds the upstream API key")
    g.add_argument("--provider", default="openrouter", help="provider label for the cost journal")
    g.add_argument("--llm-defaults", help='JSON of chat-completion fields to set when absent, e.g. {"reasoning_effort": "xhigh"}')
    g.add_argument("--declared-context-length", type=int, default=262_144,
                   help="context length the proxy reports on GET /models, the figure every adapter also declares "
                        "in its harness's provider entry (0 passes the provider's figure through)")
    g.add_argument("--capture-content", action="store_true",
                   help="also journal request and response bodies (large; off by default)")
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)   # the command line as given, for launches.jsonl
    return args.func(args)
