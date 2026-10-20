<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/continual-arc-readme-banner-dark.png">
    <img src="assets/continual-arc-readme-banner-light.png" alt="Continual-ARC, a unify benchmark" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://unify.ai/continual-arc">Explore every run</a> ·
  <a href="https://unify.ai/blog/why-we-built-continual-arc">Paper</a> ·
  <a href="https://unify.ai/blog/why-we-built-continual-arc">Blog post</a> ·
  <a href="https://unify.ai/continual-arc/leaderboard">Leaderboard</a> ·
  <a href="https://github.com/unifyai/continual-arc">Benchmark</a>
</p>

# continual-arc-baselines

<!-- ARXIV_ID_PENDING: replace every ARXIV_ID_PENDING in this file with the paper's arXiv id once the listing is out (Monday 19 October 2026, 20:00 ET), then delete this line -->
The learners that the paper [*Continual-ARC: Measuring Skill Acquisition over a Working Lifetime*](https://unify.ai/blog/why-we-built-continual-arc)
evaluates on the [Continual-ARC](https://github.com/unifyai/continual-arc) benchmark, and the harness around them: the
adapters that drive three agent harnesses through the benchmark's protocol, the sandbox they run in, the local proxy
through which every model call passes, and the scripts that launch every run the paper reports. The records of those
runs are attached to the benchmark's [v0.1.0 release](https://github.com/unifyai/continual-arc/releases/tag/v0.1.0).

## What is here

| path | contents |
|---|---|
| `src/continual_arc_baselines/systems/bare.py` | the model alone behind the harness protocol: the two floors of row F (`BareModelLearner`, stateless, and `BareDemoCacheLearner`, which shows a task's bought pairs again when it returns) and the program library of row PL (`BareProgramLibraryLearner`; row PL0 with `stateless=True`; the run records name these rows M and M0) |
| `src/continual_arc_baselines/systems/prime_agent.py`, `hermes_agent.py`, `openclaw.py` | the harness adapters of row H: prime-agent over its RPC mode, hermes-agent and openclaw over the Agent Client Protocol (`acp.py`). Row H0 is `stateless=True`, row HA `ask_to_store=True` and row HPL `program_library=True`. `opencode.py` drives a harness that is not in the grid, since it has no mechanism that writes memory or skills |
| `src/continual_arc_baselines/systems/protocol.py`, `base.py` | the messages a learner is shown, the lenient parser that reads an action from a reply, and the adapter base class with its budgets, restarts and wipes |
| `src/continual_arc_baselines/systems/sandbox.py` | the sandbox each harness runs in: bubblewrap on Linux, Seatbelt on macOS |
| `src/continual_arc_baselines/proxy.py`, `tracking.py` | the OpenAI-compatible reverse proxy that records every model call, and the per-instance counters it feeds (`llm_*` in every record) |
| `src/continual_arc_baselines/cli.py` | `arc-baselines run`: builds a schedule, runs a learner over it with the benchmark's runner and writes the run directory |
| `src/continual_arc_baselines/evidence.py`, `src/continual_workbench/` | the attempt directory, event log and cost journal each run writes. `continual_workbench` holds the three modules of our experiment workbench that the baselines use |
| `src/continual_arc_baselines/learners/` | the prompting learners and the program checker that `bare.py` uses |
| `scripts/launch-cell.sh` | launches one cell: one row at one preset, configuration, order and seed |
| `scripts/paper-grid.sh`, `scripts/paper-grid.txt` | every cell the paper reports, one line each |
| `scripts/arc-leaderboard.py` | reads the ARC Prize leaderboard's published data, from which every model in the grid was chosen |

## Install

Linux, with [bubblewrap](https://github.com/containers/bubblewrap) for the sandbox (macOS works with its own sandbox),
Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/unifyai/continual-arc-baselines
cd continual-arc-baselines
uv sync                                                # installs the benchmark from github.com/unifyai/continual-arc
uv run continual-arc data fetch --datasets arc1 arc2   # Re-ARC, ARC-AGI-1 and ARC-AGI-2 at pinned commits
uv run pytest                                          # no network or keys needed
```

On Ubuntu 24.04, bubblewrap needs unprivileged user namespaces, which AppArmor restricts by default:
`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`, as the paper's machines had.

Each harness row needs its harness, at the versions of the paper's Appendix D:

- prime-agent 0.9.5: `curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh`
- hermes-agent 0.20.2 (git `0c3452bf75ce6aecc603954d44c8d470d2a20c15`): a checkout of
  [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) at `~/hermes-agent` with
  `uv pip install -e ".[acp]"` in its virtual environment, or pass `command` to the adapter
- openclaw 2026.9.5: `curl -fsSL https://openclaw.ai/install-cli.sh | bash`, or the npm package

## Run a learner

```bash
# a smoke run with a learner that never solves anything: no model, no key
uv run arc-baselines run --learner continual_arc.testing:GiveUpLearner --config arc1-smoke --mode workstreams --seed 0

# the stateless floor on the ARC-AGI-1 track at low effort, as the paper ran it
export OPENROUTER_API_KEY=...
bash scripts/launch-cell.sh floors luna6-low workstreams 0 arc1-small
```

A run directory holds `records.jsonl` (one line per instance), `summary.json`, `schedule.json`, `config.json` and
`meta.json`, the files of every run in the run records; `uv run continual-arc report <run directory>` prints its
scores. `arc-baselines run` starts the tracking proxy for the run and exports its address as `OPENAI_BASE_URL`,
`OPENROUTER_BASE_URL` and `ARC_LLM_BASE_URL` with a placeholder key, so any system pointed at it is measured without
knowing it; the real key stays in the proxy. The proxy forwards each request unchanged except that it asks the
provider to report usage and adds the run's defaults (`--llm-defaults`) where the system set none.

To run your own system, subclass `SystemLearner` (`systems/base.py`), which speaks the benchmark's protocol and leaves
an adapter only `start`, `stop`, `ask`, `new_session` and `wipe_state`; or write any learner against the benchmark's
four-method interface that builds its client from `OPENAI_BASE_URL`.

## Reproduce the paper's runs

`bash scripts/paper-grid.sh` launches every cell the paper reports: 254 runs on the two tracks, ARC-AGI-1 on GPT-6
Luna at low, medium and high effort and ARC-AGI-2 at medium to max effort (the floors at low too). Each writes `runs/<track>/<cell>/`, under the name its record carries in the run
records. Every cell is detached and long-lived,
so launch a part at a time (`bash scripts/paper-grid.sh arc2`), and `DRY_RUN=1` shows each cell's command first. The
paper's 254 runs made about 1,800 USD of model calls through OpenRouter, by the cost the proxy recorded on each
instance. The models are sampled through provider APIs, so a rerun differs from the paper's by about the spread the
paper reports.

The paper's cells were launched between 27 and 29 September 2026, each row from the code of its launch day. The
learners here are that code as it stood once the last row was launched; since then only a limit on the upstream
connections the proxy keeps open (`max_connections`) was added. The cells launched before 17:11 UTC on 28 September ran
under the earlier rule for empty replies, and `paper-grid.txt` launches them under it.

## Licence

Apache-2.0, like the benchmark.

## About

The baselines are part of [Continual-ARC](https://github.com/unifyai/continual-arc), made by
[unify](https://unify.ai). unify is building continual learning for the agents companies have already built, training
the model to drive their own harness, so the agent gets faster and cheaper the more it runs, on a model they own. We
publish the research behind it.
