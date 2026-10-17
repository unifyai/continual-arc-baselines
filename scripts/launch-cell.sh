#!/usr/bin/env bash
# Launch one cell of the paper's grid: `arc-baselines run` for one row at one preset, configuration, order and seed,
# detached, writing the run directory runs/<track>/<cell>/ and its log beside it. A cell whose run directory exists is
# skipped, so a line can be run again; RESUME=1 continues a cell that stopped part-way. DRY_RUN=1 prints each cell's
# run directory and command and runs nothing. RUNS=<dir> writes somewhere other than runs/.
#
#   bash scripts/launch-cell.sh floors <preset> <mode> <seed> [config]           row F: the bare model, stateless and
#                                                                                 with its demonstration cache (two cells)
#   bash scripts/launch-cell.sh method <preset> <mode> <seed> [config]           row M: the bare program library
#   bash scripts/launch-cell.sh harness <preset> <config> <mode> <seed>          row H: prime-agent, every learning
#                                                                                 mechanism it ships switched on
#   bash scripts/launch-cell.sh acp <mode> <seed> [config] [preset] [hermes|openclaw]   row H: hermes-agent and openclaw
#
# A prefix on the verb gives the other rows: stateless-harness and stateless-acp (row H0, persistence switched off:
# stopped, wiped and restarted before every instance), stateless-method (row M0), ask-harness and ask-acp (row HA,
# asked to store after every verdict), pl-harness and pl-acp (row HPL, told to keep a program library as row M does),
# stateless-pl-harness and stateless-pl-acp (row HPL0). Modes are workstreams, sequential, random and fresh (the
# transfer reference); configurations arc1-small, arc2-small and arc3-small.
#
# An empty reply is re-prompted like any unusable reply, and a system is restarted only when its process is dead or
# silent. EMPTY_REPLY=restart gives the earlier rule, under which an empty reply restarted the system: the paper's
# cells launched before 17:11 UTC on 28 September 2026 ran under it (scripts/paper-grid.txt marks them; the
# paper's Appendix D).
#
# Model calls go through OpenRouter: OPENROUTER_API_KEY must be in the environment. The harness rows need the harness
# installed (README) and Linux with bubblewrap for the sandbox.
set -uo pipefail
kind=$1; shift
REPLY_KW=', "empty_reply": "reprompt", "archive_wipes": true'
[ "${EMPTY_REPLY:-reprompt}" = restart ] && REPLY_KW=""
ROW=H; STATELESS_KW=""; ASK_KW=""; suffix=""; zero=""
while :; do
    case $kind in
        stateless-*) zero=0; STATELESS_KW=', "stateless": true'; kind=${kind#stateless-};;
        ask-*) suffix=A; ASK_KW=', "ask_to_store": true'; kind=${kind#ask-};;
        pl-*) suffix=PL; ASK_KW=', "program_library": true'; kind=${kind#pl-};;
        *) break;;
    esac
done
ROW="H$suffix$zero"
[ "$ROW" = HA0 ] && { echo "a stateless row is never asked to store (ask_to_store): no HA0"; exit 1; }
case $kind in floors|method) [ -n "$suffix" ] && { echo "ask- and pl- are harness rows: not for $kind"; exit 1; };; esac
[ "$kind" = floors ] && [ -n "$zero" ] && { echo "the floors need no stateless- (BareModelLearner is the stateless floor)"; exit 1; }
export PATH="$HOME/.local/bin:$HOME/.opencode/bin:$HOME/.openclaw/bin:$PATH"
cd "$(dirname "$0")/.."
RUNS=${RUNS:-runs}
modeabbr() { case $1 in workstreams) echo ws;; sequential) echo seq;; random) echo rnd;; fresh) echo fresh;; *) echo "${1:0:3}";; esac; }
launch() {  # cell learner kwargs defaults config mode seed
    local cell=$1 learner=$2 kwargs=$3 defaults=$4 cfg=$5 mode=$6 seed=$7
    local dir="$RUNS/${cfg%%-*}/$cell"
    local cmd=(uv run --no-sync python -m continual_arc_baselines run --learner "$learner" --learner-kwargs "$kwargs"
               --llm-defaults "$defaults" --config "$cfg" --mode "$mode" --seed "$seed" --out "$dir" --events --quiet)
    [ -n "${RESUME:-}" ] && cmd+=(--resume)
    if [ -n "${DRY_RUN:-}" ]; then echo "$dir"; printf '  %q' "${cmd[@]}"; echo; return; fi
    [ -d "$dir" ] && [ -z "${RESUME:-}" ] && { echo "exists: $dir"; return; }
    mkdir -p "$(dirname "$dir")"
    nohup "${cmd[@]}" > "$dir.log" 2>&1 &
    echo "launched $dir"
}
# A preset names a model at one reasoning effort, the pair the ARC Prize leaderboard verifies. Each grid model was
# chosen there (scripts/arc-leaderboard.py): verified zero-shot between about 15 and 60 percent on its track, the band
# where a harness has room to learn. The grid is one family, GPT-6 Luna, with effort as the ability axis, and Claude
# Opus 5.5 at low effort on ARC-AGI-3. The leaderboard figures are those of September 2026.
model_of() {
    case $1 in
      luna6-low)    model=openai/gpt-6-luna;         effort=low;    tag=luna6-low ;;     # ARC-AGI-1 37.7% at 0.003 USD/task
      luna6-medium) model=openai/gpt-6-luna;         effort=medium; tag=luna6-medium ;;  # ARC-AGI-1 61.0% at 0.006
      luna6-high)   model=openai/gpt-6-luna;         effort=high;   tag=luna6-high ;;    # ARC-AGI-1 70.3% at 0.009; ARC-AGI-2 31.4% at 0.023
      luna6-xhigh)  model=openai/gpt-6-luna;         effort=xhigh;  tag=luna6-xhigh ;;   # ARC-AGI-2 41.9% at 0.032
      luna6-max)    model=openai/gpt-6-luna;         effort=max;    tag=luna6-max ;;     # ARC-AGI-2 59.3% at 0.062
      opus55-low)   model=anthropic/claude-opus-5.5; effort=low;    tag=opus55-low ;;    # ARC-AGI-3, not on the board
      *) echo "unknown preset $1"; exit 1 ;;
    esac
}
case $kind in
harness)
    preset=$1; cfg=$2; mode=$3; seed=$4
    model_of "$preset"
    m=$(modeabbr "$mode"); track=${cfg%%-*}; [ "$track" != arc1 ] && tag="$tag-$track"
    case $cfg in *-smoke) tag="$tag-smoke";; esac   # a smoke never takes a grid cell's name
    think=$effort; [ "$think" = minimal ] && think=low   # prime-agent's own thinking levels stop at low; the model call carries minimal
    launch "$ROW-prime-agent-$tag-$m$seed" continual_arc_baselines.systems.prime_agent:PrimeAgentLearner \
        "{\"model\": \"$model\", \"thinking\": \"$think\", \"end\": \"finish\"$REPLY_KW, \"action_budget_s\": 300, \"max_model_calls\": 15$STATELESS_KW$ASK_KW}" \
        "{\"reasoning_effort\": \"$effort\"}" "$cfg" "$mode" "$seed" ;;
acp)
    mode=$1; seed=$2; cfg=${3:-arc1-small}; preset=${4:-luna6-low}; only=${5:-all}; m=$(modeabbr "$mode")
    model_of "$preset"
    track=${cfg%%-*}; [ "$track" != arc1 ] && tag="$tag-$track"
    case $cfg in *-smoke) tag="$tag-smoke";; esac
    acp() { [ "$only" != all ] && [ "$only" != "$1" ] && return
        # the rows that change what a harness is told (HA, HPL, HPL0) set openclaw's memory search, left unset otherwise
        local oc=""; [ -n "$suffix" ] && [ "$1" = openclaw ] && oc=', "memory_search": true'
        launch "$ROW-$1-$tag-$m$seed" "continual_arc_baselines.systems.$2" \
        "{\"model\": \"$model\", \"timeout_s\": 600, \"idle_timeout_s\": 300, \"end\": \"finish\"$REPLY_KW$STATELESS_KW$ASK_KW$oc}" \
        "{\"reasoning_effort\": \"$effort\"}" "$cfg" "$mode" "$seed"; }
    acp hermes hermes_agent:HermesAgentLearner
    acp openclaw openclaw:OpenClawLearner ;;
floors)
    # Row F: the bare model behind the harness protocol (systems/bare.py), with no tools and no memory, one model call
    # per message, deciding itself when to request a pair and when to submit. BareModelLearner is the stateless floor;
    # BareDemoCacheLearner the demo-cache floor, the same model with the pairs it bought for a task (up to eight) shown
    # again free when the task returns. max_tokens is the model's maximum completion length (128k for GPT-6 Luna).
    preset=$1; mode=$2; seed=$3; cfg=${4:-arc1-small}; m=$(modeabbr "$mode")
    model_of "$preset"
    track=${cfg%%-*}; [ "$track" != arc1 ] && tag="$tag-$track"
    case $cfg in *-smoke) tag="$tag-smoke";; esac
    for L in BareModelLearner BareDemoCacheLearner; do
        launch "F-$L-$tag-$m$seed" "continual_arc_baselines.systems.bare:$L" \
            "{\"model\": \"$model\", \"max_tokens\": 128000, \"end\": \"finish\"$REPLY_KW}" \
            "{\"reasoning\": {\"effort\": \"$effort\"}}" "$cfg" "$mode" "$seed"
    done ;;
method)
    # Row M: the program library, the bare demo-cache floor with a program for every submission (systems/bare.py,
    # BareProgramLibraryLearner): each program is checked free on the task's pairs, and the last one that reproduces
    # all of them is kept and run first when the task returns. Row M0 (stateless-method) wipes the kept programs and
    # the demonstration cache before every instance.
    preset=$1; mode=$2; seed=$3; cfg=${4:-arc1-small}; m=$(modeabbr "$mode")
    model_of "$preset"
    track=${cfg%%-*}; [ "$track" != arc1 ] && tag="$tag-$track"
    case $cfg in *-smoke) tag="$tag-smoke";; esac
    mrow=M; [ -n "$zero" ] && mrow=M0
    launch "$mrow-BareProgramLibraryLearner-$tag-$m$seed" "continual_arc_baselines.systems.bare:BareProgramLibraryLearner" \
        "{\"model\": \"$model\", \"max_tokens\": 128000, \"end\": \"finish\"$REPLY_KW$STATELESS_KW}" \
        "{\"reasoning\": {\"effort\": \"$effort\"}}" "$cfg" "$mode" "$seed" ;;
*) echo "usage: launch-cell.sh [stateless-|ask-|pl-]harness|acp|floors|method ..."; exit 1 ;;
esac
