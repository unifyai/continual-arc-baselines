"""End to end: CLI -> proxy -> prime-agent adapter -> fake prime-agent -> fake upstream."""

import json
import os
import platform
import sys
from pathlib import Path

import pytest
from continual_arc.schedule import epilogue_schedule, stream_schedule
from continual_arc.testing import FakeSource
from continual_workbench.accounting import CostJournal
from continual_workbench.context import create_context

from continual_arc_baselines.cli import main

FAKE = str(Path(__file__).with_name("fake_prime_agent.py"))
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def upstream():
    from .fake_upstream import FakeUpstream

    server = FakeUpstream().start()
    yield server
    server.stop()


def run_mode(tmp_path, monkeypatch, upstream, mode: str, extra_kwargs: dict | None = None,
             n_slots: int = 12):
    source = FakeSource(n_tasks=4)
    sched = tmp_path / f"{mode}.json"
    if mode == "fresh":
        # the transfer reference: every task met by a fresh system, three instances each
        epilogue_schedule(source.task_ids(), seed=1, source=source.name, reset_each=True, instances_per_task=3).save(sched)
    else:
        stream_schedule(source.task_ids(), n_slots, seed=1, source=source.name).save(sched)
    context = create_context(tmp_path / "wb", f"pa-{mode}", {"experiment_id": "t", "project_key": "t"})
    monkeypatch.setenv("WORKBENCH_RUN_DIR", str(context.directory))
    monkeypatch.setenv("FAKE_AGENT_LOG", str(tmp_path / f"{mode}-commands.log"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-upstream")
    kwargs = {"command": [sys.executable, FAKE], "model": "vendor/model", "thinking": "low",
              "startup_timeout_s": 30, "timeout_s": 30, "sandbox": "none",
              # The child environment is an allowlist; the fake's own switches must be named.
              "passthrough_env": [k for k in os.environ if k.startswith("FAKE_AGENT_")],
              **(extra_kwargs or {})}
    code = main(["run", "--learner", "continual_arc_baselines.systems.prime_agent:PrimeAgentLearner",
                 "--learner-kwargs", json.dumps(kwargs), "--schedule", str(sched),
                 "--registry", "continual_arc_baselines.testing:fake_registry",
                 "--upstream", upstream.base_url, "--llm-defaults", '{"reasoning_effort": "low"}',
                 "--quiet"])
    assert code == 0
    return context, tmp_path / f"{mode}-commands.log"


def records_of(context):
    return [json.loads(line) for line in (context.directory / "continual-arc" / "records.jsonl").read_text().splitlines()]


def adapter_log(context) -> str:
    return (context.directory / "system" / "adapter.log").read_text()


def test_workstreams_learns_and_is_measured_at_the_boundary(tmp_path, monkeypatch, upstream, capsys):
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams")
    outcome = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert outcome["stream_score"] > 0
    records = records_of(context)
    # First exposure needs demos (cost 1); later exposures of a task are solved from memory (cost 0).
    first = [r for r in records if r["exposure_index"] == 0 and r["phase"] == "stream"]
    later = [r for r in records if r["exposure_index"] > 0 and r["phase"] == "stream"]
    assert first and all(r["cost"] == 1.0 for r in first)
    assert later and all(r["cost"] == 0.0 for r in later)
    # Proxy-measured usage lands in every record and in the cost journal: one model call per
    # action, plus the closing turn that carries the verdict (paid and journaled, never scored).
    assert all(r["resources"]["llm_requests"] == r["resources"]["actions"] + 1 for r in records)
    assert all(r["resources"]["llm_cost_usd"] > 0 for r in records)
    # one turn per action plus the closing turn that carries the instance's verdict
    assert all(r["resources"]["system_turns"] == r["resources"]["actions"] + 1 for r in records)
    assert all(r["resources"]["system_closings"] == 1 for r in records)
    summary = CostJournal(context.directory).summary()
    assert summary["entities"] == sum(r["resources"]["actions"] + 1 for r in records)
    # The system never saw the upstream key, and the run-level default reached the model call.
    sent = upstream.requests[0]
    assert sent["authorization"] == "Bearer sk-upstream" and sent["body"]["reasoning_effort"] == "low"
    params = next(json.loads(line)["payload"]["values"] for line in (context.directory / "events.jsonl").read_text().splitlines()
                  if json.loads(line)["kind"] == "parameters")
    assert params["model"] == "vendor/model" and params["system"]["system_version"] == "fake-prime-agent 0.0.1"
    assert params["proxy"]["enabled"] and params["proxy"]["upstream_key_present"]
    assert params["system"]["sandbox"] == "none" and params["system"]["task_aliases"] is True
    # No resets in workstreams mode, one process for the whole run, and a fresh conversation for
    # every instance after the first.
    assert commands.read_text().count("new_session") == len(records) - 1
    assert (context.directory / "system" / "state" / "agent" / "models.json").is_file()
    models = json.loads((context.directory / "system" / "state" / "agent" / "models.json").read_text())
    entry = next(iter(models["providers"].values()))["models"][0]
    assert entry["thinkingLevelMap"] == {"xhigh": "xhigh", "max": "max"}  # else pi-ai clamps both to high
    assert models["providers"]["arc-proxy"]["baseUrl"] == params["proxy"]["base_url"]


def test_system_never_sees_real_task_ids(tmp_path, monkeypatch, upstream):
    """Task ids are lookup keys into public solutions; systems get run-specific aliases."""
    context, _ = run_mode(tmp_path, monkeypatch, upstream, "workstreams")
    # Everything the system was told went through the proxy (the fake forwards its prompts).
    prompts = " ".join(json.dumps(r["body"]) for r in upstream.requests)
    assert "Task id: task-" in prompts
    for task_id in {r["task_id"] for r in records_of(context)}:
        assert task_id not in prompts
    # The records on disk keep the real ids, so analysis is unaffected.
    assert all(r["task_id"].startswith("fake") for r in records_of(context))


def test_fresh_reference_wipes_state_between_tasks(tmp_path, monkeypatch, upstream, capsys):
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "fresh")
    records = records_of(context)
    by_task: dict[str, list[dict]] = {}
    for r in records:
        by_task.setdefault(r["task_id"], []).append(r)
    for rows in by_task.values():
        # Within a task block the memory persists; the next block starts from nothing.
        assert rows[0]["cost"] == 1.0 and all(r["cost"] == 0.0 for r in rows[1:])
    resets = [r for r in records if r["reset_before"]]
    assert len(resets) == len(by_task)
    # A reset stops the process, wipes its directory and starts afresh (one get_state handshake per start).
    assert commands.read_text().count("get_state") == len(by_task)
    # The adapter's own forensics survive the wipes.
    assert (context.directory / "system" / "adapter.log").is_file()
    assert (context.directory / "system" / "rpc-events.jsonl").is_file()


def test_refine_is_invoked_between_instances(tmp_path, monkeypatch, upstream):
    _, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"refine_every": 2})
    lines = commands.read_text().splitlines()
    assert lines.count("prompt") > 12  # actions plus refine prompts
    assert "abort" not in lines  # refine_complete was recognised as the end of the turn


def test_home_agent_dir_is_the_agent_directory(tmp_path):
    """prime-agent's documented path for a personal skill (~/.prime/agent/skills/) is the one it loads: ~/.prime/agent
    links to the adapter's agent directory, relatively; a directory already there has its skills moved in and the rest
    kept aside; linking again changes nothing."""
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner

    learner = PrimeAgentLearner(command=[sys.executable, FAKE], sandbox="none")
    learner.workdir = tmp_path / "system"
    before = learner.home_dir / ".prime" / "agent" / "skills" / "arc-lib"
    before.mkdir(parents=True)
    (before / "SKILL.md").write_text("---\nname: arc-lib\ndescription: grid helpers\n---\n")
    learner.agent_dir.mkdir(parents=True)
    learner._link_home_agent_dir()
    link = learner.home_dir / ".prime" / "agent"
    assert link.is_symlink() and not os.path.isabs(os.readlink(link)) and link.resolve() == learner.agent_dir.resolve()
    assert (learner.agent_dir / "skills" / "arc-lib" / "SKILL.md").exists() and (link / "skills" / "arc-lib").is_dir()
    assert [p.name.startswith("agent.before-link.") for p in (learner.home_dir / ".prime").iterdir() if p != link] == [True]
    learner._link_home_agent_dir()
    assert link.is_symlink() and link.resolve() == learner.agent_dir.resolve()


def test_the_finish_protocol_ends_every_instance_with_the_verdict_alone(tmp_path, monkeypatch, upstream):
    """end="finish" (the harness rows since 27 Sep): the rules state facts only, prime-agent's one fact about its store among
    them; every instance ends with the verdict alone, and the agent ends it with finish."""
    context, _ = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"end": "finish"})
    records = records_of(context)
    turns = [json.loads(line) for line in (context.directory / "system" / "turns.jsonl").read_text().splitlines()]
    verdicts = [t for t in turns if t["kind"] == "verdict"]
    assert len(verdicts) == len(records) == sum(r["resources"]["system_finishes"] for r in records)
    assert not any(t["kind"] in ("closing", "storage") for t in turns)
    assert all(t["message"].startswith("Your submission was") and "\n" not in t["message"] for t in verdicts)
    firsts = [t for t in turns if t["kind"] == "observation" and t.get("fresh")]
    assert firsts and all('{"action": "finish"}' in t["message"] and "global_=True" in t["message"]
                          and "worth keeping" not in t["message"] for t in firsts)


def test_the_stateless_row_ends_its_instances_the_same_way(tmp_path, monkeypatch, upstream):
    context, _ = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"end": "finish", "stateless": True})
    records = records_of(context)
    turns = [json.loads(line) for line in (context.directory / "system" / "turns.jsonl").read_text().splitlines()]
    assert sum(t["kind"] == "verdict" for t in turns) == len(records) == sum(r["resources"]["system_finishes"] for r in records)


def test_a_storage_turn_ends_every_instance_with_the_verdict(tmp_path, monkeypatch, upstream):
    """storage_turn (the team rows): every instance ends with a turn of the agent's own in which to save what will help
    later, carrying the verdict and the fact about prime-agent its agent cannot know (entries without global_=True die
    with the session), in place of the closing turn."""
    context, _ = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"storage_turn": True})
    records = records_of(context)
    turns = [json.loads(line) for line in (context.directory / "system" / "turns.jsonl").read_text().splitlines()]
    storage = [t for t in turns if t["kind"] == "storage"]
    assert len(storage) == len(records) == sum(r["resources"]["system_storage_turns"] for r in records)
    assert not any(t["kind"] == "closing" for t in turns)
    assert all(t["message"].startswith("Your submission was") and "only what you save outlasts it" in t["message"]
               and "global_=True" in t["message"] and t["message"].endswith("reply with the single word: done.")
               for t in storage)
    # the documented skills path is the loaded one in a real run too
    state = context.directory / "system" / "state"
    assert (state / "home" / ".prime" / "agent").resolve() == (state / "agent").resolve()


def test_refine_via_rpc_is_the_refinement_call_alone_with_the_verdict(tmp_path, monkeypatch, upstream):
    """refine_via rpc (the team rows): after every instance the adapter starts the global refinement itself, with the
    RPC refine command and the instance's verdict as its instructions. No closing turn and no turn asking the model to
    refine: the refinement's is the only model call after the instance's last action."""
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"refine_every": 1, "refine_via": "rpc"})
    records = records_of(context)
    lines = commands.read_text().splitlines()
    turns = [json.loads(line) for line in (context.directory / "system" / "turns.jsonl").read_text().splitlines()]
    assert lines.count("refine") == len(records) and "abort" not in lines
    assert lines.count("prompt") == sum(1 for t in turns if t["kind"] in ("observation", "reprompt"))
    assert not any(t["kind"] == "closing" for t in turns)
    refines = [t for t in turns if t["kind"] == "refine"]
    assert len(refines) == len(records) == sum(r["resources"]["system_refines"] for r in records)
    assert all(t["message"].startswith("Your submission was") and "This piece of work is finished" in t["message"]
               and t["reply"] == "fake global refine" for t in refines)
    # one model call per refinement, through the proxy, with the verdict in it
    asked = [r["body"]["messages"][-1]["content"] for r in upstream.requests if r["body"] and r["body"].get("messages")]
    assert sum(1 for m in asked if isinstance(m, str) and m.startswith("REFINE\nYour submission was")) == len(records)


def test_provider_error_retry_is_not_an_empty_reply(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("FAKE_AGENT_ERROR_FIRST", "1")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams")
    records = records_of(context)
    assert sum(r["resources"]["system_parse_failures"] for r in records) == 0
    assert sum(r["resources"]["system_restarts"] for r in records) == 0
    assert "waiting for prime-agent's retry" in adapter_log(context)
    events = [json.loads(line) for line in (context.directory / "system" / "rpc-events.jsonl").read_text().splitlines()]
    assert {"response", "agent_end", "agent_start"} <= {e["type"] for e in events}
    assert all(len(m["text"]) <= 2000 for e in events if e["type"] == "agent_end" for m in e["messages"])


def test_lost_closing_events_are_recovered_over_the_command_path(tmp_path, monkeypatch, upstream):
    """prime-agent 0.9.5's daemon relay drops the tail of a turn (turn_end, agent_end)
    while get_state still answers: the adapter notices the quiet stream, asks the worker,
    and reads the reply, in seconds instead of an idle timeout."""
    monkeypatch.setenv("FAKE_AGENT_LOSE_TAIL", "1")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                                 {"quiet_probe_s": 0.3, "idle_timeout_s": 20, "final_turn_grace_s": 20})
    records = records_of(context)
    assert sum(r["resources"]["system_stalls"] for r in records) == 1
    assert sum(r["resources"]["system_restarts"] for r in records) == 0
    assert sum(r["resources"]["system_idle_timeouts"] for r in records) == 0
    assert all(r["solved"] for r in records)
    assert "recovered the reply over the command path" in adapter_log(context)
    lines = commands.read_text().splitlines()
    # The worker was asked over the command path; the reply itself came from the final
    # message_end already observed, so no abort and no restart were needed.
    assert lines.count("get_state") >= 2 and "abort" not in lines


def test_repeated_event_loss_reattaches_the_client(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("FAKE_AGENT_LOSE_TAIL", "2")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                                 {"quiet_probe_s": 0.3, "idle_timeout_s": 20})
    records = records_of(context)
    assert sum(r["resources"]["system_stalls"] for r in records) == 2
    assert sum(r["resources"]["system_restarts"] for r in records) == 1
    assert all(r["solved"] for r in records)
    assert "event attachment is dead" in adapter_log(context)
    assert commands.read_text().count("get_state") >= 3  # start, probes, restart


def test_provider_retry_delay_is_honoured(tmp_path, monkeypatch, upstream):
    """prime-agent's auto-retry backoff grows to minutes; the adapter waits for the retry it
    announces instead of declaring the turn empty after a fixed 15 s."""
    monkeypatch.setenv("FAKE_AGENT_RATE_LIMITED", "2")
    context, _ = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"quiet_probe_s": 0.5})
    records = records_of(context)
    assert all(r["solved"] for r in records)
    assert sum(r["resources"]["system_parse_failures"] for r in records) == 0
    assert sum(r["resources"]["system_provider_retries"] for r in records) == 0
    assert sum(r["resources"]["system_stalls"] for r in records) == 0
    assert adapter_log(context).count("is retrying the model call") == 2


def test_provider_outage_resends_the_prompt_instead_of_giving_up(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("FAKE_AGENT_PROVIDER_DOWN", "2")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                                 {"quiet_probe_s": 0.3, "provider_backoff_s": 0.2})
    records = records_of(context)
    assert all(r["solved"] for r in records)
    assert not any(r["gave_up"] for r in records)
    assert sum(r["resources"]["system_provider_retries"] for r in records) == 2
    # The previous turn's text was never mistaken for this turn's reply.
    assert sum(r["resources"]["system_stalls"] for r in records) == 0
    assert "re-sending the prompt" in adapter_log(context)


def test_final_reply_without_agent_end_is_taken_after_grace(tmp_path, monkeypatch, upstream):
    """The fallback when probing is disabled: a turn that ended on a final reply is taken
    after a short grace instead of waiting for an agent_end that never comes."""
    monkeypatch.setenv("FAKE_AGENT_STALL_ONCE", "1")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                                 {"quiet_probe_s": 0, "final_turn_grace_s": 0.5, "idle_timeout_s": 20})
    records = records_of(context)
    assert sum(r["resources"]["system_stalls"] for r in records) == 1
    assert sum(r["resources"]["system_restarts"] for r in records) == 0
    assert all(r["solved"] for r in records)
    assert "taking the reply" in adapter_log(context)
    assert "abort" in commands.read_text().splitlines()


def test_a_prompt_refused_while_prime_agent_compacts_waits_and_is_sent_again(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("FAKE_AGENT_COMPACT_ONCE", "1")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams")
    records = records_of(context)
    assert all(r["solved"] for r in records)
    assert sum(r["resources"]["system_restarts"] for r in records) == 0  # not taken for a dead system
    assert "busy (compacting)" in adapter_log(context)


def test_a_run_stopped_to_compact_is_waited_for_not_taken_for_an_empty_reply(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("FAKE_AGENT_COMPACT_MIDRUN_ONCE", "1")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams")
    records = records_of(context)
    assert all(r["solved"] for r in records)
    assert sum(r["resources"]["system_restarts"] for r in records) == 0
    assert sum(r["resources"]["system_parse_failures"] for r in records) == 0
    assert "stopped after a tool result to compact" in adapter_log(context)


def test_silent_system_is_restarted_after_idle_timeout(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("FAKE_AGENT_HANG_ONCE", "1")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                                 {"idle_timeout_s": 1.5, "timeout_s": 60})
    records = records_of(context)
    assert sum(r["resources"]["system_idle_timeouts"] for r in records) == 1
    assert sum(r["resources"]["system_restarts"] for r in records) == 1
    assert all(r["solved"] for r in records)
    assert commands.read_text().count("get_state") == 2  # the original start and the restart
    assert "no RPC event and no proxy traffic" in adapter_log(context)


def test_action_budget_stops_an_endless_loop(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("FAKE_AGENT_LOOP_ONCE", "1")
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                                 {"max_model_calls": 5, "idle_timeout_s": 20})
    records = records_of(context)
    assert sum(r["resources"]["system_budget_hits"] for r in records) == 1
    assert sum(r["resources"]["system_reprompts"] for r in records) == 1
    assert sum(r["resources"]["system_restarts"] for r in records) == 0
    assert all(r["solved"] for r in records)
    assert "action budget exceeded" in adapter_log(context)
    # The aborted run's text had no action, so the system was asked once to decide.
    body = " ".join(json.dumps(r["body"]) for r in upstream.requests)
    assert "exceeded the time or model-call budget" in body


def test_dead_system_costs_actions_not_the_run(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("FAKE_AGENT_DIE", "1")
    context, _ = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"max_restarts": 1}, n_slots=3)
    records = records_of(context)
    assert len(records) == 3 and all(r["gave_up"] for r in records)
    assert all(r["resources"]["system_forced_give_ups"] == 1 for r in records)
    assert all(r["resources"]["system_restarts"] == 1 for r in records)
    assert "this action is a give-up" in adapter_log(context)


def test_sandbox_none_must_be_explicit(tmp_path, monkeypatch):
    from continual_arc_baselines.systems.sandbox import SandboxUnavailable, resolve_backend

    monkeypatch.setattr("continual_arc_baselines.systems.sandbox.platform.system", lambda: "Linux")
    with pytest.raises(SandboxUnavailable):
        resolve_backend("auto")
    assert resolve_backend("none") == "none"


@pytest.mark.skipif(platform.system() != "Darwin", reason="seatbelt is macOS only")
def test_seatbelt_confines_the_system(tmp_path, monkeypatch, upstream):
    """Inside the sandbox the system cannot read the repository that launched it."""
    monkeypatch.setenv("FAKE_AGENT_PROBE", str(REPO / "pyproject.toml"))
    assert (REPO / "pyproject.toml").is_file()
    context, commands = run_mode(
        tmp_path, monkeypatch, upstream, "workstreams",
        {"sandbox": "seatbelt", "sandbox_read_paths": [sys.prefix, str(Path(FAKE).parent)]},
        n_slots=4,
    )
    lines = commands.read_text().splitlines()
    assert "probe:denied" in lines and "probe:ok" not in lines
    assert all(r["solved"] for r in records_of(context))  # the proxy was still reachable
    profile = (context.directory / "system" / "sandbox.sb").read_text()
    assert "(deny default)" in profile and "remote ip \"localhost:" in profile


def test_integrations_are_off(tmp_path):
    from continual_arc_baselines.systems.prime_agent import INTEGRATION_SKILLS, PrimeAgentLearner
    s = PrimeAgentLearner(model="m", sandbox="none", command=sys.executable)
    s.attach(tmp_path / "system", {})
    s._write_config()
    settings = json.loads((s.agent_dir / "settings.json").read_text())
    assert settings["bundledSkills"] == {"websearch": False}
    assert set(settings["skills"]) == {f"-{n}/SKILL.md" for n in INTEGRATION_SKILLS}
    assert "compaction" not in settings  # prime-agent's own threshold


def test_compaction_can_come_earlier(tmp_path):
    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner
    s = PrimeAgentLearner(model="m", sandbox="none", command=sys.executable, context_window=131_072, compact_at=49_152)
    s.attach(tmp_path / "system", {})
    s._write_config()
    # prime-agent compacts once a finished turn's context passes contextWindow - reserveTokens
    assert json.loads((s.agent_dir / "settings.json").read_text())["compaction"] == {"reserveTokens": 81_920}
    s = PrimeAgentLearner(model="m", sandbox="none", command=sys.executable, context_window=131_072, compact_at=32_768,
                          compact_keep=8_000)
    s.attach(tmp_path / "system2", {})
    s._write_config()
    assert json.loads((s.agent_dir / "settings.json").read_text())["compaction"] == {"reserveTokens": 98_304,
                                                                                     "keepRecentTokens": 8_000}
    with pytest.raises(ValueError):
        PrimeAgentLearner(model="m", context_window=131_072, compact_at=200_000)


def test_budget_abort_with_nothing_said_is_reprompted_not_restarted(tmp_path, monkeypatch, upstream):
    """A turn the adapter cut for budget can end with no text at all. That is not a dead
    system: the follow-up carries the budget notice and the instance goes on."""
    monkeypatch.setenv("FAKE_AGENT_LOOP_SILENT", "1")
    context, _ = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"action_budget_s": 2, "reprompt_budget_s": 30})
    records = records_of(context)
    assert sum(r["resources"]["system_budget_hits"] for r in records) >= 1
    assert sum(r["resources"]["system_restarts"] for r in records) == 0
    assert "asking for a decision" in adapter_log(context)


def test_stateless_row_wipes_and_restarts_before_every_instance(tmp_path, monkeypatch, upstream):
    """Row H0: the harness is stopped, its state deleted and started again before each instance."""
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams",
                                 {"stateless": True, "refine_every": 1, "storage_turn": True})
    records = records_of(context)
    assert len(records) > 2 and all(r["solved"] for r in records)
    # no storage turn either: the closing carries the verdict, as nothing saved could last
    assert sum(r["resources"]["system_storage_turns"] for r in records) == 0
    assert sum(r["resources"]["system_wipes"] for r in records) == len(records) - 1
    assert sum(r["resources"]["system_resets"] for r in records) == len(records) - 1
    assert sum(r["resources"]["system_refines"] for r in records) == 0   # nothing could survive it
    assert commands.read_text().count("get_state") == len(records)        # one start per instance
    assert commands.read_text().count("new_session") == 0
    # the provider configuration survives every wipe; the stores are rebuilt by each start
    state = context.directory / "system" / "state" / "agent"
    assert (state / "models.json").is_file()
    params = next(json.loads(line)["payload"]["values"] for line in (context.directory / "events.jsonl").read_text().splitlines()
                  if json.loads(line)["kind"] == "parameters")
    assert params["system"]["stateless"] is True


def test_turn_log_records_every_exchange(tmp_path, monkeypatch, upstream):
    """turns.jsonl beside the run: one line per message the adapter sent, with the reply, the kind
    (observation, reprompt, closing, refine) and the instance it belonged to."""
    context, commands = run_mode(tmp_path, monkeypatch, upstream, "workstreams", {"refine_every": 1})
    records = records_of(context)
    turns = [json.loads(line) for line in (context.directory / "system" / "turns.jsonl").read_text().splitlines()]
    kinds = {t["kind"] for t in turns}
    assert {"observation", "closing", "refine"} <= kinds
    assert sum(1 for t in turns if t["kind"] == "closing") == len(records)
    assert sum(1 for t in turns if t["kind"] == "refine") == len(records)
    assert sum(1 for t in turns if t["kind"] == "observation") == sum(r["resources"]["actions"] for r in records)
    first = next(t for t in turns if t["kind"] == "observation")
    assert first["message"].startswith("You are being evaluated") and first["reply"] and "task" in first
    assert all("t" in t and "reply" in t and "message" in t for t in turns)


def test_the_turn_budget_leaves_out_what_the_proxy_credits(tmp_path, monkeypatch):
    """A row whose proxy answers some calls itself spends part of each call outside the model: its
    proxy credits those seconds to the tracker (Tracker.credit_s), and the turn's time budget and hard timeout move out
    by what was credited during the turn, so the row is held to the same budget of model time as any other. A turn the
    model keeps working at is cut after its budget plus the credit, not after its budget alone; with no credit, as for
    every other row, after the budget."""
    import time as _time

    from continual_arc_baselines.systems.prime_agent import PrimeAgentLearner
    from continual_arc_baselines.tracking import Tracker

    class Crediting(Tracker):
        """Credits every second of the turn's first ``upto`` seconds, as a proxy busy in the pipeline would."""

        def __init__(self, upto: float) -> None:
            super().__init__()
            self.t0, self.upto = _time.monotonic(), upto

        @property
        def credit_s(self) -> float:
            return min(_time.monotonic() - self.t0, self.upto)

        @credit_s.setter
        def credit_s(self, value: float) -> None:
            pass

    def cut_after(tracker) -> tuple[str, float, PrimeAgentLearner]:
        s = PrimeAgentLearner(model="m", action_budget_s=1.0, timeout_s=60, idle_timeout_s=60, quiet_probe_s=0)
        s.tracker = tracker
        monkeypatch.setattr(s, "_send", lambda command: None)
        monkeypatch.setattr(s, "_wait_response", lambda command, timeout: {})
        monkeypatch.setattr(s, "_abort", lambda collect_s=0.0: "")
        monkeypatch.setattr(s, "_note", lambda message: None)
        s._events.put({"type": "message_start", "message": {"role": "assistant"}})   # the model at work, then quiet
        t0 = _time.monotonic()
        outcome, _ = s._turn("go", budget_s=1.0, max_calls=40, until=("agent_end",))
        return outcome, _time.monotonic() - t0, s

    outcome, seconds, s = cut_after(Tracker())
    assert outcome == "budget" and 0.9 <= seconds < 2.5 and s.stats["budget_hits"] == 1
    outcome, seconds, s = cut_after(Crediting(upto=2.0))
    assert outcome == "budget" and 2.9 <= seconds < 5.0 and s.stats["budget_hits"] == 1   # 1 s of model time plus 2 credited
    t = Tracker()
    t.credit(1.5)
    t.credit(-3.0)   # nothing negative is taken
    assert t.credit_s == 1.5
