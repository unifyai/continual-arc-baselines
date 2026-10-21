"""A harness's frozen tests after its stream (arc-baselines test): one instance at a time, every instance from a copy
of the state the stream left, at the stream's own path, the stream's own state set aside meanwhile and given back.
arc1-smoke's real schedules, played over FakeSource tasks, so that none of the pinned ARC data is needed, which CI does
not fetch."""

import hashlib
import json
import os
import stat
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
from continual_arc.configs import build_schedule, get_config
from continual_arc.configs import test_schedule as make_test  # not collected as a test
from continual_arc.env import DemosFeedback, RequestDemos, Submit
from continual_arc.schedule import Schedule
from continual_arc.tasks import TaskRegistry
from continual_arc.testing import FakeSource

from continual_arc_baselines.cli import main
from continual_arc_baselines.frozen import hold_aside, holding, put_back
from continual_arc_baselines.systems.base import SystemLearner
from continual_arc_baselines.tracking import instance_ended

from .test_acp import FakeSystem, obs

CONFIG = "arc1-smoke"
IDS = FakeSource(n_tasks=24).task_ids()
STREAM_POOL, HELD_OUT_POOL = IDS[:12], IDS[12:]
INSTANCES = {"held-out": 8, "seen": 32}   # arc1-smoke's 8 held-out tasks once, and 4 of each of its 8 stream tasks


def fake_arc1() -> TaskRegistry:
    """arc1-smoke's two sources, each serving FakeSource tasks."""
    cfg = get_config(CONFIG)
    return TaskRegistry({s: FakeSource(name=s, n_tasks=len(IDS)) for s in (cfg.stream_source, cfg.epilogue_source)})


def listing(root: Path) -> dict:
    """Every entry under ``root``: a file's mode, modification time and content, a link's target, a directory's mode;
    a socket or a pipe by its kind alone."""
    out = {}
    for folder, dirs, files in os.walk(root):
        for name in dirs + files:
            path = Path(folder) / name
            st = path.lstat()
            mode, kind = stat.S_IMODE(st.st_mode), stat.S_IFMT(st.st_mode)
            if kind == stat.S_IFLNK:
                out[str(path.relative_to(root))] = ("link", os.readlink(path))
            elif kind == stat.S_IFDIR:
                out[str(path.relative_to(root))] = ("dir", mode)
            elif kind == stat.S_IFREG:
                out[str(path.relative_to(root))] = ("file", mode, st.st_mtime_ns, path.read_bytes())
            else:
                out[str(path.relative_to(root))] = ("special",)
    return out


def digest(path: Path) -> str:
    return hashlib.sha256(repr(sorted(listing(path).items())).encode()).hexdigest()


class FileMemoryHarness(SystemLearner):
    """ShiftMemoryLearner kept as a harness keeps what it learns, in files under its state directory: each task's
    shift, written down once a demonstration shows it, and a note of every instance it played, named by its step, so
    that a test instance's note would change what the next one finds. As an instance begins it logs, outside its state,
    the hash of the state it finds (``starts.jsonl`` in its workdir)."""

    name = "file-memory"

    def start(self):
        pass

    def stop(self):
        pass

    def new_session(self):
        pass

    def wipe_state(self):
        (self.state_dir / "memory.json").unlink(missing_ok=True)
        for note in self.state_dir.glob("notes/*"):
            note.unlink()

    def memory(self) -> dict:
        path = self.state_dir / "memory.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def act(self, obs):
        if obs.is_fresh:
            with (self.workdir / "starts.jsonl").open("a") as log:
                log.write(json.dumps({"step": obs.step, "state": digest(self.state_dir)}) + "\n")
        shift = self.memory().get(obs.task_id)
        if shift is None:
            return RequestDemos()
        return Submit([[(c + shift) % 10 for c in row] for row in obs.input])

    def observe(self, obs, action, feedback):
        if isinstance(feedback, DemosFeedback) and feedback.pairs:
            pair = feedback.pairs[0]
            self.state_dir.mkdir(parents=True, exist_ok=True)
            (self.state_dir / "memory.json").write_text(
                json.dumps({**self.memory(), obs.task_id: (pair.output[0][0] - pair.input[0][0]) % 10}))
        if instance_ended(feedback):
            (self.state_dir / "notes").mkdir(parents=True, exist_ok=True)
            (self.state_dir / "notes" / f"{obs.step:04d}").write_text(obs.task_id)


def run_stream(out: Path, *extra: str, learner: str = "tests.test_frozen:FileMemoryHarness") -> Path:
    assert main(["run", "--learner", learner, "--no-proxy", "--registry", "tests.test_frozen:fake_arc1", "--quiet",
                 "--out", str(out), *extra]) == 0
    return out


def system_of(run: Path) -> Path:
    return Path(json.loads((run / "attempt.json").read_text())["attempt_dir"]) / "system"


def starts(play: Path) -> list[dict]:
    attempt = Path(json.loads((play / "attempt.json").read_text())["attempt_dir"])
    return [json.loads(line) for line in (attempt / "system" / "starts.jsonl").read_text().splitlines()]


def records(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "records.jsonl").read_text().splitlines()]


@pytest.fixture
def frozen_tests(tmp_path, monkeypatch) -> dict[str, Path]:
    """The two tests of arc1-smoke at seed 0, one instance at a time, with build_schedule drawing on the fake pools."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("continual_arc.configs.build_schedule",
                        partial(build_schedule, rearc_task_ids=STREAM_POOL, eval_task_ids=HELD_OUT_POOL))
    paths = {}
    for test in INSTANCES:
        paths[test] = tmp_path / f"{test}-1.json"
        make_test(CONFIG, 0, test, width=1, rearc_task_ids=STREAM_POOL, eval_task_ids=HELD_OUT_POOL).save(
            paths[test])
    return paths


def test_both_tests_play_every_instance_from_the_state_the_stream_left(frozen_tests, tmp_path):
    run = run_stream(tmp_path / "runs" / "harness-ws0", "--config", CONFIG, "--no-epilogue")
    assert {r["phase"] for r in records(run)} == {"stream"} and len(records(run)) == 120
    system = system_of(run)
    end = digest(system / "state")
    memory = json.loads((system / "state" / "memory.json").read_text())
    assert len(memory) == 8   # every stream task, written down under the name the run gave it
    for test, n in INSTANCES.items():
        assert main(["test", str(run), "--schedule", str(frozen_tests[test]), "--quiet"]) == 0
        play = run / "frozen" / test / "play"
        assert [r["phase"] for r in records(play)] == [f"{test}-test"] * n
        assert (play / "alias-salt").read_text() == (run / "alias-salt").read_text()
        assert json.loads((play / "frozen.json").read_text()) == {"stream": str(run.resolve()), "instances": 120}
        # every instance found the state the stream left, though each wrote to it, and the stream's own is back
        assert [s["state"] for s in starts(play)] == [end] * n
        assert digest(system / "state") == end and not list(system.glob("stream-end*"))
    # the seen test is solved from the stream's memory under the stream's names, at no cost; the held-out tasks, which
    # the stream never served, each need a demonstration
    assert all(r["solved"] and r["cost"] == 0 for r in records(run / "frozen" / "seen" / "play"))
    assert all(r["solved"] and r["cost"] == 1 for r in records(run / "frozen" / "held-out" / "play"))


def test_a_test_cut_short_goes_on_and_one_played_whole_is_not_played_again(frozen_tests, tmp_path):
    run = run_stream(tmp_path / "run", "--config", CONFIG, "--no-epilogue")
    end = digest(system_of(run) / "state")
    command = ["test", str(run), "--schedule", str(frozen_tests["held-out"]), "--quiet"]
    assert main(command) == 0
    play = run / "frozen" / "held-out" / "play"
    lines = (play / "records.jsonl").read_text().splitlines()
    (play / "records.jsonl").write_text("\n".join(lines[:3]) + "\n")
    assert main(command) == 0
    assert (play / "records.jsonl").read_text().splitlines()[:3] == lines[:3]
    assert [(r["step"], r["solved"], r["cost"]) for r in records(play)] == [
        (r["step"], r["solved"], r["cost"]) for r in map(json.loads, lines)]
    assert json.loads((play / "meta.json").read_text())["resumed_at"][0]["records"] == 3
    assert [s["state"] for s in starts(play)] == [end] * (8 + 5)   # one attempt, the 5 left played again
    assert main(command) == 0
    assert len(starts(play)) == 13


def test_a_test_that_fails_gives_the_streams_state_back_and_goes_on_when_launched_again(frozen_tests, tmp_path,
                                                                                       monkeypatch):
    run = run_stream(tmp_path / "run", "--config", CONFIG, "--no-epilogue")
    system = system_of(run)
    end = digest(system / "state")
    act = FileMemoryHarness.act

    def dies_at_the_fourth(self, obs):
        if obs.is_fresh and obs.step == 3:
            raise RuntimeError("the harness died")
        return act(self, obs)

    monkeypatch.setattr(FileMemoryHarness, "act", dies_at_the_fourth)
    command = ["test", str(run), "--schedule", str(frozen_tests["seen"]), "--quiet"]
    with pytest.raises(RuntimeError, match="the harness died"):
        main(command)
    assert digest(system / "state") == end and not list(system.glob("stream-end*"))
    monkeypatch.undo()
    assert main(command) == 0
    play = run / "frozen" / "seen" / "play"
    assert len(records(play)) == 32 and [s["state"] for s in starts(play)] == [end] * (3 + 29)


def test_a_test_stopped_hard_leaves_the_streams_state_aside_and_the_next_play_takes_it_back(frozen_tests, tmp_path):
    run = run_stream(tmp_path / "run", "--config", CONFIG, "--no-epilogue")
    system = system_of(run)
    end = digest(system / "state")

    def killed_mid_test():
        """What a test killed during an instance leaves: the stream's state aside, a copy written to in its place."""
        put_back(hold_aside(system, 120), system / "state")
        (system / "state" / "memory.json").write_text("{}")

    killed_mid_test()
    assert main(["test", str(run), "--schedule", str(frozen_tests["held-out"]), "--quiet"]) == 0
    assert [s["state"] for s in starts(run / "frozen" / "held-out" / "play")] == [end] * 8
    assert digest(system / "state") == end and not list(system.glob("stream-end*"))
    killed_mid_test()
    run_stream(run, "--config", CONFIG, "--no-epilogue", "--resume")   # the stream taken up again, with nothing to play
    assert digest(system / "state") == end and not list(system.glob("stream-end*"))
    hold_aside(system, 60)   # set aside at another point of the stream than where it stands
    with pytest.raises(SystemExit, match="has played 120 instances since"):
        run_stream(run, "--config", CONFIG, "--no-epilogue", "--resume")


def test_a_test_of_a_stream_that_went_on_is_refused_and_the_stream_plays_on_from_its_own_state(frozen_tests, tmp_path):
    full = build_schedule(CONFIG, "workstreams", 0, rearc_task_ids=STREAM_POOL, eval_task_ids=HELD_OUT_POOL,
                          epilogue=False)
    full.save(tmp_path / "full.json")
    full.first_rounds(60).save(tmp_path / "cut.json")
    run = run_stream(tmp_path / "run", "--schedule", str(tmp_path / "cut.json"))
    state = system_of(run) / "state"
    at_60 = digest(state)
    command = ["test", str(run), "--schedule", str(frozen_tests["seen"]), "--quiet"]
    assert main(command) == 0
    assert digest(state) == at_60
    run_stream(run, "--schedule", str(tmp_path / "full.json"), "--resume")
    assert len(records(run)) == 120
    with pytest.raises(SystemExit, match="another stream end"):
        main(command)
    at_120 = digest(state)
    assert main([*command, "--out", str(tmp_path / "seen-at-120")]) == 0
    assert [s["state"] for s in starts(tmp_path / "seen-at-120")] == [at_120] * 32


def test_a_stateless_row_plays_every_test_instance_wiped_and_leaves_the_stream_as_it_was(frozen_tests, tmp_path):
    run = run_stream(tmp_path / "run", "--config", CONFIG, "--no-epilogue", "--learner-kwargs", '{"stateless": true}')
    state = system_of(run) / "state"
    end = digest(state)
    assert main(["test", str(run), "--schedule", str(frozen_tests["seen"]), "--quiet"]) == 0
    play = run / "frozen" / "seen" / "play"
    assert len({s["state"] for s in starts(play)}) == 1 and starts(play)[0]["state"] != end
    assert all(r["solved"] and r["cost"] == 1 for r in records(play))   # nothing of the stream's memory
    assert digest(state) == end


def test_what_a_harness_cannot_play_is_refused_before_anything_plays(frozen_tests, tmp_path):
    run = run_stream(tmp_path / "run", "--config", CONFIG, "--no-epilogue")
    wide = tmp_path / "held-out-8.json"
    make_test(CONFIG, 0, "held-out", rearc_task_ids=STREAM_POOL, eval_task_ids=HELD_OUT_POOL).save(wide)
    unreset = Schedule.load(frozen_tests["held-out"])
    slots = list(unreset.slots)
    slots[3] = replace(slots[3], reset_before=False)
    replace(unreset, slots=slots).save(tmp_path / "held-out-unreset.json")
    other_seed = tmp_path / "held-out-seed1.json"
    make_test(CONFIG, 1, "held-out", width=1, rearc_task_ids=STREAM_POOL, eval_task_ids=HELD_OUT_POOL).save(
        other_seed)
    for schedule, refusal in ((wide, "one instance at a time"), (tmp_path / "held-out-unreset.json",
                              "one instance at a time"), (run / "schedule.json", "not a frozen test"),
                              (other_seed, "at seed 1")):
        with pytest.raises(SystemExit, match=refusal):
            main(["test", str(run), "--schedule", str(schedule)])
    launches = (run / "launches.jsonl").read_text()
    with holding(run):   # another process playing the run: its stream, or a test of it
        with pytest.raises(SystemExit, match="another process is playing"):
            main(["test", str(run), "--schedule", str(frozen_tests["seen"])])
        with pytest.raises(SystemExit, match="another process is playing"):
            run_stream(run, "--config", CONFIG, "--no-epilogue", "--resume")
    assert not (run / "frozen").exists() and not list(system_of(run).glob("stream-end*"))
    assert (run / "launches.jsonl").read_text() == launches

    with_epilogue = run_stream(tmp_path / "with-epilogue", "--config", CONFIG)
    held = run_stream(tmp_path / "held", "--config", CONFIG, "--no-epilogue", "--stop-after", "50")
    in_memory = run_stream(tmp_path / "in-memory", "--config", CONFIG, "--no-epilogue",
                           learner="continual_arc.testing:ShiftMemoryLearner")
    for other, refusal in ((with_epilogue, "and then epilogue"), (held, "has played 50 of its stream's 120"),
                           (in_memory, "keeps no state as files")):
        with pytest.raises(SystemExit, match=refusal):
            main(["test", str(other), "--schedule", str(frozen_tests["held-out"])])
        assert not (other / "frozen").exists()
    with pytest.raises(SystemExit, match="--no-epilogue builds the schedule from --config"):
        run_stream(tmp_path / "x", "--schedule", str(frozen_tests["seen"]), "--no-epilogue")


def test_a_frozen_reset_puts_the_saved_state_back_and_a_stateless_one_still_wipes(tmp_path):
    saved = tmp_path / "saved"
    (saved / "notes").mkdir(parents=True)
    (saved / "memory.json").write_text('{"task-1": 3}')
    for stateless, memory in ((False, {"task-1": 3}), (True, {})):
        harness = FileMemoryHarness(stateless=stateless)
        harness.attach(tmp_path / f"work-{stateless}", {}, state_dir=tmp_path / f"state-{stateless}")
        harness.freeze(saved)
        (harness.state_dir / "memory.json").write_text('{"task-2": 5}')
        harness.reset()
        assert harness.memory() == memory
        assert (harness.state_dir / "notes").is_dir()
        assert not (harness.workdir / "state").exists()


def test_an_adapter_frozen_plays_from_the_saved_state_where_it_lies_and_keeps_its_logs_apart(tmp_path):
    saved, state, work = tmp_path / "stream" / "stream-end-5", tmp_path / "stream" / "state", tmp_path / "test"
    (saved / "agent").mkdir(parents=True)
    (saved / "config.json").write_text("{}")
    (saved / "agent" / "kept.txt").write_text("what the stream left")
    system = FakeSystem(model="fake-model", sandbox="none", startup_timeout_s=60, timeout_s=60)
    system.attach(work, {}, state_dir=state)
    try:
        system.freeze(saved)
        assert system.child_env()["HOME"] == str(state / "home") and state in system.sandbox_spec().write_paths
        system.act(obs())
        assert (state / "agent" / "prompt-1.txt").exists() and (state / "agent" / "kept.txt").exists()
        system.reset()
        assert listing(state) == listing(saved)
        assert (work / "fake.log").exists() and not (work / "state").exists()
    finally:
        system.close()


def test_put_back_makes_a_whole_copy_whatever_an_instance_changed(tmp_path):
    saved, state = tmp_path / "saved", tmp_path / "state"
    (saved / "Skills").mkdir(parents=True)
    (saved / "Skills" / "a.py").write_text("print(1)")
    (saved / "Notes.md").write_text("AAAA")
    (saved / "run.sh").write_text("#!/bin/sh\n")
    (saved / "run.sh").chmod(0o755)
    (saved / "link").symlink_to("Skills")
    (saved / "go" / "mod").mkdir(parents=True)
    (saved / "go" / "mod" / "x.go").write_text("package x")
    (saved / "go" / "mod").chmod(0o555)   # a Go module cache's directories are read-only
    os.mkfifo(saved / "pipe")
    copy = {k: v for k, v in listing(saved).items() if v != ("special",)}
    put_back(saved, state)
    assert listing(state) == copy
    (state / "Notes.md").rename(state / "notes.md")   # a name changed in case alone
    (state / "Skills").rename(state / "skills")
    times = (state / "run.sh").stat()
    (state / "run.sh").write_text("#!/bin/X\n")       # the same size, its times put back as tar or cp -p would
    os.utime(state / "run.sh", ns=(times.st_atime_ns, times.st_mtime_ns))
    (state / "run.sh").chmod(0o700)
    (state / "go" / "mod").chmod(0o755)
    (state / "go" / "mod" / "y.go").write_text("package y")
    (state / "go" / "mod").chmod(0o555)
    (state / "link").unlink()
    (state / "link").mkdir()
    os.mkfifo(state / "fifo")
    (state / "hidden").mkdir()
    (state / "hidden" / "notes.md").write_text("kept from the listing")
    (state / "hidden").chmod(0o300)                   # a directory that cannot be listed
    put_back(saved, state)
    assert listing(state) == copy
    for root in (saved, state):   # so that the test's folder can be removed
        (root / "go" / "mod").chmod(0o755)
