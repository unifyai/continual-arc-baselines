"""A harness's two frozen tests, played from the state its stream left.

After the stream a learner plays two frozen tests (continual-arc's ``configs.test_schedule``), the held-out test and the
seen test, every instance of each starting from the learner as the stream left it. A harness
(:class:`~continual_arc_baselines.systems.base.SystemLearner`) keeps one conversation, and everything it keeps between
instances as files under its state directory, so it plays a test one instance at a time (``continual-arc schedule make
--test ... --width 1``, every instance marked ``reset_before``), and before each instance its reset puts that directory
back as the stream left it instead of wiping it (``SystemLearner.freeze``). The state goes back where it was, at the
same path, so every absolute path the harness wrote into its own files (its workspace, its home, its sessions' folders,
the rows of its databases) still names that state, and no file is rewritten, whatever the harness.

``arc-baselines test RUN --schedule TEST.json [--out OUT]`` plays one test after the run in ``RUN``, the run directory
``arc-baselines run --out`` wrote for a stream played whole and nothing after it (``--no-epilogue``), with the harness
the stream's first launch built (``launches.jsonl``: the learner and its settings, the upstream, its key and the request
defaults). For the test the stream's own state directory is set aside whole, renamed ``system/stream-end-<instances
played>`` beside it, and a copy of it is put in its place before every instance (:func:`put_back`). When the test ends,
or is stopped with Ctrl-C, ``kill`` or a hangup, the copy is deleted and the stream's directory renamed back. A test
stopped harder leaves it aside, and the next test, or the stream taken up again with ``arc-baselines run --resume``,
puts it back first (:func:`settle`). One process at a time plays a run, its stream or a test of it (:func:`holding`),
so a run's two tests play one after the other.

The test's records go to a run directory of its own, by default ``RUN/frozen/<the test>/play``, holding the run's
``alias-salt`` so the harness meets its tasks under the names it learnt them by, and ``frozen.json``, which stream end
the test is played from; its logs and model-call journal go to an attempt of its own. A test launched again goes on from
where it stopped, and one played whole is not played again, unless the stream has gone on since, which is refused.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

HELD = "stream-end"     # the stream's own state directory while a test plays from copies of it, beside it
PLAYED_FROM = "frozen.json"


def put_back(saved: Path, state: Path) -> None:
    """Make ``state`` a copy of ``saved``, whole: deleted and copied again, links as links. A copy whole takes a second
    or two for a harness's few hundred megabytes, and takes in any change an instance made, one that kept a file's size
    and modification time or changed only the case of its name included. Anything that is neither a file, a directory
    nor a link, such as the socket of a process that has stopped, is not copied."""
    if state.exists():
        delete(state)
    shutil.copytree(saved, state, symlinks=True, ignore=_special)


def _special(folder: str, names: list[str]) -> list[str]:
    def kind(name: str) -> int:
        return stat.S_IFMT(os.lstat(os.path.join(folder, name)).st_mode)

    return [name for name in names if kind(name) not in (stat.S_IFREG, stat.S_IFDIR, stat.S_IFLNK)]


def delete(path: Path) -> None:
    """``shutil.rmtree``, through directories made read-only (a Go module cache's) or unreadable, which refuse to lose
    an entry, or to be listed, until they are made writable and readable again."""
    def unlocked(function, failed, error):
        if not isinstance(error, PermissionError):
            raise error
        for folder in (os.path.dirname(failed), failed):
            if os.path.isdir(folder) and not os.path.islink(folder):
                os.chmod(folder, os.stat(folder).st_mode | stat.S_IRWXU)
        if os.path.isdir(failed) and not os.path.islink(failed):
            shutil.rmtree(failed, onexc=unlocked)
        else:
            os.unlink(failed)

    shutil.rmtree(path, onexc=unlocked)


def check(run_dir: Path, test, name: str) -> int:
    """Refuse, before anything plays, the test ``test`` (the schedule at ``name``) unless it is a frozen test played one
    instance at a time, every instance after the first reset before it, of the configuration and seed of the run in
    ``run_dir``, whose schedule is a stream alone, played whole; returns how many instances the stream played."""
    from continual_arc.runner import prior_records
    from continual_arc.schedule import PHASE_STREAM, TESTS, Schedule

    if test.meta.get("test") not in TESTS:
        raise SystemExit(f"{name} is not a frozen test: make one with `continual-arc schedule make --test`")
    if test.users != 1 or not all(slot.reset_before for slot in test.slots[1:]):
        raise SystemExit(f"{name} does not play one instance at a time, each from the stream's end, and a harness keeps "
                         "one conversation: make the test with --width 1")
    stream = Schedule.load(Path(run_dir) / "schedule.json")
    if PHASE_STREAM not in stream.phases():
        raise SystemExit(f"{run_dir} has no stream for a test to follow")
    if stream.phases() != [PHASE_STREAM]:
        raise SystemExit(f"{run_dir} plays its stream and then {', '.join(stream.phases()[1:])}, so its system's state "
                         "is not the state the stream left: play the stream alone (arc-baselines run --no-epilogue)")
    played = len(prior_records(run_dir))
    if played < len(stream.slots):
        raise SystemExit(f"{run_dir} has played {played} of its stream's {len(stream.slots)} instances")

    def made_for(schedule) -> tuple:
        return schedule.seed, (schedule.meta.get("config") or {}).get("name")

    if made_for(test) != made_for(stream):
        raise SystemExit(f"{name} is the test of {made_for(test)[1]} at seed {made_for(test)[0]}, and {run_dir}'s "
                         f"stream is {made_for(stream)[1]} at seed {made_for(stream)[0]}")
    return played


def first_launch(run_dir: Path):
    """The run's first launch, as ``arc-baselines run`` parsed it (``launches.jsonl``)."""
    from continual_arc_baselines.cli import build_parser

    path = Path(run_dir) / "launches.jsonl"
    if not path.is_file():
        raise SystemExit(f"{run_dir} has no launches.jsonl: a test follows a stream played with arc-baselines run --out")
    return build_parser().parse_args(json.loads(path.read_text().splitlines()[0])["argv"])


def system_of(run_dir: Path) -> Path:
    """The run's system folder, in the attempt its launches wrote to (``attempt.json``)."""
    return Path(json.loads((Path(run_dir) / "attempt.json").read_text())["attempt_dir"]) / "system"


@contextmanager
def holding(run_dir: Path) -> Iterator[None]:
    """The run directory ``run_dir`` for one process at a time, its stream or a test of it, which sets the stream's
    state aside, taken before either writes anything; the lock is the directory's own, and goes with the process."""
    folder = os.open(run_dir, os.O_RDONLY)
    try:
        try:
            fcntl.flock(folder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another process is playing {run_dir}, its stream or a test of it: one at a time plays "
                             "a run") from None
        yield
    finally:
        os.close(folder)


def hold_aside(system: Path, played: int) -> Path:
    """The stream's state directory, ``system/state``, renamed aside for a test from the stream's end after ``played``
    instances (:data:`HELD`); returns where it is."""
    held = system / f"{HELD}-{played}"
    state = system / "state"
    state.mkdir(exist_ok=True)   # a stream that kept nothing
    state.rename(held)
    return held


def give_back(system: Path, held: Path) -> None:
    """The stream's state directory renamed back from where a test set it aside, the test's copy in its place
    deleted."""
    state = system / "state"
    if state.exists():
        delete(state)
    held.rename(state)


def settle(system: Path, played: int) -> None:
    """Put back the stream's state directory that a test stopped hard left aside, when the stream had played
    ``played`` instances, the stream's count now; one set aside at any other point is refused, since the state played
    on since is not the stream's."""
    for held in system.glob(f"{HELD}-*"):
        if held.name != f"{HELD}-{played}":
            raise SystemExit(f"{held} is the stream's state as a test set it aside, and the stream has played {played} "
                             f"instances since, from the test's copy in {system / 'state'}")
        give_back(system, held)
