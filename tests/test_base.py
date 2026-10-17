"""The system loop's own policies, exercised on a stub system."""
from continual_arc_baselines.systems.base import SystemLearner, SystemUnavailable


class Stub(SystemLearner):
    """Answers from a script; records starts so a restart is visible."""

    def __init__(self, replies, **kwargs):
        super().__init__(**kwargs)
        self.replies = list(replies)
        self.starts = 0

    def start(self):
        self.starts += 1

    def stop(self):
        pass

    def ask(self, message, *, reprompt=False):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def new_session(self):
        pass

    def wipe_state(self):
        pass

    def describe(self):
        return {}


def test_empty_reply_restarts_the_system_like_a_crash():
    stub = Stub(["", '{"action": "request_demos"}'], max_restarts=2)
    assert stub._ask_resilient(lambda fresh: "obs") == '{"action": "request_demos"}'
    assert stub.stats["restarts"] == 1
    assert stub.starts == 2


def test_persistently_empty_system_is_unavailable_after_its_restarts():
    stub = Stub(["", "   ", "\n"], max_restarts=2)
    try:
        stub._ask_resilient(lambda fresh: "obs")
    except SystemUnavailable as error:
        assert "empty reply" in str(error)
    else:
        raise AssertionError("an empty system must be unavailable, not answering")
    assert stub.stats["restarts"] == 2
