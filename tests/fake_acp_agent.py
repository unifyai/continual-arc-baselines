"""A minimal ACP agent for tests: replies to every prompt with a canned action.

Started as ``python fake_acp_agent.py``. It writes a marker file under
``$FAKE_ACP_STATE`` on every prompt so tests can check the state directory and
its wipe. The reply is the JSON action in ``$FAKE_ACP_REPLY`` (default: give up),
optionally preceded by prose, and it streams as two chunks to exercise
accumulation. A prompt containing ``SLEEP`` sleeps ``$FAKE_ACP_SLEEP`` seconds
first, to exercise timeouts. The first ``$FAKE_ACP_EMPTY`` prompts the agent's state has
seen (restarts included) end with no text at all (stop reason end_turn), as openclaw's
bridge ends a run that failed or whose work went to subagents.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, Agent, run_agent, update_agent_message_text
from acp.schema import (
    AgentCapabilities,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    Usage,
)


class FakeAgent(Agent):
    def __init__(self) -> None:
        self.conn: Any = None
        self.prompts = 0

    def on_connect(self, conn: Any) -> None:
        self.conn = conn

    async def initialize(self, protocol_version: int, client_capabilities: Any = None,
                         client_info: Any = None, **kwargs: Any) -> InitializeResponse:
        return InitializeResponse(protocol_version=PROTOCOL_VERSION,
                                  agent_capabilities=AgentCapabilities())

    async def new_session(self, cwd: str, additional_directories: Any = None,
                          mcp_servers: Any = None, **kwargs: Any) -> NewSessionResponse:
        return NewSessionResponse(session_id=uuid.uuid4().hex)

    async def prompt(self, session_id: str, prompt: list, **kwargs: Any) -> PromptResponse:
        self.prompts += 1
        text = "".join(getattr(block, "text", "") for block in prompt)
        state = os.environ.get("FAKE_ACP_STATE")
        if state:
            Path(state).mkdir(parents=True, exist_ok=True)
            (Path(state) / f"prompt-{self.prompts}.txt").write_text(text)
        if "SLEEP" in text:
            await asyncio.sleep(float(os.environ.get("FAKE_ACP_SLEEP", "5")))
        marks = Path(state) if state else None
        done = len(list(marks.glob("empty-*"))) if marks else self.prompts - 1
        if done < int(os.environ.get("FAKE_ACP_EMPTY") or 0):   # once per agent state, across restarts
            if marks:
                (marks / f"empty-{done + 1}").touch()
            return PromptResponse(stop_reason="end_turn",
                                  usage=Usage(input_tokens=100 * self.prompts, output_tokens=0,
                                              total_tokens=100 * self.prompts))
        reply = os.environ.get("FAKE_ACP_REPLY") or json.dumps({"action": "give_up"})
        await self.conn.session_update(session_id=session_id,
                                       update=update_agent_message_text("Thinking about it.\n"))
        await self.conn.session_update(session_id=session_id,
                                       update=update_agent_message_text(reply))
        # usage is the session total across all turns, as the ACP schema defines it
        return PromptResponse(stop_reason="end_turn",
                              usage=Usage(input_tokens=100 * self.prompts, output_tokens=10 * self.prompts,
                                          total_tokens=110 * self.prompts))

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        return None

    async def authenticate(self, *args: Any, **kwargs: Any):
        return None

    async def load_session(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    async def set_session_mode(self, *args: Any, **kwargs: Any):
        return None

    async def set_config_option(self, *args: Any, **kwargs: Any):
        return None

    async def close_session(self, *args: Any, **kwargs: Any):
        return None

    async def fork_session(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    async def list_sessions(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    async def resume_session(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    async def ext_method(self, *args: Any, **kwargs: Any):
        return {}

    async def ext_notification(self, *args: Any, **kwargs: Any):
        return None


if __name__ == "__main__":
    sys.exit(asyncio.run(run_agent(FakeAgent())))
