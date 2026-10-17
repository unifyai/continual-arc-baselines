"""A stand-in for ``prime-agent --mode rpc`` that speaks the documented protocol.

It behaves like a system: it calls its model through whatever base URL the
environment gives it (so the proxy sees usage), keeps a per-task memory file in
its agent directory (so state wipes are observable), and answers the benchmark
protocol: request demos for an unknown task, infer the FakeSource shift from the
first demo pair, submit. Every command it receives is appended to
``$FAKE_AGENT_LOG`` for the tests to inspect.

Failure modes seen with the real prime-agent are replayed on demand, once per
agent directory (a marker file records that the scenario has fired):

* ``FAKE_AGENT_ERROR_FIRST``: a provider error ends the run with a text-less
  message, then the agent retries in a fresh run and answers.
* ``FAKE_AGENT_STALL_ONCE``: the turn ends on the final reply, but no
  ``agent_end`` ever follows (prime-agent 0.9.5, 2026-09-17).
* ``FAKE_AGENT_LOSE_TAIL=<n>``: for the next ``n`` prompts, everything after
  ``message_end`` is lost (no ``turn_end``, no ``agent_end``), exactly as the
  daemon relay did on 2026-09-17; the command path still answers.
* ``FAKE_AGENT_HANG_ONCE``: the prompt is accepted and nothing happens.
* ``FAKE_AGENT_LOOP_ONCE``: the agent keeps starting model calls without
  concluding, until it is aborted; the aborted run ends without an action.
* ``FAKE_AGENT_DIE``: the process exits after accepting every prompt.
* ``FAKE_AGENT_RATE_LIMITED=<n>``: for the next ``n`` prompts the model call
  fails (errored run, no text); prime-agent announces an auto-retry with a
  delay longer than the adapter's default wait, then retries and answers.
* ``FAKE_AGENT_PROVIDER_DOWN=<n>``: for the next ``n`` prompts the model call
  fails and prime-agent's retries are exhausted (``auto_retry_end`` failure);
  the worker is then idle with the previous turn's text as its last reply.
* ``FAKE_AGENT_PROBE=<path>``: on the first prompt, try to read that file and
  log ``probe:ok`` or ``probe:denied`` (for the sandbox test).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import httpx


def log(kind: str) -> None:
    path = os.environ.get("FAKE_AGENT_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(kind + "\n")


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def call_model(prompt: str) -> None:
    base = os.environ.get("OPENAI_BASE_URL")
    if not base:
        return
    httpx.post(base + "/chat/completions", timeout=10, json={
        "model": "vendor/model", "messages": [{"role": "user", "content": prompt}],
    }, headers={"Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "")})


def grid_after(text: str, marker: str) -> list[list[int]]:
    tail = text.split(marker, 1)[1].split("\n", 1)[1]
    rows = []
    for line in tail.splitlines():
        if not re.fullmatch(r"[0-9 ]+", line.strip()) or not line.strip():
            break
        rows.append([int(c) for c in line.split()])
    return rows


def decide(prompt: str, memory: dict) -> str:
    task = re.search(r"Task id: (\S+)", prompt).group(1)
    if "Demo pair 1 input" in prompt:
        demo_in = grid_after(prompt, "Demo pair 1 input")
        demo_out = grid_after(prompt, "Demo pair 1 output")
        memory[task] = (demo_out[0][0] - demo_in[0][0]) % 10
    if task not in memory:
        return 'I do not know this task yet.\n{"action": "request_demos"}'
    grid = grid_after(prompt, "Test input")
    shift = memory[task]
    out = [[(c + shift) % 10 for c in row] for row in grid]
    return "Applying the remembered rule.\n```json\n" + json.dumps({"action": "submit", "grid": out}) + "\n```"


def assistant(text: str, stop: str = "stop") -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}], "stopReason": stop,
            "usage": {"input": 12, "output": 3, "cost": {"total": 0.00025}}}


def main() -> int:
    if "--version" in sys.argv:
        print("fake-prime-agent 0.0.1")
        return 0
    if "shutdown" in sys.argv:
        return 0
    agent_dir = Path(os.environ["PRIME_AGENT_CODING_AGENT_DIR"])
    memory_path = agent_dir / "memory.json"
    memory = json.loads(memory_path.read_text()) if memory_path.exists() else {}

    def once(flag: str) -> bool:
        """True the first time ``flag`` is set for this agent directory."""
        marker = agent_dir / f".fired-{flag}"
        if not os.environ.get(flag) or marker.exists():
            return False
        marker.write_text("1")
        return True

    def times(flag: str) -> bool:
        """True the first ``$flag`` times, counted across restarts in this agent directory."""
        limit = int(os.environ.get(flag) or 0)
        marker = agent_dir / f".count-{flag}"
        used = int(marker.read_text()) if marker.exists() else 0
        if used >= limit:
            return False
        marker.write_text(str(used + 1))
        return True

    last_prompt = ""
    last_reply: str | None = None
    aborted_reply: str | None = None
    suspended = False  # abort suspends queued input until resume_queue, as prime-agent 0.9.5 does
    compacting = 0  # get_state polls left before a compaction after a turn ends (FAKE_AGENT_COMPACT_ONCE)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        command = json.loads(line)
        kind = command.get("type")
        log(kind)
        if kind == "prompt" and compacting and not command.get("streamingBehavior"):
            emit({"type": "response", "command": "prompt", "success": False, "id": command.get("id"),
                  "error": "Agent is already processing. Specify streamingBehavior ('steer' or 'followUp') to "
                           "queue the message."})
            continue
        if kind == "prompt" and suspended and not command.get("streamingBehavior"):
            emit({"type": "response", "command": "prompt", "success": False, "id": command.get("id"),
                  "error": "Cannot admit a session action while queued session input is suspended."})
            continue
        if kind == "prompt" and command.get("streamingBehavior"):
            suspended = False  # resumeIfIdle: a prompt with a streamingBehavior lifts the suspension
        if kind == "prompt":
            emit({"type": "response", "command": "prompt", "success": True, "id": command.get("id")})
            prompt = command["message"]
            if prompt.startswith("/refine"):
                emit({"type": "refine_complete"})
                continue
            if "refine.run(" in prompt:
                # the kernel route: the turn ends, then the host refines and resumes the agent
                emit({"type": "agent_end"})
                emit({"type": "refine_complete", "result": {"scope": "global", "summary": "fake global refine"}})
                emit({"type": "agent_end"})
                continue
            if os.environ.get("FAKE_AGENT_DIE"):
                return 0
            probe = os.environ.get("FAKE_AGENT_PROBE")
            if probe and once("FAKE_AGENT_PROBE"):
                try:
                    open(probe).read(1)
                    log("probe:ok")
                except PermissionError:
                    log("probe:denied")
            # a verdict or a refusal after the instance is over, under the finish protocol: end the instance
            over = "Task id:" not in prompt and ("Your submission was" in prompt or "instance is over" in prompt)
            if "Task id:" not in prompt:
                prompt = last_prompt + "\n\n" + prompt  # a follow-up; the instance is in context
            last_prompt = prompt
            call_model(prompt)
            reply = 'Nothing to add.\n{"action": "finish"}' if over else decide(prompt, memory)
            memory_path.write_text(json.dumps(memory))
            if once("FAKE_AGENT_ERROR_FIRST"):
                # Mimic a provider error: prime-agent ends the run with an errored, text-less
                # assistant message, then retries in a fresh run and answers for real.
                emit({"type": "agent_end", "messages": [
                    {"role": "assistant", "content": [{"type": "thinking", "thinking": "..."}],
                     "stopReason": "error", "errorMessage": "Provider finish_reason: error",
                     "usage": {"input": 12, "output": 0, "cost": {"total": 0.0}}}]})
                emit({"type": "agent_start"})
            if once("FAKE_AGENT_HANG_ONCE"):
                continue
            errored = {"role": "assistant", "content": [], "stopReason": "error",
                       "errorMessage": "429 Provider returned error",
                       "usage": {"input": 12, "output": 0, "cost": {"total": 0.0}}}
            if times("FAKE_AGENT_RATE_LIMITED"):
                for kind_, payload in (("agent_start", None), ("turn_start", None),
                                       ("message_start", errored), ("message_end", errored)):
                    emit({"type": kind_} if payload is None else {"type": kind_, "message": payload})
                emit({"type": "turn_end", "message": errored})
                emit({"type": "agent_end", "messages": [errored]})
                emit({"type": "auto_retry_start", "attempt": 1, "maxAttempts": 30, "delayMs": 2500,
                      "errorMessage": "429 Provider returned error"})
                import time as _time
                _time.sleep(2.5)
                emit({"type": "auto_retry_end", "success": True, "attempt": 1})
            if times("FAKE_AGENT_PROVIDER_DOWN"):
                for kind_, payload in (("agent_start", None), ("turn_start", None),
                                       ("message_start", errored), ("message_end", errored)):
                    emit({"type": kind_} if payload is None else {"type": kind_, "message": payload})
                emit({"type": "turn_end", "message": errored})
                emit({"type": "agent_end", "messages": [errored]})
                emit({"type": "auto_retry_end", "success": False, "attempt": 3, "finalError": "429"})
                continue  # idle now; get_last_assistant_text still returns the previous reply
            emit({"type": "agent_start"})
            emit({"type": "turn_start"})
            emit({"type": "message_start", "message": {"role": "assistant", "content": []}})
            if once("FAKE_AGENT_LOOP_SILENT"):
                # works on without a final word until aborted; the aborted run then ends with
                # an empty agent_end (nothing was said)
                for _ in range(12):
                    emit({"type": "message_start", "message": {"role": "assistant", "content": []}})
                    emit({"type": "tool_execution_start", "toolName": "ipython"})
                    emit({"type": "tool_execution_end", "toolName": "ipython"})
                aborted_reply = ""
                continue
            if once("FAKE_AGENT_LOOP_ONCE"):
                for _ in range(12):
                    emit({"type": "message_start", "message": {"role": "assistant", "content": []}})
                    emit({"type": "message_end", "message": assistant("Let me look at the demos again.", "toolUse")})
                    emit({"type": "tool_execution_start", "toolName": "ipython"})
                    emit({"type": "tool_execution_end", "toolName": "ipython"})
                aborted_reply = "Still exploring the demos; no conclusion yet."
                continue
            emit({"type": "message_update", "text": reply[:10]})
            emit({"type": "message_end", "message": assistant(reply)})
            last_reply = reply
            if times("FAKE_AGENT_LOSE_TAIL"):
                continue  # the relay dropped turn_end and agent_end; the worker is idle
            emit({"type": "turn_end", "message": assistant(reply)})
            if once("FAKE_AGENT_STALL_ONCE"):
                continue  # no agent_end, ever
            if once("FAKE_AGENT_COMPACT_MIDRUN_ONCE"):
                # prime-agent stops a run whose context passed its threshold after a tool result, compacts, and
                # starts the run again
                emit({"type": "agent_end", "messages": [{"role": "user", "content": prompt},
                                                        {"role": "toolResult", "content": [{"type": "text", "text": "42"}]}]})
                emit({"type": "compaction_start", "reason": "threshold"})
                emit({"type": "compaction_end", "reason": "threshold", "aborted": False, "willRetry": False})
                emit({"type": "agent_start"})
            emit({"type": "agent_end", "messages": [{"role": "user", "content": prompt}, assistant(reply)]})
            if once("FAKE_AGENT_COMPACT_ONCE"):
                # prime-agent compacts after a turn whose context passed its threshold, and refuses a prompt meanwhile
                compacting = 2
                emit({"type": "compaction_start", "reason": "threshold"})
        elif kind == "abort":
            suspended = True
            emit({"type": "response", "command": "abort", "success": True})
            if aborted_reply is not None:
                emit({"type": "agent_end", "messages": [assistant(aborted_reply, "aborted")]})
                aborted_reply = None
        elif kind == "get_state":
            emit({"type": "response", "command": kind, "success": True, "id": command.get("id"),
                  "data": {"isStreaming": False, "isCompacting": compacting > 0}})
            if compacting:
                compacting -= 1
                if not compacting:
                    emit({"type": "compaction_end", "reason": "threshold", "aborted": False, "willRetry": False})
        elif kind == "get_last_assistant_text":
            emit({"type": "response", "command": kind, "success": True, "id": command.get("id"),
                  "data": {"text": last_reply}})
        elif kind == "refine":
            # the RPC refine (0.9.5): the refinement's one model call, its edits applied, the event, then the answer
            call_model("REFINE\n" + str(command.get("instructions") or ""))
            scope = "global" if command.get("global") else "local"
            result = {"scope": scope, "summary": f"fake {scope} refine", "appliedEdits": []}
            emit({"type": "refine_complete", "result": result})
            emit({"type": "response", "command": kind, "success": True, "id": command.get("id"), "data": result})
        elif kind in ("new_session", "get_session_stats"):
            emit({"type": "response", "command": kind, "success": True, "data": {}})
        else:
            emit({"type": "response", "command": kind, "success": False, "error": "unknown"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
