"""A minimal chat-completions client for the prompting floors.

It talks to whatever ``ARC_LLM_BASE_URL`` (the tracking proxy) or
``OPENAI_BASE_URL`` names, so every call is journaled at the HTTP boundary like
a system's. Retries cover transport errors and 429/5xx with a doubling backoff;
nothing is cached. A ``send`` callable can be injected for tests.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any

import httpx

DEFAULT_BASE = "https://openrouter.ai/api/v1"


class ChatError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, model: str, *, base_url: str | None = None, api_key: str | None = None,
                 timeout_s: float = 600.0, retries: int = 5, backoff_s: float = 10.0,
                 defaults: dict[str, Any] | None = None,
                 send: Callable[[dict], dict] | None = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("ARC_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
                         or DEFAULT_BASE).rstrip("/")
        self.api_key = api_key or os.environ.get("ARC_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.defaults = dict(defaults or {})
        self.last_finish_reason: str | None = None  # of the latest completion; "length" means the reply was cut at max_tokens
        self._send = send
        self._client: httpx.Client | None = None
        self.calls = 0
        self.failures = 0

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)
        return self._client

    def _post(self, body: dict) -> dict:
        """Stream the completion and fold the deltas into one response object.

        Streaming matters: the proxy's upstream timeout is a silence timeout, and
        a reasoning model answering a puzzle can be silent for minutes on a
        non-streaming request while it streams deltas continuously.
        """
        if self._send is not None:
            return self._send(body)
        body = {**body, "stream": True}
        text_parts: list[str] = []
        usage: dict | None = None
        finish = None
        with self._http().stream("POST", f"{self.base_url}/chat/completions", json=body,
                                 headers={"Authorization": f"Bearer {self.api_key}"}) as resp:
            if resp.status_code == 429 or resp.status_code >= 500:
                raise ChatError(f"HTTP {resp.status_code}: {resp.read()[:200]!r}")
            if resp.status_code != 200:
                raise ChatError(f"HTTP {resp.status_code} (not retried): {resp.read()[:200]!r}")
            if "text/event-stream" not in (resp.headers.get("content-type") or ""):
                return json.loads(resp.read())
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    # Keep reading until the server closes: the proxy journals the
                    # call after its last write, and the runner reads the counters
                    # as soon as this returns.
                    continue
                try:
                    event = json.loads(payload)
                except ValueError:
                    continue
                if isinstance(event.get("error"), dict):
                    msg = event["error"].get("message", "")[:200]
                    if "response_format" in msg or "abnormal" in msg:
                        # the provider aborted a JSON-mode reply because the model's output
                        # degenerated: that is the model's answer, not a transient fault
                        raise ChatError(f"upstream error (not retried): {msg}")
                    raise ChatError(f"upstream error: {msg}")
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str):
                        text_parts.append(content)
                    finish = choice.get("finish_reason") or finish
                if event.get("usage"):
                    usage = event["usage"]
        return {"choices": [{"message": {"content": "".join(text_parts)}, "finish_reason": finish}],
                "usage": usage}

    def complete(self, messages: list[dict], **overrides: Any) -> str:
        """The assistant text for ``messages``; empty on a content-less reply."""
        body = {"model": self.model, "messages": messages, **self.defaults, **overrides}
        body = {k: v for k, v in body.items() if v is not None}  # an override of None drops a default
        delay = self.backoff_s
        for attempt in range(self.retries + 1):
            self.calls += 1
            try:
                data = self._post(body)
                choice = (data.get("choices") or [{}])[0]
                self.last_finish_reason = choice.get("finish_reason")
                message = choice.get("message") or {}
                content = message.get("content")
                if isinstance(content, list):
                    content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                return content or ""
            except (httpx.HTTPError, ChatError) as error:
                self.failures += 1
                if "not retried" in str(error) or attempt == self.retries:
                    raise ChatError(str(error)) from error
                time.sleep(delay)
                delay = min(delay * 2, 300.0)
        return ""

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
