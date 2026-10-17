"""A minimal OpenAI-compatible upstream for tests: records what it was sent."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeUpstream:
    def __init__(self, reply: str = "OK", cost: float = 0.00025, status: int | list[int] = 200,
                 error: dict | None = None, reasoning: str | None = None, replies: list[str] | None = None):
        self.reply, self.cost, self.error = reply, cost, error  # error: the JSON body of a non-streamed 4xx or 5xx
        self.replies = list(replies) if replies else None  # a scripted sequence of replies; its last value repeats
        self.reasoning = reasoning  # sent as the reply's reasoning_content (a delta of its own when streamed)
        self._statuses = list(status) if isinstance(status, (list, tuple)) else [status]
        self.requests: list[dict] = []
        self._server: ThreadingHTTPServer | None = None

    @property
    def status(self) -> int:
        """The status of the next response: a scripted sequence is consumed, its last value repeats."""
        return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]

    @status.setter
    def status(self, value: int | list[int]) -> None:
        self._statuses = list(value) if isinstance(value, (list, tuple)) else [value]

    def start(self) -> FakeUpstream:
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def do_GET(self):
                # a model list as OpenRouter serves it, with the provider's advertised window
                upstream.requests.append({"path": self.path, "body": None,
                                          "authorization": self.headers.get("Authorization")})
                data = json.dumps({"data": [{"id": "vendor/model", "context_length": 1_050_000,
                                             "top_provider": {"context_length": 1_050_000}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                upstream.requests.append({"path": self.path, "body": body,
                                          "authorization": self.headers.get("Authorization"),
                                          "rank": self.headers.get("X-data-parallel-rank")})
                gen = f"gen-{len(upstream.requests)}"
                if upstream.replies:
                    upstream.reply = upstream.replies.pop(0) if len(upstream.replies) > 1 else upstream.replies[0]
                usage = {"prompt_tokens": 12, "completion_tokens": 3,
                         "completion_tokens_details": {"reasoning_tokens": 2}, "cost": upstream.cost}
                # a vLLM server asked for return_token_ids: the prompt's ids (here: one per message,
                # then 1, 2) and the reply's (7, 8)
                ids = bool(body.get("return_token_ids"))
                prompt_ids = [len(body.get("messages") or []), 1, 2]
                if body.get("stream"):
                    self.send_response(upstream.status)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    chunks = [
                        {"id": gen, "model": "vendor/model", "choices": [{"delta": {"content": upstream.reply}, "finish_reason": None}]},
                        {"id": gen, "model": "vendor/model", "choices": [{"delta": {}, "finish_reason": "stop"}]},
                        {"id": gen, "model": "vendor/model", "choices": [], "usage": usage},
                    ]
                    if ids:
                        chunks[0]["prompt_token_ids"] = prompt_ids
                        chunks[0]["choices"][0]["token_ids"] = [7]
                        chunks[1]["choices"][0]["token_ids"] = [8]
                    if upstream.reasoning is not None:
                        chunks.insert(0, {"id": gen, "model": "vendor/model", "choices": [
                            {"delta": {"reasoning_content": upstream.reasoning}, "finish_reason": None}]})
                    for chunk in chunks:
                        data = b"data: " + json.dumps(chunk).encode() + b"\n\n"
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                    done = b"data: [DONE]\n\n"
                    self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(done), done))
                    self.wfile.flush()
                    return
                payload = {"id": gen, "model": "vendor/model",
                           "choices": [{"message": {"role": "assistant", "content": upstream.reply},
                                        "finish_reason": "stop"}],
                           "usage": usage}
                if upstream.reasoning is not None:
                    payload["choices"][0]["message"]["reasoning_content"] = upstream.reasoning
                if ids:
                    payload["prompt_token_ids"] = prompt_ids
                    payload["choices"][0]["token_ids"] = [7, 8]
                status = upstream.status
                data = json.dumps(upstream.error if status >= 400 and upstream.error else payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Request-Id", "req-1")
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    @property
    def base_url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"
