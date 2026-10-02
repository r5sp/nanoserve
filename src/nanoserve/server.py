"""A small OpenAI-compatible HTTP server (stdlib only).

Endpoints:

* ``POST /v1/completions`` -- ``prompt`` (string or token-id list), ``max_tokens``,
  ``temperature``, ``top_p``, ``top_k`` (extension), ``n``, ``seed``, ``stop``,
  ``stream`` (server-sent events, terminated by ``data: [DONE]``).
* ``GET /v1/models`` and ``GET /health``.

Concurrency model: the engine is single-threaded and owned by one background
:class:`EngineWorker` thread that runs ``engine.step()`` in a loop. HTTP handler
threads only enqueue requests and read per-request output queues, so concurrent
HTTP requests are batched together by the scheduler (continuous batching).
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from nanoserve.engine import LLMEngine, RequestOutput
from nanoserve.sampling import SamplingParams
from nanoserve.tokenizer import IncrementalDetokenizer, Tokenizer


class EngineWorker:
    """Runs the engine loop on a dedicated thread and fans outputs out to callers."""

    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine
        self._inbox: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._streams: dict[str, queue.Queue[RequestOutput]] = {}
        self._thread = threading.Thread(target=self._loop, name="engine", daemon=True)
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        self._inbox.put(("noop", None))
        self._thread.join(timeout=5)

    def submit(
        self, prompt: list[int], params: SamplingParams
    ) -> tuple[str, queue.Queue[RequestOutput]]:
        rid = f"cmpl-{uuid.uuid4().hex[:24]}"
        out: queue.Queue[RequestOutput] = queue.Queue()
        self._inbox.put(("add", (rid, prompt, params, out)))
        return rid, out

    def abort(self, request_id: str) -> None:
        self._inbox.put(("abort", request_id))

    def _handle(self, kind: str, payload: Any) -> None:
        if kind == "add":
            rid, prompt, params, out = payload
            try:
                self.engine.add_request(prompt, params, request_id=rid)
                self._streams[rid] = out
            except ValueError as e:
                out.put(RequestOutput(rid, -1, [], [], True, f"error: {e}"))
        elif kind == "abort":
            self.engine.abort_request(payload)
            self._streams.pop(payload, None)

    def _loop(self) -> None:
        while not self._stop.is_set():
            # Block only when idle; otherwise drain whatever arrived and keep stepping.
            if not self.engine.has_unfinished():
                self._handle(*self._inbox.get())
            while True:
                try:
                    self._handle(*self._inbox.get_nowait())
                except queue.Empty:
                    break
            if not self.engine.has_unfinished():
                continue
            for out in self.engine.step():
                stream = self._streams.get(out.request_id)
                if stream is not None:
                    stream.put(out)
            for rid in [r for r in self._streams if r not in self.engine.sequences]:
                del self._streams[rid]


@dataclass
class _Choice:
    detok: IncrementalDetokenizer
    stops: list[str]
    sent: int = 0  # characters of detok.text already returned
    finish_reason: str | None = None
    num_tokens: int = 0
    text: str = field(default="")

    def update(self, out: RequestOutput) -> str:
        """Consume an engine output; return the text that is safe to emit now."""
        self.detok.push(out.new_token_ids)
        self.num_tokens = len(out.output_token_ids)
        if out.finished:
            self.detok.flush()
            self.finish_reason = out.finish_reason
        text = self.detok.text
        for s in self.stops:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
                self.finish_reason = "stop"
        self.text = text
        # Hold back a tail that could still turn out to be the start of a stop string.
        hold = 0 if self.finish_reason else max((len(s) - 1 for s in self.stops), default=0)
        end = max(self.sent, len(text) - hold)
        delta = text[self.sent : end]
        self.sent = end
        return delta


class CompletionServer:
    def __init__(self, engine: LLMEngine, tokenizer: Tokenizer, model_name: str) -> None:
        self.worker = EngineWorker(engine)
        self.tokenizer = tokenizer
        self.model_name = model_name

    def parse(self, body: dict[str, Any]) -> tuple[list[int], SamplingParams, list[str], bool]:
        prompt = body.get("prompt", "")
        if isinstance(prompt, str):
            ids = self.tokenizer.encode(prompt)
        elif isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
            ids = list(prompt)
        else:
            raise ValueError("prompt must be a string or a list of token ids")
        stop = body.get("stop") or []
        stops = [stop] if isinstance(stop, str) else list(stop)
        params = SamplingParams(
            temperature=float(body.get("temperature", 1.0)),
            top_p=float(body.get("top_p", 1.0)),
            top_k=int(body.get("top_k", 0)),
            max_tokens=int(body.get("max_tokens", 16)),
            seed=body.get("seed"),
            n=int(body.get("n", 1)),
            ignore_eos=bool(body.get("ignore_eos", False)),
        )
        return ids, params, stops, bool(body.get("stream", False))

    def run(
        self, ids: list[int], params: SamplingParams, stops: list[str]
    ) -> Generator[tuple[int, str, _Choice], None, None]:
        """Yield ``(choice_index, text_delta, choice)`` until every choice finishes."""
        rid, outq = self.worker.submit(ids, params)
        choices = [_Choice(IncrementalDetokenizer(self.tokenizer), stops) for _ in range(params.n)]
        try:
            while not all(c.finish_reason for c in choices):
                out = outq.get()
                if out.index < 0:
                    raise ValueError(out.finish_reason or "request rejected")
                c = choices[out.index]
                if c.finish_reason:  # already cut by a stop string
                    continue
                delta = c.update(out)
                yield out.index, delta, c
        finally:
            self.worker.abort(rid)  # no-op if finished; frees KV if the client left early

    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
        server = ThreadingHTTPServer((host, port), _make_handler(self))
        server.daemon_threads = True
        self.worker.start()
        return server


def _make_handler(app: CompletionServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # quieter default logging
            pass

        def _json(self, status: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: HTTPStatus, msg: str) -> None:
            self._json(status, {"error": {"message": msg, "type": "invalid_request_error"}})

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": app.model_name, "object": "model", "owned_by": "nanoserve"}
                ]})  # fmt: skip
            else:
                self._error(HTTPStatus.NOT_FOUND, f"no route {self.path}")

        def do_POST(self) -> None:
            if self.path != "/v1/completions":
                self._error(HTTPStatus.NOT_FOUND, f"no route {self.path}")
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                ids, params, stops, stream = app.parse(body)
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                self._error(HTTPStatus.BAD_REQUEST, str(e))
                return
            created = int(time.time())
            try:
                if stream:
                    self._stream(ids, params, stops, created)
                else:
                    self._complete(ids, params, stops, created)
            except ValueError as e:
                self._error(HTTPStatus.BAD_REQUEST, str(e))

        def _complete(
            self, ids: list[int], params: SamplingParams, stops: list[str], created: int
        ) -> None:
            choices: dict[int, _Choice] = {}
            cid = f"cmpl-{uuid.uuid4().hex[:24]}"
            for index, _, c in app.run(ids, params, stops):
                choices[index] = c
            completion_tokens = sum(c.num_tokens for c in choices.values())
            self._json(200, {
                "id": cid, "object": "text_completion", "created": created,
                "model": app.model_name,
                "choices": [
                    {"index": i, "text": c.text, "finish_reason": c.finish_reason,
                     "logprobs": None}
                    for i, c in sorted(choices.items())
                ],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": completion_tokens,
                          "total_tokens": len(ids) + completion_tokens},
            })  # fmt: skip

        def _stream(
            self, ids: list[int], params: SamplingParams, stops: list[str], created: int
        ) -> None:
            cid = f"cmpl-{uuid.uuid4().hex[:24]}"
            events = app.run(ids, params, stops)
            first = next(events, None)  # surface validation errors before headers go out
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def send(index: int, text: str, finish: str | None) -> None:
                chunk = {
                    "id": cid, "object": "text_completion", "created": created,
                    "model": app.model_name,
                    "choices": [{"index": index, "text": text, "finish_reason": finish,
                                 "logprobs": None}],
                }  # fmt: skip
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            try:
                if first is not None:
                    send(first[0], first[1], first[2].finish_reason)
                for index, delta, c in events:
                    if delta or c.finish_reason:
                        send(index, delta, c.finish_reason)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                events.close()  # triggers abort in app.run's finally block

    return Handler
