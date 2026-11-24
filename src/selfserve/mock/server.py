"""A stand-in inference server.

This is what makes the rest of the project testable, the same role the fake LLM
plays in 01-rag and the scripted model plays in 02-agents. It speaks enough of
the OpenAI API -- streaming SSE, usage accounting, `/health`, `/v1/models` --
that the client, the gateway and the benchmark cannot tell it from vLLM, and it
runs on a laptop with no GPU.

**The part that makes it worth having** is that it models *contention* rather
than fixed latency. A mock that sleeps 20ms per token regardless of load would
make every benchmark produce a straight line, and the whole point of
benchmarking an inference server is the shape of the curve as concurrency
rises. So:

* time to first token grows with prompt length (prefill is compute-bound and
  roughly linear in prompt tokens), and with how many requests are already in
  flight (yours queues behind their prefills);
* inter-token latency grows with the number of concurrent sequences, because
  continuous batching runs them in one batch and every sequence in a batch
  advances together at the speed of the batch;
* past `capacity`, requests are rejected with 503 rather than queued forever,
  which is what a server with a bounded KV cache does.

That reproduces the trade-off at the centre of serving: more concurrency means
higher total throughput and worse per-user latency, up to a cliff. The absolute
numbers are invented. The shape is not, and the shape is what you tune against.

It is a simulator for exercising the plumbing, not a predictor of what your
hardware will do. For that, point the benchmark at a real server.
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from typing import Any

from selfserve.httpserver import QuietThreadingHTTPServer

LOREM = (
    "Serving an open model yourself means owning the capacity question. "
    "Throughput and latency trade against each other through batch size, and "
    "the KV cache is what sets the ceiling on both. Measure before you tune, "
    "and measure the percentiles rather than the mean. "
)


@dataclass(slots=True)
class MockConfig:
    """Knobs for the simulated server.

    Defaults are loosely in the shape of an 8B model on one mid-range GPU:
    ~80ms to first token on a short prompt, ~45 tokens/s single-stream decode.
    """

    model: str = "mock-8b"
    base_ttft_s: float = 0.080
    prefill_s_per_1k: float = 0.035  # prefill cost per 1000 prompt tokens
    base_itl_s: float = 0.022  # ~45 tok/s with nothing else running
    # How much each additional in-flight sequence slows every sequence's decode.
    # Sub-linear on purpose: that is the whole value of continuous batching --
    # doubling the batch does not halve per-stream speed.
    batch_penalty: float = 0.06
    capacity: int = 32  # concurrent sequences before 503
    failure_rate: float = 0.0  # fraction of requests that fail with 500
    seed: int = 7


class _State:
    """In-flight request accounting, shared across handler threads."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.served = 0
        self.rejected = 0
        self.peak_active = 0

    def enter(self, capacity: int) -> tuple[bool, int]:
        with self.lock:
            if self.active >= capacity:
                self.rejected += 1
                return False, self.active
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            return True, self.active

    def leave(self) -> None:
        with self.lock:
            self.active -= 1
            self.served += 1

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return {
                "active": self.active,
                "served": self.served,
                "rejected": self.rejected,
                "peak_active": self.peak_active,
            }


class MockServer:
    """A threaded OpenAI-compatible server on an ephemeral port.

    Used as a context manager so a test cannot leak one:

        with MockServer(MockConfig(capacity=4)) as server:
            InferenceClient(server.url).chat(ChatRequest.of("hi"))
    """

    def __init__(self, config: MockConfig | None = None, port: int = 0) -> None:
        self.config = config or MockConfig()
        self.state = _State()
        handler = _make_handler(self.config, self.state)
        self._httpd = QuietThreadingHTTPServer(("127.0.0.1", port), handler)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> MockServer:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> MockServer:
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()


def _make_handler(config: MockConfig, state: _State) -> type[BaseHTTPRequestHandler]:
    rng = random.Random(config.seed)
    rng_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "selfserve-mock/0.1"

        # Silence the default stderr access log: a benchmark at 64-way
        # concurrency would otherwise print thousands of lines into the
        # middle of its own report.
        def log_message(self, *_: Any) -> None:
            return

        # ------------------------------------------------------------- GET
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
            if self.path.startswith("/health"):
                return self._json(200, {"status": "ok", **state.snapshot()})
            if self.path.startswith("/v1/models"):
                return self._json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": config.model, "object": "model", "owned_by": "selfserve"}],
                    },
                )
            if self.path.startswith("/metrics"):
                return self._json(200, state.snapshot())
            return self._error(404, f"no such path: {self.path}")

        # ------------------------------------------------------------ POST
        def do_POST(self) -> None:  # noqa: N802
            if not self.path.startswith("/v1/chat/completions"):
                return self._error(404, f"no such path: {self.path}")

            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError) as exc:
                return self._error(400, f"malformed request body: {exc}")

            with rng_lock:
                unlucky = rng.random() < config.failure_rate
            if unlucky:
                return self._error(500, "simulated backend failure")

            admitted, in_flight = state.enter(config.capacity)
            if not admitted:
                # What a real server does when the KV cache is full. Returning
                # 503 rather than queueing is the honest behaviour: a request
                # that cannot be served should be refused while the caller can
                # still do something about it.
                return self._error(
                    503,
                    f"server at capacity ({config.capacity} concurrent sequences)",
                    headers={"Retry-After": "1"},
                )
            try:
                self._complete(body, in_flight)
            finally:
                state.leave()

        # -------------------------------------------------------- generation
        def _complete(self, body: dict, in_flight: int) -> None:
            prompt_tokens = _count_prompt_tokens(body.get("messages") or [])
            max_tokens = int(body.get("max_tokens") or 64)
            stream = bool(body.get("stream"))

            ttft = config.base_ttft_s + (prompt_tokens / 1000) * config.prefill_s_per_1k
            # Queueing behind other prefills: your first token waits for them.
            ttft *= 1 + config.batch_penalty * (in_flight - 1)
            itl = config.base_itl_s * (1 + config.batch_penalty * (in_flight - 1))

            words = LOREM.split()
            tokens = [words[i % len(words)] + " " for i in range(max_tokens)]

            if not stream:
                time.sleep(ttft + itl * max_tokens)
                return self._json(
                    200, _completion_body(config.model, "".join(tokens), prompt_tokens, max_tokens)
                )

            # Chunked transfer encoding, which is what uvicorn (and therefore
            # vLLM and SGLang) actually puts on the wire for SSE. The
            # alternative -- no Content-Length and no chunking on an HTTP/1.1
            # keep-alive connection -- leaves the body with no defined end, so
            # a well-behaved proxy reading to EOF blocks until a timeout. A
            # mock that gets this wrong tests the client against a protocol no
            # real server speaks.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            time.sleep(ttft)
            try:
                for token in tokens:
                    self._sse(_chunk_body(config.model, token))
                    time.sleep(itl)
                self._sse(_usage_body(config.model, prompt_tokens, max_tokens))
                self._write_chunk(b"data: [DONE]\n\n")
                self.wfile.write(b"0\r\n\r\n")  # terminal chunk
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The client hung up mid-stream -- a cancelled request. Normal,
                # and worth not crashing a handler thread over.
                return

        # ------------------------------------------------------------ output
        def _sse(self, payload: dict) -> None:
            self._write_chunk(f"data: {json.dumps(payload)}\n\n".encode())

        def _write_chunk(self, body: bytes) -> None:
            """One HTTP chunk: hex length, CRLF, payload, CRLF."""
            self.wfile.write(f"{len(body):X}\r\n".encode() + body + b"\r\n")
            self.wfile.flush()

        def _json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, message: str, headers: dict | None = None) -> None:
            data = json.dumps({"error": {"message": message, "type": "mock_error"}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

    return Handler


# ------------------------------------------------------------------- bodies
def _count_prompt_tokens(messages: list[dict]) -> int:
    """Characters / 4, the universal rough tokenizer.

    Good enough for a simulator. Anywhere the real count matters, the number
    comes from the server's own usage block instead of from here.
    """
    return max(1, sum(len(m.get("content", "")) for m in messages) // 4)


def _chunk_body(model: str, token: str) -> dict:
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
    }


def _usage_body(model: str, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _completion_body(model: str, text: str, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
