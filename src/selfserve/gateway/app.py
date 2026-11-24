"""The gateway.

vLLM and SGLang are inference engines. They are very good at being inference
engines and they are not trying to be the edge of your deployment: there is no
authentication, no notion of more than one replica, no admission control beyond
their own scheduler queue, and no per-caller accounting. Those are the
gateway's job.

What it does, in the order it does it:

    auth ──▶ admission ──▶ route ──▶ proxy (streaming) ──▶ account
              │             │          │
              └─ 503 ───────┴─ failover ┘ (only before the first byte)

**The failover rule is the subtle one.** A request that failed to connect can
be retried on another backend, because nothing has been sent to the client yet.
A request that has already streamed one token cannot: the client holds partial
output and the 200 status is committed, so "retrying" would splice two
different completions together and the caller would have no way to tell. So
retries happen strictly before the first byte reaches the client, and after
that a failure is a truncated stream and is reported as one.

**Streaming is a passthrough, not a buffer.** Chunks are forwarded as they
arrive. Collecting the response and sending it at the end would be simpler and
would destroy the only property that matters about a streaming endpoint -- the
user would wait for the whole generation before seeing anything, and the
gateway would have turned a 200ms time-to-first-token into fifteen seconds.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from typing import Any

from selfserve.gateway.admission import AdmissionControl, Rejected, slot
from selfserve.gateway.backends import Backend, BackendPool
from selfserve.gateway.router import NoBackendAvailable, Router
from selfserve.httpserver import QuietThreadingHTTPServer

PROXY_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/embeddings")


@dataclass(slots=True)
class GatewayConfig:
    backends: list[str] = field(default_factory=lambda: ["http://localhost:8000"])
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str | None = None  # required from clients, if set
    backend_api_key: str | None = None  # sent to backends, if set
    max_concurrency: int = 64
    queue_limit: int = 128
    queue_timeout_s: float = 10.0
    request_timeout_s: float = 300.0
    max_retries: int = 2
    failure_threshold: int = 3
    recovery_s: float = 15.0
    log_requests: bool = True


@dataclass(slots=True)
class AccessLog:
    """What the gateway knows that no single backend can.

    Token counts are sniffed out of the response as it streams past, so the
    accounting is the server's own numbers rather than an estimate -- which is
    what makes this usable for cost attribution rather than just for graphs.
    """

    entries: list[dict] = field(default_factory=list)
    limit: int = 1000
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, entry: dict) -> None:
        with self._lock:
            self.entries.append(entry)
            if len(self.entries) > self.limit:
                del self.entries[: len(self.entries) - self.limit]

    def totals(self) -> dict:
        with self._lock:
            entries = list(self.entries)
        return {
            "requests": len(entries),
            "prompt_tokens": sum(e.get("prompt_tokens", 0) for e in entries),
            "completion_tokens": sum(e.get("completion_tokens", 0) for e in entries),
            "errors": sum(1 for e in entries if e.get("status", 200) >= 500),
            "rejected": sum(1 for e in entries if e.get("status") == 503),
        }

    def recent(self, n: int = 20) -> list[dict]:
        with self._lock:
            return self.entries[-n:]


class Gateway:
    """Assembles the pieces and serves them."""

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self.pool = BackendPool.of(
            config.backends,
            api_key=config.backend_api_key,
            failure_threshold=config.failure_threshold,
            recovery_s=config.recovery_s,
            timeout=config.request_timeout_s,
        )
        if not len(self.pool):
            raise ValueError("gateway needs at least one backend")
        self.router = Router(self.pool)
        self.admission = AdmissionControl(
            max_concurrency=config.max_concurrency,
            queue_limit=config.queue_limit,
            queue_timeout_s=config.queue_timeout_s,
        )
        self.log = AccessLog()
        self._httpd: QuietThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ lifecycle
    def serve_forever(self) -> None:
        httpd = self._build()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.shutdown()
            httpd.server_close()

    def start(self) -> Gateway:
        self._httpd = self._build()
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def port(self) -> int:
        if self._httpd:
            return self._httpd.server_address[1]
        return self.config.port

    @property
    def url(self) -> str:
        return f"http://{self.config.host}:{self.port}"

    def __enter__(self) -> Gateway:
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    def _build(self) -> QuietThreadingHTTPServer:
        return QuietThreadingHTTPServer((self.config.host, self.config.port), _handler(self))

    # --------------------------------------------------------------- status
    def status(self) -> dict:
        return {
            "backends": self.pool.snapshot(),
            "admission": self.admission.snapshot(),
            "usage": self.log.totals(),
        }


def _handler(gateway: Gateway) -> type[BaseHTTPRequestHandler]:
    config = gateway.config

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "selfserve-gateway/0.1"

        def log_message(self, *_: Any) -> None:
            return  # the gateway keeps its own structured log

        # ------------------------------------------------------------- GET
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/health":
                healthy = gateway.pool.healthy
                return self._json(
                    200 if healthy else 503,
                    {
                        "status": "ok" if healthy else "no healthy backend",
                        "healthy_backends": len(healthy),
                        "total_backends": len(gateway.pool),
                    },
                )
            if path == "/status":
                return self._json(200, gateway.status())
            if path == "/metrics":
                return self._json(200, {**gateway.log.totals(), "recent": gateway.log.recent(10)})
            if path == "/v1/models":
                return self._models()
            return self._error(404, f"no such path: {path}")

        def _models(self) -> None:
            for backend in gateway.pool.healthy or list(gateway.pool):
                try:
                    names = backend.client.models()
                except Exception:  # noqa: BLE001 - try the next backend
                    continue
                return self._json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {"id": n, "object": "model", "owned_by": "selfserve"} for n in names
                        ],
                    },
                )
            return self._error(503, "no backend could list models")

        # ------------------------------------------------------------ POST
        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path not in PROXY_PATHS:
                return self._error(404, f"no such path: {path}")
            if not self._authorised():
                return self._error(401, "missing or invalid Authorization header")

            try:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                body = json.loads(raw or b"{}")
            except (ValueError, json.JSONDecodeError) as exc:
                return self._error(400, f"malformed request body: {exc}")

            started = time.perf_counter()
            try:
                with slot(gateway.admission) as admitted:
                    self._proxy(path, raw, body, started, admitted.queued_s)
            except Rejected as exc:
                gateway.log.record(
                    {
                        "path": path,
                        "status": 503,
                        "error": str(exc),
                        "duration_s": round(time.perf_counter() - started, 4),
                    }
                )
                return self._error(503, str(exc), {"Retry-After": str(exc.retry_after)})

        # ----------------------------------------------------------- proxy
        def _proxy(
            self, path: str, raw: bytes, body: dict, started: float, queued_s: float
        ) -> None:
            tried: set[str] = set()
            attempts = config.max_retries + 1

            for attempt in range(attempts):
                try:
                    backend = gateway.router.pick(exclude=tried)
                except NoBackendAvailable as exc:
                    # Having tried backends and run out means they failed:
                    # that is a bad gateway. Having none to try at all is a
                    # configuration or capacity problem. Different causes,
                    # different status codes, different runbook entries.
                    return self._fail(path, 502 if tried else 503, str(exc), started)

                tried.add(backend.url)
                backend.acquire()
                try:
                    self._forward(path, raw, body, backend, started, queued_s)
                    return
                except _BackendBusy as exc:
                    # Already counted as a rejection by _forward. Recording it
                    # as a failure here as well would open the circuit on a
                    # backend that is healthy and merely full, removing
                    # capacity at exactly the wrong moment.
                    if attempt == attempts - 1:
                        return self._fail(path, 503, str(exc), started)
                    continue
                except _RetryableError as exc:
                    backend.record_failure(str(exc))
                    if attempt == attempts - 1:
                        return self._fail(
                            path, 502, f"all {len(tried)} backend(s) failed: {exc}", started
                        )
                    # Nothing has been written to the client yet, so another
                    # backend can still serve this request cleanly.
                    continue
                except _StreamBroken as exc:
                    # Past the point of no return: the client already holds
                    # part of a 200 response. Record it and stop.
                    backend.record_failure(str(exc))
                    gateway.log.record(
                        {
                            "path": path,
                            "backend": backend.url,
                            "status": 200,
                            "truncated": True,
                            "error": str(exc),
                            "duration_s": round(time.perf_counter() - started, 4),
                        }
                    )
                    return
                finally:
                    backend.release()

        def _forward(
            self,
            path: str,
            raw: bytes,
            body: dict,
            backend: Backend,
            started: float,
            queued_s: float,
        ) -> None:
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
            if config.backend_api_key:
                headers["Authorization"] = f"Bearer {config.backend_api_key}"
            request = urllib.request.Request(backend.url + path, data=raw, headers=headers)

            try:
                upstream = urllib.request.urlopen(request, timeout=config.request_timeout_s)  # noqa: S310
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code == 503:
                    # The backend is shedding load, not broken. Try another one
                    # if there is one, but do not count it against its health.
                    backend.record_rejection()
                    raise _BackendBusy(f"{backend.url} at capacity") from None
                if exc.code >= 500:
                    raise _RetryableError(f"{backend.url} returned {exc.code}: {detail}") from None
                # 4xx is the caller's fault and will fail identically on every
                # backend. Pass it straight through rather than retrying it.
                return self._relay_error(exc.code, detail, path, backend, started)
            except (urllib.error.URLError, TimeoutError) as exc:
                raise _RetryableError(f"{backend.url} unreachable: {exc}") from None

            with upstream:
                if body.get("stream"):
                    self._stream(upstream, path, backend, started, queued_s)
                else:
                    self._blocking(upstream, path, backend, started, queued_s)

        def _stream(
            self, upstream: Any, path: str, backend: Backend, started: float, queued_s: float
        ) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            prompt_tokens = completion_tokens = 0
            ttft: float | None = None
            # An SSE frame is "data: ...\n" followed by a blank line, so the
            # terminator arrives as two chunks. A client that stops reading the
            # moment it sees [DONE] -- which is the correct thing for a client
            # to do -- disconnects while that trailing newline is still being
            # written. That is a completed stream, not a truncated one, and
            # treating it as a failure would mark every healthy backend as
            # failing and open its circuit.
            finished = False
            try:
                for chunk in upstream:
                    if ttft is None and b'"content"' in chunk:
                        ttft = time.perf_counter() - started
                    # Forwarded first, inspected second. Accounting must never
                    # sit between the backend and the user.
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    if b"[DONE]" in chunk:
                        # `data: [DONE]` is the end of an OpenAI SSE stream.
                        # Stopping here rather than reading on to EOF matters:
                        # a backend keeping the connection alive for reuse will
                        # never send one, and a gateway that waits for it holds
                        # the slot, the socket and the thread until a timeout.
                        finished = True
                        self.wfile.write(b"\n")  # complete the final frame
                        self.wfile.flush()
                        break
                    usage = _sniff_usage(chunk)
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens) or prompt_tokens
                        completion_tokens = (
                            usage.get("completion_tokens", completion_tokens) or completion_tokens
                        )
            except (BrokenPipeError, ConnectionResetError):
                if not finished:
                    # A genuine cancellation: the caller went away mid-generation.
                    raise _StreamBroken("client disconnected mid-stream") from None
            except Exception as exc:  # noqa: BLE001
                raise _StreamBroken(f"upstream failed mid-stream: {exc}") from None

            duration = time.perf_counter() - started
            backend.record_success(duration)
            gateway.log.record(
                {
                    "path": path,
                    "backend": backend.url,
                    "status": 200,
                    "stream": True,
                    "queued_s": round(queued_s, 4),
                    "ttft_s": round(ttft, 4) if ttft else None,
                    "duration_s": round(duration, 4),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )

        def _blocking(
            self, upstream: Any, path: str, backend: Backend, started: float, queued_s: float
        ) -> None:
            payload = upstream.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

            duration = time.perf_counter() - started
            backend.record_success(duration)
            usage = {}
            with contextlib.suppress(json.JSONDecodeError):
                usage = (json.loads(payload) or {}).get("usage") or {}
            gateway.log.record(
                {
                    "path": path,
                    "backend": backend.url,
                    "status": 200,
                    "stream": False,
                    "queued_s": round(queued_s, 4),
                    "duration_s": round(duration, 4),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                }
            )

        def _relay_error(
            self, status: int, detail: str, path: str, backend: Backend, started: float
        ) -> None:
            gateway.log.record(
                {
                    "path": path,
                    "backend": backend.url,
                    "status": status,
                    "duration_s": round(time.perf_counter() - started, 4),
                }
            )
            data = detail.encode() if detail else b'{"error":{"message":"upstream error"}}'
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _fail(self, path: str, status: int, message: str, started: float) -> None:
            gateway.log.record(
                {
                    "path": path,
                    "status": status,
                    "error": message,
                    "duration_s": round(time.perf_counter() - started, 4),
                }
            )
            self._error(status, message)

        # ------------------------------------------------------------ util
        def _authorised(self) -> bool:
            if not config.api_key:
                return True
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {config.api_key}"

        def _json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, message: str, headers: dict | None = None) -> None:
            data = json.dumps({"error": {"message": message, "type": "gateway_error"}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

    return Handler


class _RetryableError(Exception):
    """Failed before anything reached the client -- another backend may work."""


class _BackendBusy(_RetryableError):
    """The backend refused for capacity, not because it is broken.

    Retryable on another backend, but it must not count against this one's
    health: a full KV cache is the server working as designed.
    """


class _StreamBroken(Exception):
    """Failed after the response started -- retrying would corrupt the stream."""


def _sniff_usage(chunk: bytes) -> dict | None:
    """Pull the usage block out of an SSE frame without buffering the stream."""
    if b'"usage"' not in chunk:
        return None
    line = chunk.decode("utf-8", errors="replace").strip()
    if not line.startswith("data:"):
        return None
    try:
        return json.loads(line[5:].strip()).get("usage")
    except (json.JSONDecodeError, AttributeError):
        return None
