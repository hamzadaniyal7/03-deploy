"""A client for OpenAI-compatible inference servers, standard library only.

vLLM, SGLang, TGI, llama.cpp and Ollama all expose `/v1/chat/completions` with
the same request and SSE response shape. That compatibility is the single most
useful property of the ecosystem: it means the gateway, the benchmark and your
application all speak one protocol, and swapping the engine underneath is a URL
change.

So this depends on nothing. `openai` would be a hundred megabytes of
dependency to POST JSON and split on newlines, and the moment a benchmark
depends on a client library it starts measuring that library -- its connection
pooling, its retries, its JSON parser -- alongside the server.

**On timing.** `time.perf_counter()` is read the instant a chunk arrives, not
after parsing it. Parsing is microseconds and token gaps are milliseconds, but
the ordering is free and the habit matters: a benchmark that times its own
bookkeeping is a benchmark that reports its own bookkeeping.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from selfserve.types import RequestResult

USER_AGENT = "selfserve/0.1"


@dataclass(slots=True)
class ChatRequest:
    """One chat completion."""

    messages: list[dict[str, str]]
    model: str = "default"
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = True
    extra: dict[str, Any] | None = None

    @classmethod
    def of(cls, prompt: str, **kw: Any) -> ChatRequest:
        return cls(messages=[{"role": "user", "content": prompt}], **kw)

    def payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": self.stream,
        }
        if self.stream:
            # Without this, most servers omit usage from the stream entirely
            # and the benchmark has to guess token counts. Servers that do not
            # recognise it ignore it.
            body["stream_options"] = {"include_usage": True}
        if self.extra:
            body.update(self.extra)
        return body


class InferenceClient:
    """Talks to one OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        connect_timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.connect_timeout = connect_timeout

    # ------------------------------------------------------------------ api
    def chat(self, request: ChatRequest, *, index: int = 0) -> RequestResult:
        """Send one request and measure it.

        Never raises for a server-side failure: a 503 under load is a
        measurement, not an exception. A benchmark that stops at the first
        error cannot measure the thing you most want to know, which is what
        happens when the server is overloaded.
        """
        result = RequestResult(index=index, backend=self.base_url)
        started = time.perf_counter()
        try:
            if request.stream:
                self._stream(request, result, started)
            else:
                self._blocking(request, result, started)
        except urllib.error.HTTPError as exc:
            result.ok = False
            result.status = exc.code
            result.error = f"HTTP {exc.code}: {_read_error(exc)}"
        except urllib.error.URLError as exc:
            result.ok = False
            result.error = f"connection failed: {exc.reason}"
        except TimeoutError:
            result.ok = False
            result.error = f"timed out after {self.timeout:g}s"
        except Exception as exc:  # noqa: BLE001 - a client bug is still a datum
            result.ok = False
            result.error = f"{type(exc).__name__}: {exc}"

        result.total_s = time.perf_counter() - started
        return result

    def models(self) -> list[str]:
        """`GET /v1/models` -- what this server will answer to."""
        with self._open("/v1/models", None, timeout=self.connect_timeout) as response:
            data = json.loads(response.read())
        return [entry["id"] for entry in data.get("data", []) if "id" in entry]

    def health(self) -> tuple[bool, str]:
        """Liveness, by whichever endpoint the server offers.

        vLLM has `/health`, SGLang has `/health` and `/health_generate`, others
        have neither. `/v1/models` works everywhere and is the fallback, which
        is also a slightly better check: it proves the model is loaded rather
        than that the process is up.
        """
        for path in ("/health", "/v1/models"):
            try:
                with self._open(path, None, timeout=self.connect_timeout) as response:
                    if 200 <= response.status < 300:
                        return True, f"{path} -> {response.status}"
            except urllib.error.HTTPError as exc:
                if path == "/v1/models":
                    return False, f"{path} -> HTTP {exc.code}"
            except urllib.error.URLError as exc:
                return False, f"unreachable: {exc.reason}"
            except Exception as exc:  # noqa: BLE001
                return False, f"{type(exc).__name__}: {exc}"
        return False, "no health endpoint responded"

    # ------------------------------------------------------------ internals
    def _stream(self, request: ChatRequest, result: RequestResult, started: float) -> None:
        with self._open("/v1/chat/completions", request.payload()) as response:
            result.status = response.status
            chunks: list[str] = []
            for event, at in _sse_events(response):
                if event == "[DONE]":
                    break
                try:
                    data = json.loads(event)
                except json.JSONDecodeError:
                    continue  # keep-alive comment or a partial frame

                usage = data.get("usage")
                if usage:
                    # Usage arrives in a final chunk with no choices. Taking it
                    # from the server rather than counting characters is the
                    # difference between measured and estimated throughput.
                    result.prompt_tokens = usage.get("prompt_tokens", 0) or 0
                    result.completion_tokens = usage.get("completion_tokens", 0) or 0

                for choice in data.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")
                    if not piece:
                        continue
                    if result.ttft_s is None:
                        result.ttft_s = at - started
                    result.token_times.append(at - started)
                    chunks.append(piece)

            result.text = "".join(chunks)

        if not result.completion_tokens:
            # The server did not report usage. Chunk count is the best
            # available proxy; it is close for most tokenizers and this is
            # flagged in the report rather than passed off as measured.
            result.completion_tokens = len(result.token_times)

    def _blocking(self, request: ChatRequest, result: RequestResult, started: float) -> None:
        with self._open("/v1/chat/completions", request.payload()) as response:
            result.status = response.status
            data = json.loads(response.read())
        usage = data.get("usage") or {}
        result.prompt_tokens = usage.get("prompt_tokens", 0) or 0
        result.completion_tokens = usage.get("completion_tokens", 0) or 0
        choices = data.get("choices") or [{}]
        result.text = (choices[0].get("message") or {}).get("content", "")
        # With no stream there is no first-token moment; the whole response
        # arrives at once. Recording TTFT = total would be a lie that makes
        # non-streaming numbers look comparable with streaming ones.
        result.ttft_s = None

    def _open(self, path: str, body: dict[str, Any] | None, timeout: float | None = None):
        headers = {"User-Agent": USER_AGENT, "Accept": "text/event-stream, application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)  # noqa: S310


def _sse_events(response: Any) -> Iterator[tuple[str, float]]:
    """Yield `(payload, arrival_time)` for each `data:` line.

    Server-sent events are newline-delimited frames; `data: ` prefixes the
    payload and a blank line ends a frame. Reading line by line from the raw
    socket keeps arrival times honest -- buffering the whole body first and
    splitting afterwards would record every token as arriving at once, which
    is the classic way to produce a flat and completely wrong ITL curve.
    """
    for raw in response:
        at = time.perf_counter()
        line = raw.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            yield line[5:].strip(), at


def _read_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")[:400]
    except Exception:  # noqa: BLE001
        return exc.reason or ""
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                return str(error.get("message", body))
            if error:
                return str(error)
    except json.JSONDecodeError:
        pass
    return body


Prompt = Callable[[int], str]
