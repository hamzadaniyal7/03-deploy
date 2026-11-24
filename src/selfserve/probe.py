"""Does this server actually behave like an OpenAI-compatible endpoint?

"OpenAI-compatible" is a spectrum. Every serving stack claims it; they differ
in which optional pieces they implement, and the differences only surface once
something downstream depends on them. The three that bite in practice:

* **Usage in streaming responses.** Without `stream_options.include_usage`
  honoured, a streaming response carries no token counts, so every piece of
  accounting and every throughput measurement downstream is an estimate from
  chunk counts. vLLM supports it; several others ignore it silently.
* **A health endpoint.** Load balancers and orchestrators need one. `/health`
  is conventional, not universal, and `/v1/models` is the portable fallback.
* **Behaviour at capacity.** A 503 with `Retry-After` is a server telling you
  to back off. Silently queueing until the client times out looks identical
  from the outside until you measure it, and the difference decides whether
  your retry policy helps or makes things worse.

`selfserve probe` checks these before you build on them, rather than after.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from selfserve.client import ChatRequest, InferenceClient


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str = ""
    advisory: bool = False  # a warning, not a failure


@dataclass(slots=True)
class ProbeReport:
    url: str
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks if not c.advisory)

    def render(self) -> str:
        lines = [f"probing {self.url}", ""]
        for check in self.checks:
            mark = "ok  " if check.ok else ("warn" if check.advisory else "FAIL")
            lines.append(f"  [{mark}] {check.name:<28} {check.detail}")
        lines.append("")
        lines.append("  usable" if self.passed else "  NOT usable -- see the failures above")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "passed": self.passed,
            "checks": [
                {"name": c.name, "ok": c.ok, "detail": c.detail, "advisory": c.advisory}
                for c in self.checks
            ],
        }


def probe(url: str, *, api_key: str | None = None, model: str = "default") -> ProbeReport:
    """Run the compatibility checks against one endpoint."""
    client = InferenceClient(url, api_key=api_key, timeout=60.0)
    report = ProbeReport(url=url)

    ok, detail = client.health()
    report.checks.append(Check("health endpoint", ok, detail))
    if not ok:
        # Nothing else is meaningful if it is not answering at all.
        return report

    try:
        names = client.models()
        report.checks.append(
            Check("GET /v1/models", bool(names), ", ".join(names) or "returned an empty list")
        )
        if names and model == "default":
            model = names[0]
    except Exception as exc:  # noqa: BLE001
        report.checks.append(Check("GET /v1/models", False, f"{type(exc).__name__}: {exc}"))

    blocking = client.chat(ChatRequest.of("Say OK.", model=model, max_tokens=8, stream=False))
    report.checks.append(
        Check(
            "non-streaming completion",
            blocking.ok,
            f"{blocking.completion_tokens} tokens in {blocking.total_s:.2f}s"
            if blocking.ok
            else blocking.error,
        )
    )

    started = time.perf_counter()
    stream = client.chat(ChatRequest.of("Count to five.", model=model, max_tokens=24))
    report.checks.append(
        Check(
            "streaming completion",
            stream.ok and bool(stream.token_times),
            f"{len(stream.token_times)} chunks, first at {stream.ttft_s:.3f}s"
            if stream.ok and stream.ttft_s is not None
            else (stream.error or "no chunks received"),
        )
    )

    # Did the server report usage, or did the client have to count chunks?
    reported_usage = stream.ok and stream.prompt_tokens > 0
    report.checks.append(
        Check(
            "usage in stream",
            reported_usage,
            "token counts came from the server"
            if reported_usage
            else "no usage block: token counts downstream will be estimates",
            advisory=True,
        )
    )

    if stream.ok and len(stream.token_times) > 2:
        spread = stream.token_times[-1] - stream.token_times[0]
        # If every chunk lands at once the response was buffered somewhere --
        # usually a reverse proxy without `proxy_buffering off`. The endpoint
        # works and streaming does not, which is invisible until a user
        # complains that the UI shows nothing for ten seconds.
        incremental = spread > 1e-3
        report.checks.append(
            Check(
                "chunks arrive incrementally",
                incremental,
                f"{spread * 1000:.0f}ms between first and last chunk"
                if incremental
                else "all chunks arrived together -- something is buffering the stream",
                advisory=True,
            )
        )

    report.checks.append(Check("round trip", True, f"{time.perf_counter() - started:.2f}s"))
    return report
