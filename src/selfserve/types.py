"""Shared data structures.

The benchmark's unit of measurement is one request, and `RequestResult` is
deliberately rich: every number a latency report needs is recorded per request
rather than aggregated on the fly. Aggregates computed during a run cannot be
re-sliced afterwards, and "what did the p99 look like for just the long
prompts?" is exactly the question you want to ask after a surprising result.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class RequestResult:
    """One request, start to finish.

    The three timings that matter for a streaming endpoint, and why:

    * **TTFT** (time to first token) is what a user experiences as "did it
      hear me?". It is dominated by prefill, so it scales with prompt length
      and with how many other requests are queued ahead.
    * **ITL** (inter-token latency) is what they experience as reading speed.
      It is dominated by decode, so it scales with batch size -- more
      concurrency makes every individual stream slower while making the
      server's total throughput higher.
    * **total** is what a non-streaming caller waits.

    Reporting only mean latency hides all three. A server tuned for throughput
    and a server tuned for responsiveness can post identical means.
    """

    index: int
    ok: bool = True
    status: int = 0
    error: str = ""

    queued_s: float = 0.0  # time spent waiting for a worker slot, client side
    ttft_s: float | None = None
    total_s: float = 0.0

    prompt_tokens: int = 0
    completion_tokens: int = 0
    # One timestamp per token received, relative to request start. The raw
    # material for ITL, kept because the distribution matters more than a mean
    # -- a stall halfway through a stream is invisible in an average.
    token_times: list[float] = field(default_factory=list)

    backend: str = ""
    text: str = ""

    @property
    def inter_token_latencies(self) -> list[float]:
        return [b - a for a, b in zip(self.token_times, self.token_times[1:], strict=False)]

    @property
    def output_tokens_per_second(self) -> float:
        """Decode rate for this one stream, excluding prefill."""
        if self.ttft_s is None or self.completion_tokens < 2:
            return 0.0
        decode_time = self.total_s - self.ttft_s
        return (self.completion_tokens - 1) / decode_time if decode_time > 0 else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        # The per-token timestamps are large and only needed for recomputation;
        # the report keeps the derived numbers instead.
        data.pop("token_times")
        data.pop("text")
        data["output_tokens_per_second"] = round(self.output_tokens_per_second, 2)
        return data


@dataclass(slots=True, frozen=True)
class Percentiles:
    """A distribution summary. Means are reported but never on their own."""

    count: int
    mean: float
    p50: float
    p90: float
    p95: float
    p99: float
    minimum: float
    maximum: float

    @classmethod
    def of(cls, values: list[float]) -> Percentiles:
        if not values:
            return cls(0, 0, 0, 0, 0, 0, 0, 0)
        ordered = sorted(values)
        return cls(
            count=len(ordered),
            mean=statistics.fmean(ordered),
            p50=_percentile(ordered, 50),
            p90=_percentile(ordered, 90),
            p95=_percentile(ordered, 95),
            p99=_percentile(ordered, 99),
            minimum=ordered[0],
            maximum=ordered[-1],
        )

    def render(self, label: str, unit: str = "s", scale: float = 1.0) -> str:
        return (
            f"  {label:<22} p50 {self.p50 * scale:>8.1f}  p90 {self.p90 * scale:>8.1f}  "
            f"p95 {self.p95 * scale:>8.1f}  p99 {self.p99 * scale:>8.1f}  "
            f"max {self.maximum * scale:>8.1f}  {unit}"
        )

    def to_dict(self) -> dict:
        return {k: round(v, 6) if isinstance(v, float) else v for k, v in asdict(self).items()}


def _percentile(ordered: list[float], pct: float) -> float:
    """Linear-interpolated percentile over an already-sorted list.

    `statistics.quantiles` would do this, but it needs at least two data points
    and silently uses a different interpolation method than most load-testing
    tools report. Doing it explicitly means the numbers here are comparable
    with the ones from `wrk`, `k6` and vLLM's own benchmark script.
    """
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight
