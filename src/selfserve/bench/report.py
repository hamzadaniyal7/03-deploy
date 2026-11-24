"""Turning a pile of requests into the numbers worth reporting.

The guiding rule: **never report a mean on its own.** Inference latency is
long-tailed by construction -- a request that queued behind a 32k prefill waits
an order of magnitude longer than one that did not -- so a mean sits in a gap
where almost no request actually landed. Two servers with identical means can
have p99s a factor of ten apart, and the p99 is what your users complain about.

The second rule: **separate the server's throughput from the user's latency.**
They move in opposite directions as batch size grows, and a report that
collapses them into one "performance" number cannot show the trade-off that is
the entire subject of tuning an inference server.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from selfserve.types import Percentiles, RequestResult


@dataclass(slots=True)
class BenchReport:
    """One benchmark run at one concurrency level."""

    concurrency: int
    duration_s: float
    results: list[RequestResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------- counts
    @property
    def completed(self) -> list[RequestResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> list[RequestResult]:
        return [r for r in self.results if not r.ok]

    @property
    def rejected(self) -> list[RequestResult]:
        """Shed load: the server said "at capacity" rather than falling over.

        Counted separately from errors on purpose. A 503 under load is the
        system working as designed; a 500 is a bug, and a timeout is a request
        that consumed capacity and produced nothing. Lumping them together
        hides which of the three is happening.
        """
        return [r for r in self.results if r.status == 503]

    @property
    def errored(self) -> list[RequestResult]:
        return [r for r in self.failed if r.status != 503]

    # ------------------------------------------------------------ aggregates
    @property
    def requests_per_second(self) -> float:
        return len(self.completed) / self.duration_s if self.duration_s else 0.0

    @property
    def output_tokens_per_second(self) -> float:
        """The server's decode throughput -- the number that pays for the GPU."""
        total = sum(r.completion_tokens for r in self.completed)
        return total / self.duration_s if self.duration_s else 0.0

    @property
    def total_tokens_per_second(self) -> float:
        total = sum(r.prompt_tokens + r.completion_tokens for r in self.completed)
        return total / self.duration_s if self.duration_s else 0.0

    def ttft(self) -> Percentiles:
        return Percentiles.of([r.ttft_s for r in self.completed if r.ttft_s is not None])

    def itl(self) -> Percentiles:
        """Pooled across every token of every request.

        Pooled rather than per-request-then-averaged because a stall is a
        property of a moment, not of a request: averaging within a request
        first would smooth exactly the event worth seeing.
        """
        pooled: list[float] = []
        for result in self.completed:
            pooled.extend(result.inter_token_latencies)
        return Percentiles.of(pooled)

    def latency(self) -> Percentiles:
        return Percentiles.of([r.total_s for r in self.completed])

    def per_stream_tps(self) -> Percentiles:
        return Percentiles.of(
            [r.output_tokens_per_second for r in self.completed if r.output_tokens_per_second]
        )

    # -------------------------------------------------------------- display
    def render(self) -> str:
        lines = [
            f"concurrency {self.concurrency}  --  {len(self.results)} requests "
            f"in {self.duration_s:.1f}s",
            "",
            f"  completed              {len(self.completed):>8}",
        ]
        if self.rejected:
            lines.append(f"  rejected (503)         {len(self.rejected):>8}   load shed")
        if self.errored:
            lines.append(f"  errors                 {len(self.errored):>8}")
        lines += [
            "",
            f"  requests/s             {self.requests_per_second:>8.2f}",
            f"  output tokens/s        {self.output_tokens_per_second:>8.1f}   server throughput",
            f"  total tokens/s         {self.total_tokens_per_second:>8.1f}   prompt + output",
            "",
        ]
        if not self.completed:
            lines.append("  no completed requests -- nothing to summarise")
            return "\n".join(lines + self._error_lines())

        lines += [
            self.ttft().render("time to first token", "ms", 1000),
            self.itl().render("inter-token latency", "ms", 1000),
            self.latency().render("end-to-end", "ms", 1000),
            self.per_stream_tps().render("per-stream tok/s", "t/s", 1),
        ]
        return "\n".join(lines + self._error_lines())

    def _error_lines(self) -> list[str]:
        if not self.errored:
            return []
        seen: dict[str, int] = {}
        for result in self.errored:
            seen[result.error[:100]] = seen.get(result.error[:100], 0) + 1
        out = ["", "  errors:"]
        out.extend(f"    {count:>4}x  {message}" for message, count in sorted(seen.items()))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "duration_s": round(self.duration_s, 3),
            "config": self.config,
            "counts": {
                "total": len(self.results),
                "completed": len(self.completed),
                "rejected_503": len(self.rejected),
                "errored": len(self.errored),
            },
            "throughput": {
                "requests_per_second": round(self.requests_per_second, 3),
                "output_tokens_per_second": round(self.output_tokens_per_second, 2),
                "total_tokens_per_second": round(self.total_tokens_per_second, 2),
            },
            "ttft_s": self.ttft().to_dict(),
            "itl_s": self.itl().to_dict(),
            "latency_s": self.latency().to_dict(),
            "per_stream_tokens_per_second": self.per_stream_tps().to_dict(),
            "requests": [r.to_dict() for r in self.results],
        }


@dataclass(slots=True)
class Sweep:
    """The same workload at several concurrency levels.

    This is the shape worth looking at. A single concurrency number tells you
    what the server did once; the sweep tells you where throughput stops rising
    and latency starts running away, which is the only thing you can actually
    make a deployment decision from.
    """

    reports: list[BenchReport] = field(default_factory=list)

    def render(self) -> str:
        header = (
            f"  {'conc':>5}  {'req/s':>8}  {'out tok/s':>10}  {'TTFT p50':>9}  "
            f"{'TTFT p99':>9}  {'ITL p50':>8}  {'ITL p99':>8}  {'503s':>5}"
        )
        lines = [header, "  " + "-" * (len(header) - 2)]
        for report in self.reports:
            ttft, itl = report.ttft(), report.itl()
            lines.append(
                f"  {report.concurrency:>5}  {report.requests_per_second:>8.2f}  "
                f"{report.output_tokens_per_second:>10.1f}  "
                f"{ttft.p50 * 1000:>8.0f}m  {ttft.p99 * 1000:>8.0f}m  "
                f"{itl.p50 * 1000:>7.1f}m  {itl.p99 * 1000:>7.1f}m  "
                f"{len(report.rejected):>5}"
            )
        lines += ["", self._verdict()]
        return "\n".join(lines)

    def _verdict(self) -> str:
        """Point at the knee, since that is what the table is for."""
        usable = [r for r in self.reports if r.completed]
        if len(usable) < 2:
            return "  (need at least two concurrency levels to see a curve)"

        best = max(usable, key=lambda r: r.output_tokens_per_second)
        lines = [
            f"  peak throughput at concurrency {best.concurrency}: "
            f"{best.output_tokens_per_second:.0f} output tok/s, "
            f"TTFT p99 {best.ttft().p99 * 1000:.0f}ms"
        ]
        first = usable[0]
        if best is not first and first.ttft().p99 > 0:
            latency_cost = best.ttft().p99 / first.ttft().p99
            throughput_gain = (
                best.output_tokens_per_second / first.output_tokens_per_second
                if first.output_tokens_per_second
                else 0.0
            )
            lines.append(
                f"  versus concurrency {first.concurrency}: "
                f"{throughput_gain:.1f}x the throughput for {latency_cost:.1f}x the TTFT p99"
            )
        shedding = [r for r in self.reports if r.rejected]
        if shedding:
            lines.append(
                f"  load shedding begins at concurrency {shedding[0].concurrency} "
                f"({len(shedding[0].rejected)} rejected)"
            )
        return "\n".join(lines)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(
            {
                "sweep": [
                    {k: v for k, v in r.to_dict().items() if k != "requests"} for r in self.reports
                ]
            },
            indent=indent,
        )
