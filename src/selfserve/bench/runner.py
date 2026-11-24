"""Driving load at the server.

Threads, not asyncio. The work here is almost entirely waiting on a socket, so
both models would serve; threads win because the client is `urllib` and because
a benchmark whose own concurrency model is exotic invites the question of
whether it is measuring the server or itself. One thread per in-flight request
is the most boring possible answer, and boring is what you want underneath a
measurement.

**Open versus closed loop.** This is a *closed-loop* generator: `concurrency`
workers each send a request, wait for the full response, and send the next. It
measures what a fixed pool of callers experiences, and it has a property worth
knowing about -- when the server slows down, the offered load drops with it, so
the server is never pushed past the point where it stops keeping up. An
*open-loop* generator sends at a fixed rate regardless, which is what real
traffic does and which is how you find the cliff. Closed-loop is implemented
here because it is the right default for "how fast can this serve N users" and
because its results are stable enough to compare across runs. Open-loop arrival
is in the roadmap, and the distinction is one of the more common ways published
benchmark numbers turn out not to mean what they claim.

**Warm-up matters.** The first request to a freshly started server pays for
CUDA graph capture, weight paging and an empty prefix cache, and it can be ten
times slower than the second. Including it in a percentile is how a p99 ends up
reported as a latency problem that does not exist.
"""

from __future__ import annotations

import itertools
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from selfserve.bench.report import BenchReport, Sweep
from selfserve.client import ChatRequest, InferenceClient
from selfserve.types import RequestResult

# Filler used to build prompts of a requested token length. Real words rather
# than repeated characters: tokenizers collapse "aaaa..." into very few tokens,
# so a prompt built that way is a fraction of the length you asked for and the
# prefill cost you are trying to measure never happens.
FILLER = [
    "context",
    "window",
    "throughput",
    "latency",
    "batching",
    "scheduler",
    "cache",
    "attention",
    "kernel",
    "quantisation",
    "tensor",
    "parallel",
    "pipeline",
    "prefill",
    "decode",
    "sequence",
    "request",
    "token",
    "memory",
    "bandwidth",
    "utilisation",
    "deployment",
    "inference",
    "serving",
    "gateway",
    "capacity",
]


@dataclass(slots=True)
class BenchConfig:
    """What to send, and how hard."""

    url: str
    model: str = "default"
    concurrency: int = 8
    requests: int = 64
    prompt_tokens: int = 256
    max_tokens: int = 128
    stream: bool = True
    api_key: str | None = None
    timeout: float = 120.0
    warmup: int = 2
    seed: int = 11

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "model": self.model,
            "concurrency": self.concurrency,
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }


@dataclass(slots=True)
class Progress:
    """Live counters, so a long run is not a silent one."""

    total: int
    done: int = 0
    failed: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, result: RequestResult) -> None:
        with self._lock:
            self.done += 1
            if not result.ok:
                self.failed += 1


def build_prompt(target_tokens: int, rng: random.Random) -> str:
    """A prompt of roughly `target_tokens` tokens.

    Approximated at ~0.75 tokens per word, which is close enough for common
    English tokenizers. The exact count does not need to be right; what matters
    is that it is *consistent between runs*, so that a prefill cost measured
    today is comparable with one measured after a config change.
    """
    words = max(1, int(target_tokens / 0.75))
    return " ".join(rng.choice(FILLER) for _ in range(words))


def run_bench(
    config: BenchConfig,
    *,
    on_result: Callable[[RequestResult], None] | None = None,
) -> BenchReport:
    """Send `config.requests` requests at `config.concurrency` and measure."""
    client = InferenceClient(config.url, api_key=config.api_key, timeout=config.timeout)
    rng = random.Random(config.seed)
    prompts = [
        build_prompt(config.prompt_tokens, rng) for _ in range(min(32, config.requests or 1))
    ]
    cycle = itertools.cycle(prompts)

    def send(index: int) -> RequestResult:
        request = ChatRequest.of(
            next(cycle),
            model=config.model,
            max_tokens=config.max_tokens,
            stream=config.stream,
        )
        result = client.chat(request, index=index)
        if on_result is not None:
            on_result(result)
        return result

    # Warm-up requests are sent and their results thrown away -- deliberately
    # not merely excluded from the percentiles, because they also serve to
    # populate the prefix cache and bring the scheduler to a steady state.
    if config.warmup:
        with ThreadPoolExecutor(max_workers=min(config.warmup, config.concurrency)) as pool:
            list(
                pool.map(
                    lambda i: client.chat(
                        ChatRequest.of(
                            prompts[0], model=config.model, max_tokens=8, stream=config.stream
                        ),
                        index=-1,
                    ),
                    range(config.warmup),
                )
            )

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        results = list(pool.map(send, range(config.requests)))
    duration = time.perf_counter() - started

    return BenchReport(
        concurrency=config.concurrency,
        duration_s=duration,
        results=results,
        config=config.to_dict(),
    )


def run_sweep(
    config: BenchConfig,
    levels: list[int],
    *,
    on_report: Callable[[BenchReport], None] | None = None,
) -> Sweep:
    """The same workload at several concurrency levels.

    Request count scales with concurrency so that every level runs for a
    comparable wall-clock duration. Holding the request count fixed instead
    would make the high-concurrency levels finish in a few seconds, which is
    not long enough for the scheduler to reach a steady state -- and a
    throughput number measured during ramp-up is a number about ramp-up.
    """
    sweep = Sweep()
    per_worker = max(1, config.requests // max(1, config.concurrency))

    for level in levels:
        level_config = BenchConfig(
            **{**config.to_dict(), "concurrency": level, "requests": per_worker * level},
            api_key=config.api_key,
            timeout=config.timeout,
            warmup=config.warmup if not sweep.reports else 0,  # warm once, not per level
            seed=config.seed,
        )
        report = run_bench(level_config)
        sweep.reports.append(report)
        if on_report is not None:
            on_report(report)
    return sweep
