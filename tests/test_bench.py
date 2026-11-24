"""The load generator and the report it produces."""

from __future__ import annotations

import random

from selfserve.bench import BenchConfig, build_prompt, run_bench, run_sweep
from selfserve.bench.report import BenchReport, Sweep
from selfserve.mock import MockConfig, MockServer
from selfserve.types import RequestResult


def config(url: str, **kw) -> BenchConfig:
    base = {"concurrency": 4, "requests": 8, "prompt_tokens": 64, "max_tokens": 6, "warmup": 1}
    return BenchConfig(url=url, **{**base, **kw})


# ---------------------------------------------------------------- prompts
def test_prompt_length_scales_with_the_request() -> None:
    rng = random.Random(1)
    assert len(build_prompt(512, rng)) > 5 * len(build_prompt(64, rng))


def test_prompts_are_built_from_words_not_repeated_characters() -> None:
    """Tokenizers collapse "aaaa..." into very few tokens, so a prompt built
    that way never pays the prefill cost the benchmark is trying to measure."""
    prompt = build_prompt(128, random.Random(1))
    assert len(set(prompt.split())) > 5


def test_prompts_are_reproducible_for_a_seed() -> None:
    assert build_prompt(64, random.Random(3)) == build_prompt(64, random.Random(3))


# ------------------------------------------------------------------- runs
def test_a_run_completes_every_request(mock: MockServer) -> None:
    report = run_bench(config(mock.url))
    assert len(report.results) == 8
    assert len(report.completed) == 8
    assert not report.failed


def test_throughput_is_positive(mock: MockServer) -> None:
    report = run_bench(config(mock.url))
    assert report.requests_per_second > 0
    assert report.output_tokens_per_second > 0
    assert report.total_tokens_per_second > report.output_tokens_per_second


def test_latency_percentiles_are_populated(mock: MockServer) -> None:
    report = run_bench(config(mock.url))
    assert report.ttft().count == 8
    assert report.itl().count > 8  # pooled across tokens, not requests
    assert report.latency().p50 > 0


def test_warmup_requests_are_excluded(mock: MockServer) -> None:
    """A first request pays for cold caches and can be ten times slower;
    including it is how a p99 becomes a problem that does not exist."""
    report = run_bench(config(mock.url, warmup=3, requests=5))
    assert len(report.results) == 5


def test_non_streaming_runs_report_no_ttft(mock: MockServer) -> None:
    report = run_bench(config(mock.url, stream=False))
    assert report.completed
    assert report.ttft().count == 0
    assert report.latency().p50 > 0


def test_errors_are_measured_not_raised() -> None:
    with MockServer(MockConfig(failure_rate=1.0)) as server:
        report = run_bench(config(server.url, warmup=0))
    assert not report.completed
    assert len(report.errored) == 8
    assert "no completed requests" in report.render()


def test_a_dead_endpoint_does_not_crash_the_run() -> None:
    report = run_bench(config("http://127.0.0.1:9", warmup=0, requests=3))
    assert len(report.errored) == 3
    assert "connection failed" in report.render()


def test_load_shedding_is_counted_separately_from_errors() -> None:
    """A 503 is the system working; a 500 is a bug. Lumping them together
    hides which one is happening."""
    slow = MockConfig(capacity=1, base_ttft_s=0.02, base_itl_s=0.01)
    with MockServer(slow) as server:
        report = run_bench(config(server.url, concurrency=6, requests=12, warmup=0))
    assert report.rejected
    assert not report.errored
    assert "load shed" in report.render()


# ----------------------------------------------------------------- sweep
def test_a_sweep_covers_every_level(mock: MockServer) -> None:
    sweep = run_sweep(config(mock.url, requests=8, concurrency=4), [1, 2, 4])
    assert [r.concurrency for r in sweep.reports] == [1, 2, 4]
    assert all(r.completed for r in sweep.reports)


def test_a_sweep_scales_request_count_with_concurrency() -> None:
    """Holding requests fixed makes the high-concurrency levels finish before
    the scheduler reaches a steady state, and a throughput number measured
    during ramp-up is a number about ramp-up."""
    with MockServer(MockConfig(base_ttft_s=0.002, base_itl_s=0.001)) as server:
        sweep = run_sweep(config(server.url, requests=8, concurrency=4), [1, 4])
    assert len(sweep.reports[1].results) > len(sweep.reports[0].results)


def test_concurrency_raises_throughput_and_latency_together() -> None:
    """The trade-off the whole tool exists to show.

    Its own slower mock rather than the shared fast one: measuring a
    throughput ratio over thirty milliseconds is measuring scheduler noise.
    At 5ms per token each level runs for a few hundred milliseconds, which
    is long enough for the comparison to mean something on any machine.
    """
    config_slow = MockConfig(base_ttft_s=0.005, base_itl_s=0.005, batch_penalty=0.05, capacity=16)
    with MockServer(config_slow) as server:
        sweep = run_sweep(config(server.url, requests=16, concurrency=4, max_tokens=20), [1, 8])
    low, high = sweep.reports
    assert high.output_tokens_per_second > low.output_tokens_per_second
    assert high.ttft().p99 >= low.ttft().p99


def test_sweep_render_names_the_peak(mock: MockServer) -> None:
    sweep = run_sweep(config(mock.url, requests=8, concurrency=4), [1, 4])
    rendered = sweep.render()
    assert "peak throughput at concurrency" in rendered
    assert "conc" in rendered and "out tok/s" in rendered


def test_a_single_level_sweep_says_it_cannot_show_a_curve(mock: MockServer) -> None:
    sweep = run_sweep(config(mock.url, requests=4, concurrency=2), [2])
    assert "need at least two concurrency levels" in sweep.render()


def test_sweep_serialises_without_per_request_detail(mock: MockServer) -> None:
    import json

    sweep = run_sweep(config(mock.url, requests=4, concurrency=2), [1, 2])
    data = json.loads(sweep.to_json())
    assert len(data["sweep"]) == 2
    assert "requests" not in data["sweep"][0]


# ---------------------------------------------------------------- report
def test_report_counts_without_any_server() -> None:
    results = [
        RequestResult(index=0, ok=True, status=200, ttft_s=0.1, total_s=1.0, completion_tokens=10),
        RequestResult(index=1, ok=False, status=503, error="at capacity"),
        RequestResult(index=2, ok=False, status=500, error="boom"),
    ]
    report = BenchReport(concurrency=1, duration_s=2.0, results=results)
    assert len(report.completed) == 1
    assert len(report.rejected) == 1
    assert len(report.errored) == 1
    assert report.output_tokens_per_second == 5.0


def test_report_serialises(mock: MockServer) -> None:
    import json

    data = json.loads(json.dumps(run_bench(config(mock.url)).to_dict()))
    assert data["counts"]["completed"] == 8
    assert data["throughput"]["output_tokens_per_second"] > 0
    assert len(data["requests"]) == 8


def test_identical_errors_are_grouped_in_the_report() -> None:
    results = [RequestResult(index=i, ok=False, status=500, error="boom") for i in range(5)]
    rendered = BenchReport(concurrency=1, duration_s=1.0, results=results).render()
    assert "5x" in rendered


def test_empty_sweep_renders() -> None:
    assert "need at least two" in Sweep().render()
