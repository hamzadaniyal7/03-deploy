"""Percentiles and per-request derivations."""

from __future__ import annotations

from selfserve.types import Percentiles, RequestResult


def test_percentiles_of_a_known_distribution() -> None:
    p = Percentiles.of([float(i) for i in range(1, 101)])
    assert p.count == 100
    assert p.minimum == 1.0 and p.maximum == 100.0
    assert 50 <= p.p50 <= 51
    assert 99 <= p.p99 <= 100


def test_percentiles_of_nothing_do_not_explode() -> None:
    p = Percentiles.of([])
    assert p.count == 0 and p.p99 == 0.0


def test_percentiles_of_one_value() -> None:
    """statistics.quantiles raises here; a benchmark with one request should
    still report."""
    p = Percentiles.of([4.2])
    assert p.p50 == p.p99 == p.maximum == 4.2


def test_a_tail_moves_p99_but_barely_moves_p50() -> None:
    """The reason a mean is never reported alone.

    5% of requests at 100x the latency leaves p50 untouched, drags the mean
    to a value almost no request experienced, and shows up properly only in
    p95 and p99 -- which is the whole argument for reporting them.
    """
    p = Percentiles.of([0.1] * 95 + [10.0] * 5)
    assert p.p50 == 0.1
    assert p.p90 == 0.1
    assert p.p99 == 10.0
    assert p.mean > 5 * p.p50  # the mean has already left the data


def test_one_outlier_in_a_hundred_is_interpolated_not_rounded_up() -> None:
    """A single slow request does not make p99 equal to the maximum: p99 sits
    between the 98th and 99th samples. Tools that report the max and call it
    p99 overstate the tail on small samples."""
    p = Percentiles.of([0.1] * 99 + [10.0])
    assert p.p50 == 0.1
    assert p.p50 < p.p99 < p.maximum


def test_percentiles_are_order_independent() -> None:
    assert Percentiles.of([3.0, 1.0, 2.0]).p50 == Percentiles.of([1.0, 2.0, 3.0]).p50


# ------------------------------------------------------------ RequestResult
def test_inter_token_latencies_are_gaps_not_timestamps() -> None:
    r = RequestResult(index=0, token_times=[0.1, 0.2, 0.35])
    assert r.inter_token_latencies == [
        pytest_approx(0.1),
        pytest_approx(0.15),
    ]


def pytest_approx(value: float, tol: float = 1e-9):
    class _Approx(float):
        def __eq__(self, other):  # noqa: D105
            return abs(float(self) - other) < tol

    return _Approx(value)


def test_a_single_token_has_no_inter_token_latency() -> None:
    assert RequestResult(index=0, token_times=[0.1]).inter_token_latencies == []


def test_output_rate_excludes_prefill() -> None:
    """Including TTFT in the decode rate makes a slow prefill look like slow
    decode, which points tuning at the wrong half of the system."""
    r = RequestResult(index=0, ttft_s=1.0, total_s=3.0, completion_tokens=21)
    assert r.output_tokens_per_second == 10.0  # 20 tokens over 2 seconds


def test_output_rate_is_zero_without_a_first_token() -> None:
    assert RequestResult(index=0, total_s=1.0, completion_tokens=10).output_tokens_per_second == 0.0


def test_output_rate_is_zero_for_a_one_token_response() -> None:
    r = RequestResult(index=0, ttft_s=0.5, total_s=0.5, completion_tokens=1)
    assert r.output_tokens_per_second == 0.0


def test_serialisation_drops_the_bulky_fields() -> None:
    r = RequestResult(
        index=0, ttft_s=0.1, total_s=1.0, completion_tokens=5, token_times=[0.1] * 5, text="hello"
    )
    data = r.to_dict()
    assert "token_times" not in data and "text" not in data
    assert "output_tokens_per_second" in data
