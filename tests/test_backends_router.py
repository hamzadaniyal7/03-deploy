"""Circuit breaking, and choosing where a request goes."""

from __future__ import annotations

import time

import pytest

from selfserve.gateway.backends import Backend, BackendPool
from selfserve.gateway.router import NoBackendAvailable, Router


def backend(url: str = "http://a", **kw) -> Backend:
    return Backend(url, **kw)


# ------------------------------------------------------------------- health
def test_a_new_backend_is_healthy() -> None:
    assert backend().healthy


def test_failures_below_the_threshold_do_not_open_the_circuit() -> None:
    b = backend(failure_threshold=3)
    b.record_failure("boom")
    b.record_failure("boom")
    assert b.healthy


def test_the_threshold_opens_the_circuit() -> None:
    b = backend(failure_threshold=2)
    b.record_failure("boom")
    b.record_failure("boom")
    assert not b.healthy
    assert b.last_error == "boom"


def test_a_success_resets_the_failure_count() -> None:
    b = backend(failure_threshold=2)
    b.record_failure("boom")
    b.record_success(0.1)
    b.record_failure("boom")
    assert b.healthy


def test_the_circuit_closes_after_the_recovery_window() -> None:
    """Half-open: the next request is a probe, so a backend that is still
    broken costs one request per window rather than all of them."""
    b = backend(failure_threshold=1, recovery_s=0.05)
    b.record_failure("boom")
    assert not b.healthy
    time.sleep(0.08)
    assert b.healthy


def test_a_503_is_load_shedding_not_a_fault() -> None:
    """Counting it as a failure would remove a healthy backend from the pool
    at exactly the moment there is not enough capacity."""
    b = backend(failure_threshold=1)
    b.record_rejection()
    b.record_rejection()
    assert b.healthy
    assert b.snapshot()["rejected"] == 2


def test_in_flight_tracking() -> None:
    b = backend()
    b.acquire()
    b.acquire()
    assert b.in_flight == 2
    b.release()
    assert b.in_flight == 1


def test_in_flight_never_goes_negative() -> None:
    b = backend()
    b.release()
    assert b.in_flight == 0


def test_stats_record_latency() -> None:
    b = backend()
    b.record_success(0.2)
    b.record_success(0.4)
    assert b.stats.requests == 2
    assert abs(b.snapshot()["mean_latency_s"] - 0.3) < 1e-6


# ------------------------------------------------------------------- router
def test_picks_the_least_loaded() -> None:
    """The one routing decision that is genuinely different for LLM traffic:
    request costs differ by orders of magnitude, so counting requests sent
    (round robin) is not the same as counting requests still running."""
    busy, idle = backend("http://busy"), backend("http://idle")
    for _ in range(5):
        busy.acquire()
    router = Router(BackendPool([busy, idle]))
    assert router.pick().url == "http://idle"


def test_ties_are_broken_randomly_not_by_order() -> None:
    """Deterministic tie-breaking sends every request to the first backend
    whenever the pool is idle, which is most of the time at low traffic."""
    pool = BackendPool([backend("http://a"), backend("http://b")])
    picks = {Router(pool, seed=s).pick().url for s in range(20)}
    assert picks == {"http://a", "http://b"}


def test_unhealthy_backends_are_skipped() -> None:
    sick, well = backend("http://sick", failure_threshold=1), backend("http://well")
    sick.record_failure("boom")
    router = Router(BackendPool([sick, well]))
    assert router.pick().url == "http://well"


def test_already_tried_backends_are_excluded() -> None:
    a, b = backend("http://a"), backend("http://b")
    router = Router(BackendPool([a, b]))
    assert router.pick(exclude={"http://a"}).url == "http://b"


def test_unhealthy_backends_are_a_last_resort_not_a_dead_end() -> None:
    """A circuit opened by a transient blip is better probed with a real
    request than left idle while the caller gets an error."""
    sick = backend("http://sick", failure_threshold=1)
    sick.record_failure("boom")
    router = Router(BackendPool([sick]))
    assert router.pick().url == "http://sick"


def test_no_backends_at_all_raises() -> None:
    with pytest.raises(NoBackendAvailable):
        Router(BackendPool([])).pick()


def test_every_backend_already_tried_raises() -> None:
    router = Router(BackendPool([backend("http://a")]))
    with pytest.raises(NoBackendAvailable, match="already tried"):
        router.pick(exclude={"http://a"})


def test_pool_filters_blank_urls() -> None:
    assert len(BackendPool.of(["http://a", "", "  "])) == 1
