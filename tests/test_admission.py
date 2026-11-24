"""Bounded in flight, bounded waiting, fast refusal past that."""

from __future__ import annotations

import contextlib
import threading
import time

import pytest

from selfserve.gateway.admission import AdmissionControl, Rejected, slot


def test_admits_up_to_the_limit() -> None:
    admission = AdmissionControl(max_concurrency=3, queue_limit=0)
    for _ in range(3):
        admission.acquire()
    assert admission.snapshot()["in_flight"] == 3


def test_refuses_immediately_when_the_queue_is_zero() -> None:
    admission = AdmissionControl(max_concurrency=1, queue_limit=0)
    admission.acquire()
    with pytest.raises(Rejected, match="queue full"):
        admission.acquire()


def test_a_rejection_tells_the_client_when_to_come_back() -> None:
    admission = AdmissionControl(max_concurrency=1, queue_limit=0)
    admission.acquire()
    with pytest.raises(Rejected) as excinfo:
        admission.acquire()
    assert excinfo.value.retry_after >= 1


def test_releasing_admits_the_next_waiter() -> None:
    admission = AdmissionControl(max_concurrency=1, queue_limit=4, queue_timeout_s=5)
    admission.acquire()
    got: list[float] = []

    def waiter() -> None:
        got.append(admission.acquire())

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.05)
    assert admission.snapshot()["waiting"] == 1
    admission.release()
    thread.join(timeout=2)
    assert len(got) == 1
    assert got[0] > 0, "the queued request should report how long it waited"


def test_waiting_past_the_timeout_is_refused() -> None:
    """A request that has waited ten seconds has already blown its latency
    budget; telling the caller now beats serving it in thirty."""
    admission = AdmissionControl(max_concurrency=1, queue_limit=4, queue_timeout_s=0.1)
    admission.acquire()
    started = time.monotonic()
    with pytest.raises(Rejected, match="did not get one"):
        admission.acquire()
    assert time.monotonic() - started < 1.0
    assert admission.snapshot()["rejected_timeout"] == 1


def test_the_queue_itself_is_bounded() -> None:
    """Concurrency alone with unlimited waiting is the slow-queue failure."""
    admission = AdmissionControl(max_concurrency=1, queue_limit=2, queue_timeout_s=5)
    admission.acquire()
    threads = [threading.Thread(target=_swallow, args=(admission,)) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.1)
    with pytest.raises(Rejected, match="queue full"):
        admission.acquire()
    for _ in range(3):
        admission.release()
    for t in threads:
        t.join(timeout=2)


def _swallow(admission: AdmissionControl) -> None:
    with contextlib.suppress(Rejected):
        admission.acquire()


def test_the_slot_context_manager_cannot_leak() -> None:
    admission = AdmissionControl(max_concurrency=1, queue_limit=0)
    with pytest.raises(ValueError), slot(admission):
        raise ValueError("boom")
    assert admission.snapshot()["in_flight"] == 0


def test_release_never_goes_negative() -> None:
    admission = AdmissionControl(max_concurrency=2)
    admission.release()
    admission.release()
    assert admission.snapshot()["in_flight"] == 0


def test_peaks_are_tracked() -> None:
    admission = AdmissionControl(max_concurrency=4, queue_limit=0)
    for _ in range(3):
        admission.acquire()
    for _ in range(3):
        admission.release()
    assert admission.snapshot()["peak_in_flight"] == 3


def test_concurrency_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        AdmissionControl(max_concurrency=0)


def test_many_threads_never_exceed_the_limit() -> None:
    """The property that actually matters, checked under contention."""
    admission = AdmissionControl(max_concurrency=4, queue_limit=64, queue_timeout_s=5)
    breaches: list[int] = []
    lock = threading.Lock()
    live = {"n": 0}

    def worker() -> None:
        with slot(admission):
            with lock:
                live["n"] += 1
                if live["n"] > 4:
                    breaches.append(live["n"])
            time.sleep(0.005)
            with lock:
                live["n"] -= 1

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not breaches
    assert admission.snapshot()["in_flight"] == 0
