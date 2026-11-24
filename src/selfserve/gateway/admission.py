"""Deciding what not to accept.

An inference server has a hard capacity: the KV cache either has room for
another sequence or it does not. Past that point a request can be refused
quickly or queued slowly, and queueing slowly is almost always the wrong
choice.

The reason is that a queued request keeps costing the caller time without
making progress, and by the time it reaches the front of the queue its client
has often given up -- so the server spends real GPU time generating tokens for
a socket nobody is reading. A queue deep enough to absorb a spike is a queue
deep enough to serve nothing but timed-out requests during one.

So: a bounded number in flight, a bounded number waiting, and a fast 503 with
`Retry-After` past that. Shedding load is a feature. It is how a system under
more load than it can serve degrades into "some requests fail quickly" instead
of "all requests fail slowly", and the first is recoverable while the second
takes the caller down too.

The limits belong here rather than in the engine because the gateway is the
only place that can see the whole deployment. vLLM's own scheduler bounds what
one process will run concurrently; it cannot know you have two of them, and it
cannot refuse a request before it has paid to receive it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


class Rejected(Exception):
    """Admission refused. Carries what the client should be told."""

    def __init__(self, message: str, retry_after: int = 1) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(slots=True)
class AdmissionStats:
    admitted: int = 0
    rejected_concurrency: int = 0
    rejected_queue: int = 0
    rejected_timeout: int = 0
    peak_in_flight: int = 0
    peak_waiting: int = 0


class AdmissionControl:
    """A bounded semaphore with a bounded waiting room.

    `max_concurrency` is how many requests may be in flight across the whole
    deployment; `queue_limit` is how many may wait for a slot. Both are needed:
    concurrency alone with unlimited waiting is the slow-queue failure above,
    and a waiting room with no concurrency bound is not a limit at all.
    """

    def __init__(
        self,
        max_concurrency: int = 64,
        queue_limit: int = 128,
        queue_timeout_s: float = 10.0,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.max_concurrency = max_concurrency
        self.queue_limit = queue_limit
        self.queue_timeout_s = queue_timeout_s

        self._lock = threading.Lock()
        self._slot_freed = threading.Condition(self._lock)
        self._in_flight = 0
        self._waiting = 0
        self.stats = AdmissionStats()

    # ------------------------------------------------------------- acquire
    def acquire(self) -> float:
        """Wait for a slot. Returns seconds queued. Raises `Rejected` if not.

        The queue timeout is a second line of defence: a request that has
        waited ten seconds for a slot has already blown most latency budgets,
        and telling its caller now is more useful than serving it in thirty.
        """
        deadline = time.monotonic() + self.queue_timeout_s
        with self._slot_freed:
            if self._in_flight < self.max_concurrency:
                return self._admit(0.0)

            if self._waiting >= self.queue_limit:
                self.stats.rejected_queue += 1
                raise Rejected(
                    f"queue full ({self._waiting} waiting, {self._in_flight} in flight); "
                    "the deployment is at capacity",
                    retry_after=2,
                )

            self._waiting += 1
            self.stats.peak_waiting = max(self.stats.peak_waiting, self._waiting)
            started = time.monotonic()
            try:
                while self._in_flight >= self.max_concurrency:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.stats.rejected_timeout += 1
                        raise Rejected(
                            f"waited {self.queue_timeout_s:g}s for a slot and did not get one",
                            retry_after=5,
                        )
                    self._slot_freed.wait(remaining)
            finally:
                self._waiting -= 1
            return self._admit(time.monotonic() - started)

    def _admit(self, queued_s: float) -> float:
        self._in_flight += 1
        self.stats.admitted += 1
        self.stats.peak_in_flight = max(self.stats.peak_in_flight, self._in_flight)
        return queued_s

    def release(self) -> None:
        with self._slot_freed:
            self._in_flight = max(0, self._in_flight - 1)
            self._slot_freed.notify()

    # -------------------------------------------------------------- status
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "in_flight": self._in_flight,
                "waiting": self._waiting,
                "max_concurrency": self.max_concurrency,
                "queue_limit": self.queue_limit,
                "admitted": self.stats.admitted,
                "rejected_queue": self.stats.rejected_queue,
                "rejected_timeout": self.stats.rejected_timeout,
                "peak_in_flight": self.stats.peak_in_flight,
                "peak_waiting": self.stats.peak_waiting,
            }


class _Slot:
    """Context manager so a slot cannot be leaked by an early return."""

    def __init__(self, admission: AdmissionControl) -> None:
        self.admission = admission
        self.queued_s = 0.0

    def __enter__(self) -> _Slot:
        self.queued_s = self.admission.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.admission.release()


def slot(admission: AdmissionControl) -> _Slot:
    return _Slot(admission)
