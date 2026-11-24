"""Backends, and knowing which of them are worth sending traffic to.

One vLLM process serves one model on one set of GPUs. The moment you have two
-- for redundancy, for a second model, or because one card could not hold the
context length you needed -- something in front has to decide where each
request goes and notice when one of them stops working.

**Health is passive first, active second.** The authoritative signal that a
backend is broken is that it just failed a real request; a `/health` poll that
succeeds thirty seconds from now does not undo that. So failures from live
traffic open the circuit immediately, and the health poll is only used to close
it again. Relying on polling alone means every request in the next poll
interval goes to a backend you already know is down.

**The circuit breaker is per backend, not global.** With two backends and one
of them wedged, the useful behaviour is all traffic to the healthy one, not
half the traffic failing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from selfserve.client import InferenceClient


@dataclass(slots=True)
class BackendStats:
    requests: int = 0
    failures: int = 0
    rejected: int = 0
    total_latency_s: float = 0.0

    @property
    def mean_latency_s(self) -> float:
        return self.total_latency_s / self.requests if self.requests else 0.0


class Backend:
    """One upstream inference server, with live state.

    `in_flight` is the number the router actually decides on. It is maintained
    by the gateway around each proxied request rather than sampled from the
    backend, because the backend cannot tell you about the request you are
    about to send it.
    """

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        failure_threshold: int = 3,
        recovery_s: float = 15.0,
        timeout: float = 300.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.client = InferenceClient(self.url, api_key=api_key, timeout=timeout)
        self.failure_threshold = failure_threshold
        self.recovery_s = recovery_s

        self._lock = threading.Lock()
        self.in_flight = 0
        self.consecutive_failures = 0
        self.opened_at: float | None = None  # circuit-open timestamp
        self.stats = BackendStats()
        self.last_error: str = ""

    # -------------------------------------------------------------- health
    @property
    def healthy(self) -> bool:
        """Open circuits close themselves after `recovery_s`.

        Half-open rather than closed: the next request is a probe. If it fails
        the circuit opens again immediately, so a backend that is still broken
        costs one request per recovery window instead of all of them.
        """
        with self._lock:
            if self.opened_at is None:
                return True
            if time.monotonic() - self.opened_at >= self.recovery_s:
                self.opened_at = None
                self.consecutive_failures = 0
                return True
            return False

    def record_success(self, latency_s: float) -> None:
        with self._lock:
            self.consecutive_failures = 0
            self.opened_at = None
            self.stats.requests += 1
            self.stats.total_latency_s += latency_s

    def record_failure(self, error: str) -> None:
        with self._lock:
            self.consecutive_failures += 1
            self.stats.failures += 1
            self.last_error = error[:200]
            if self.consecutive_failures >= self.failure_threshold and self.opened_at is None:
                self.opened_at = time.monotonic()

    def record_rejection(self) -> None:
        """A 503 from the backend is load shedding, not a fault.

        Counting it as a failure would open the circuit on a backend that is
        healthy and busy, which removes capacity at exactly the moment there is
        not enough of it.
        """
        with self._lock:
            self.stats.rejected += 1

    def probe(self) -> bool:
        ok, detail = self.client.health()
        if ok:
            with self._lock:
                self.opened_at = None
                self.consecutive_failures = 0
        else:
            self.record_failure(f"health probe: {detail}")
        return ok

    # ------------------------------------------------------------ in-flight
    def acquire(self) -> None:
        with self._lock:
            self.in_flight += 1

    def release(self) -> None:
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "url": self.url,
                "healthy": self.opened_at is None,
                "in_flight": self.in_flight,
                "requests": self.stats.requests,
                "failures": self.stats.failures,
                "rejected": self.stats.rejected,
                "mean_latency_s": round(self.stats.mean_latency_s, 3),
                "last_error": self.last_error,
            }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Backend({self.url}, in_flight={self.in_flight})"


@dataclass(slots=True)
class BackendPool:
    """Every backend the gateway knows about."""

    backends: list[Backend] = field(default_factory=list)

    @classmethod
    def of(cls, urls: list[str], **kw) -> BackendPool:
        return cls([Backend(url, **kw) for url in urls if url.strip()])

    @property
    def healthy(self) -> list[Backend]:
        return [b for b in self.backends if b.healthy]

    def probe_all(self) -> dict[str, bool]:
        return {b.url: b.probe() for b in self.backends}

    def snapshot(self) -> list[dict]:
        return [b.snapshot() for b in self.backends]

    def __len__(self) -> int:
        return len(self.backends)

    def __iter__(self):
        return iter(self.backends)
