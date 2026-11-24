"""Choosing a backend.

**Least outstanding requests, not round robin.** This is the one routing
decision that is genuinely different for LLM traffic. Round robin assumes
requests cost roughly the same; for inference they differ by two orders of
magnitude -- a 32k-token prompt generating 2k tokens occupies a slot for
perhaps a hundred times as long as a 200-token prompt generating 50. Round
robin will hand a backend its fifth long request while another sits idle,
because it is counting requests sent rather than requests still running.

Least-outstanding needs no cost model and no prediction. A backend that is busy
with expensive work has a high in-flight count by definition, so it stops
receiving traffic until it catches up.

Ties break randomly rather than by order. Deterministic tie-breaking sends
every request to the first backend in the list whenever the pool is idle, which
is most of the time at low traffic -- so the "load balanced" deployment runs
entirely on one node until it gets busy, and the second node's first request in
an hour is the one that pays for a cold prefix cache.
"""

from __future__ import annotations

import random

from selfserve.gateway.backends import Backend, BackendPool


class NoBackendAvailable(RuntimeError):
    """Every backend is unhealthy. Distinct from 'every backend is busy'."""


class Router:
    def __init__(self, pool: BackendPool, *, seed: int | None = None) -> None:
        self.pool = pool
        self._rng = random.Random(seed)

    def pick(self, exclude: set[str] | None = None) -> Backend:
        """The least-loaded healthy backend not in `exclude`."""
        exclude = exclude or set()
        candidates = [b for b in self.pool.healthy if b.url not in exclude]
        if not candidates:
            # Everything healthy has already been tried for this request. Fall
            # back to unhealthy backends rather than failing outright: a circuit
            # opened by a transient blip is better probed with a real request
            # than left idle while the caller gets an error.
            candidates = [b for b in self.pool.backends if b.url not in exclude]
        if not candidates:
            raise NoBackendAvailable(
                f"no backend available ({len(self.pool)} configured, "
                f"{len(self.pool.healthy)} healthy, {len(exclude)} already tried)"
            )

        fewest = min(b.in_flight for b in candidates)
        return self._rng.choice([b for b in candidates if b.in_flight == fewest])
