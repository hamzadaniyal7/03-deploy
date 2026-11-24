from __future__ import annotations

from collections.abc import Iterator

import pytest

from selfserve.gateway import Gateway, GatewayConfig
from selfserve.mock import MockConfig, MockServer

# Fast enough that a full suite of real HTTP round trips stays under a couple
# of seconds, slow enough that timings are still distinguishable from zero.
FAST = MockConfig(base_ttft_s=0.004, base_itl_s=0.001, batch_penalty=0.05, capacity=16)


@pytest.fixture
def mock() -> Iterator[MockServer]:
    with MockServer(FAST) as server:
        yield server


@pytest.fixture
def mock_pair() -> Iterator[tuple[MockServer, MockServer]]:
    with MockServer(FAST) as a, MockServer(FAST) as b:
        yield a, b


@pytest.fixture
def gateway(mock: MockServer) -> Iterator[Gateway]:
    gw = Gateway(GatewayConfig(backends=[mock.url], port=0, max_concurrency=16))
    with gw:
        yield gw
