"""The gateway, end to end over real sockets."""

from __future__ import annotations

import threading
import time

from selfserve.client import ChatRequest, InferenceClient
from selfserve.gateway import Gateway, GatewayConfig
from selfserve.mock import MockConfig, MockServer

FAST = MockConfig(base_ttft_s=0.004, base_itl_s=0.001, capacity=16)


def _settles(gateway: Gateway, predicate, timeout: float = 5.0) -> bool:
    """Wait briefly for gateway bookkeeping to catch up with the client.

    The gateway records a request's outcome *after* the response has been
    delivered, and a streaming client returns the moment it reads [DONE]. So
    between the last `chat()` returning and the counters being accurate there
    is a genuine window. Tests that assert on counts synchronise on it here
    rather than racing it -- the alternative is a suite that passes most of
    the time, which is worse than one that fails.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate(gateway.status()):
            return True
        time.sleep(0.01)
    return False


def test_proxies_a_streaming_request(gateway: Gateway) -> None:
    result = InferenceClient(gateway.url).chat(ChatRequest.of("hello", max_tokens=8))
    assert result.ok and result.completion_tokens == 8
    assert len(result.token_times) == 8
    assert result.text


def test_streaming_is_passed_through_not_buffered(gateway: Gateway) -> None:
    """Buffering would turn a 200ms time-to-first-token into the full
    generation time, which is the only property a streaming endpoint has.

    Enough tokens that the margin survives a slow machine: if the response
    were buffered, TTFT would equal total, so asserting it is under a third
    cannot be closed by scheduling noise the way `< 0.9 x total` could on a
    two-core runner.
    """
    result = InferenceClient(gateway.url).chat(ChatRequest.of("hello", max_tokens=60))
    assert result.ttft_s is not None
    assert len(result.token_times) == 60
    assert result.ttft_s < result.total_s / 3
    assert all(gap > 0 for gap in result.inter_token_latencies)


def test_proxies_a_blocking_request(gateway: Gateway) -> None:
    result = InferenceClient(gateway.url).chat(ChatRequest.of("hello", max_tokens=5, stream=False))
    assert result.ok and result.completion_tokens == 5


def test_a_completed_stream_counts_as_a_success(gateway: Gateway) -> None:
    """Regression: a client that stops reading at [DONE] -- the correct thing
    for a client to do -- was being recorded as a truncated stream, which
    marked every healthy backend as failing."""
    client = InferenceClient(gateway.url)
    for _ in range(3):
        client.chat(ChatRequest.of("hi", max_tokens=2))
    # The client returns the moment it reads [DONE]; the gateway thread still
    # has to unwind, record the outcome and release its slot. Every assertion
    # about those counters has to come after they have caught up -- an earlier
    # version of this test settled only on in_flight, *after* asserting the
    # request count, so the count itself was still a race and CI duly failed
    # it with `assert 2 == 3`.
    assert _settles(gateway, lambda s: s["backends"][0]["requests"] == 3)
    assert _settles(gateway, lambda s: s["backends"][0]["in_flight"] == 0)
    snapshot = gateway.status()["backends"][0]
    assert snapshot["requests"] == 3
    assert snapshot["failures"] == 0


def test_token_accounting_comes_from_the_stream(gateway: Gateway) -> None:
    InferenceClient(gateway.url).chat(ChatRequest.of("a prompt here", max_tokens=7))
    assert _settles(gateway, lambda s: s["usage"]["requests"] == 1)
    usage = gateway.status()["usage"]
    assert usage["requests"] == 1
    assert usage["completion_tokens"] == 7
    assert usage["prompt_tokens"] > 0


def test_health_and_models_are_served(gateway: Gateway) -> None:
    client = InferenceClient(gateway.url)
    ok, _ = client.health()
    assert ok
    assert client.models() == ["mock-8b"]


# -------------------------------------------------------------------- auth
def test_api_key_is_enforced_when_set(mock: MockServer) -> None:
    config = GatewayConfig(backends=[mock.url], port=0, api_key="secret")
    with Gateway(config) as gateway:
        anonymous = InferenceClient(gateway.url).chat(ChatRequest.of("hi"))
        assert not anonymous.ok and anonymous.status == 401

        authorised = InferenceClient(gateway.url, api_key="secret").chat(
            ChatRequest.of("hi", max_tokens=2)
        )
        assert authorised.ok


def test_no_api_key_means_no_auth(gateway: Gateway) -> None:
    assert InferenceClient(gateway.url).chat(ChatRequest.of("hi", max_tokens=2)).ok


# ---------------------------------------------------------------- failover
def test_a_dead_backend_is_routed_around() -> None:
    with MockServer(FAST) as good:
        config = GatewayConfig(backends=["http://127.0.0.1:9", good.url], port=0)
        with Gateway(config) as gateway:
            client = InferenceClient(gateway.url)
            # Every request succeeds even though half the pool is unreachable.
            assert all(client.chat(ChatRequest.of("hi", max_tokens=2)).ok for _ in range(12))
            assert _settles(gateway, lambda s: s["backends"][1]["requests"] == 12)
            dead, alive = gateway.status()["backends"]
            assert dead["failures"] > 0, "the dead backend should have recorded failures"
            assert dead["requests"] == 0
            assert alive["requests"] == 12


def test_every_backend_dead_returns_502() -> None:
    config = GatewayConfig(backends=["http://127.0.0.1:9"], port=0, max_retries=1)
    with Gateway(config) as gateway:
        result = InferenceClient(gateway.url).chat(ChatRequest.of("hi"))
    assert not result.ok and result.status == 502


def test_a_503_does_not_open_the_circuit() -> None:
    """A 503 is the backend working as designed. Opening its circuit would
    remove capacity exactly when there is not enough.

    One backend, always full, so which backend gets the request is not a coin
    flip. The earlier version of this test used a full backend alongside a
    healthy one and asserted the full one had been offered a request -- with
    random tie-breaking that is a 1-in-16 bet, and it duly failed about one
    run in sixteen.
    """
    with MockServer(MockConfig(capacity=0)) as full:
        config = GatewayConfig(backends=[full.url], port=0, max_retries=0)
        with Gateway(config) as gateway:
            result = InferenceClient(gateway.url).chat(ChatRequest.of("hi", max_tokens=2))
            assert result.status == 503
            assert _settles(gateway, lambda s: s["backends"][0]["rejected"] == 1)
            busy = gateway.status()["backends"][0]
            assert busy["healthy"], "shedding load must not count against health"
            assert busy["failures"] == 0


def test_a_full_backend_is_routed_around() -> None:
    """With somewhere healthy to send them, requests still succeed."""
    with MockServer(MockConfig(capacity=0)) as full, MockServer(FAST) as good:
        config = GatewayConfig(backends=[full.url, good.url], port=0)
        with Gateway(config) as gateway:
            client = InferenceClient(gateway.url)
            assert all(client.chat(ChatRequest.of("hi", max_tokens=2)).ok for _ in range(8))
            assert _settles(gateway, lambda s: s["backends"][1]["requests"] == 8)
            assert gateway.status()["backends"][0]["failures"] == 0


def test_a_client_error_is_relayed_not_retried(mock: MockServer) -> None:
    """A 4xx will fail identically on every backend; retrying it just
    multiplies the work."""
    import urllib.error
    import urllib.request

    with Gateway(GatewayConfig(backends=[mock.url], port=0)) as gateway:
        request = urllib.request.Request(
            gateway.url + "/v1/chat/completions",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(request, timeout=5)
            raise AssertionError("expected a 400")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400


# ------------------------------------------------------------- admission
def test_the_gateway_sheds_load_past_its_limit() -> None:
    slow = MockConfig(base_ttft_s=0.05, base_itl_s=0.02, capacity=64)
    with MockServer(slow) as server:
        config = GatewayConfig(backends=[server.url], port=0, max_concurrency=2, queue_limit=0)
        with Gateway(config) as gateway:
            client = InferenceClient(gateway.url)
            results: list = []
            lock = threading.Lock()

            def send() -> None:
                r = client.chat(ChatRequest.of("hi", max_tokens=10))
                with lock:
                    results.append(r)

            threads = [threading.Thread(target=send) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            assert any(r.status == 503 for r in results)
            assert any(r.ok for r in results)
            assert gateway.status()["admission"]["peak_in_flight"] <= 2


def test_load_is_spread_across_backends(mock_pair) -> None:
    """Sequential requests leave both backends idle, so each pick is a coin
    flip between them. Ten requests gives a 1-in-512 chance of a shutout --
    small, and over enough CI runs it happens, as it did here. Thirty makes it
    1-in-500-million while asserting exactly the same property.
    """
    requests = 30
    a, b = mock_pair
    with Gateway(GatewayConfig(backends=[a.url, b.url], port=0)) as gateway:
        client = InferenceClient(gateway.url)
        for _ in range(requests):
            client.chat(ChatRequest.of("hi", max_tokens=2))
        assert _settles(gateway, lambda s: sum(bk["requests"] for bk in s["backends"]) == requests)
        counts = [bk["requests"] for bk in gateway.status()["backends"]]
    assert sum(counts) == requests
    assert all(c > 0 for c in counts), f"one backend got nothing: {counts}"


# ----------------------------------------------------------------- status
def test_status_exposes_what_an_autoscaler_needs(gateway: Gateway) -> None:
    InferenceClient(gateway.url).chat(ChatRequest.of("hi", max_tokens=2))
    status = gateway.status()
    assert {"backends", "admission", "usage"} == set(status)
    assert "in_flight" in status["admission"]
    assert "queue_limit" in status["admission"]


def test_a_gateway_needs_at_least_one_backend() -> None:
    import pytest

    with pytest.raises(ValueError, match="at least one backend"):
        Gateway(GatewayConfig(backends=[]))


def test_unknown_paths_404(gateway: Gateway) -> None:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(gateway.url + "/nope", timeout=5)
        raise AssertionError("expected a 404")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404


def test_access_log_is_bounded() -> None:
    from selfserve.gateway.app import AccessLog

    log = AccessLog(limit=10)
    for i in range(50):
        log.record({"path": "/v1/chat/completions", "status": 200, "completion_tokens": i})
    assert len(log.entries) == 10
    assert log.totals()["requests"] == 10
