"""The simulator itself: it has to model contention or it measures nothing."""

from __future__ import annotations

import threading

from selfserve.client import ChatRequest, InferenceClient
from selfserve.mock import MockConfig, MockServer


def test_prefill_cost_grows_with_prompt_length(mock: MockServer) -> None:
    client = InferenceClient(mock.url)
    short = client.chat(ChatRequest.of("hi", max_tokens=2))
    long = client.chat(ChatRequest.of("word " * 4000, max_tokens=2))
    assert long.ttft_s > short.ttft_s


def test_decode_slows_as_concurrency_rises() -> None:
    """The behaviour that makes a benchmark sweep meaningful: a batch advances
    together, so every stream in it is slower than it would be alone."""
    config = MockConfig(base_ttft_s=0.002, base_itl_s=0.004, batch_penalty=0.4, capacity=16)
    with MockServer(config) as server:
        client = InferenceClient(server.url)
        alone = client.chat(ChatRequest.of("hi", max_tokens=10))

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
            t.join()

    def mean_itl(result) -> float:
        gaps = result.inter_token_latencies
        return sum(gaps) / len(gaps)

    contended = sum(mean_itl(r) for r in results if r.ok) / sum(1 for r in results if r.ok)
    assert contended > mean_itl(alone)


def test_capacity_is_enforced_and_counted() -> None:
    config = MockConfig(capacity=2, base_itl_s=0.02, base_ttft_s=0.02)
    with MockServer(config) as server:
        client = InferenceClient(server.url)
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
            t.join()
        snapshot = server.state.snapshot()

    assert snapshot["peak_active"] <= 2
    assert snapshot["rejected"] > 0
    assert sum(1 for r in results if r.status == 503) == snapshot["rejected"]


def test_503_carries_retry_after() -> None:
    import urllib.error
    import urllib.request

    with MockServer(MockConfig(capacity=0)) as server:
        request = urllib.request.Request(
            server.url + "/v1/chat/completions",
            data=b'{"messages":[{"role":"user","content":"x"}]}',
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(request, timeout=5)
            raise AssertionError("expected a 503")
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
            assert exc.headers.get("Retry-After") == "1"


def test_failure_rate_is_deterministic_for_a_seed() -> None:
    def failures() -> int:
        with MockServer(MockConfig(failure_rate=0.5, seed=99)) as server:
            client = InferenceClient(server.url)
            return sum(not client.chat(ChatRequest.of("x", max_tokens=1)).ok for _ in range(20))

    assert failures() == failures()


def test_sse_is_chunked_like_a_real_server(mock: MockServer) -> None:
    """Not an incidental detail: SSE on an HTTP/1.1 keep-alive connection with
    neither Content-Length nor chunked framing has no defined end, and a proxy
    reading to EOF blocks until it times out."""
    import urllib.request

    request = urllib.request.Request(
        mock.url + "/v1/chat/completions",
        data=b'{"messages":[{"role":"user","content":"x"}],"max_tokens":2,"stream":true}',
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.headers.get("Transfer-Encoding") == "chunked"
        body = response.read().decode()
    assert "data: [DONE]" in body


def test_unknown_paths_404(mock: MockServer) -> None:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(mock.url + "/nope", timeout=5)
        raise AssertionError("expected a 404")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404


def test_malformed_json_is_a_400(mock: MockServer) -> None:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        mock.url + "/v1/chat/completions",
        data=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=5)
        raise AssertionError("expected a 400")
    except urllib.error.HTTPError as exc:
        assert exc.code == 400
