"""The client against the simulated server -- real HTTP, real SSE, no GPU."""

from __future__ import annotations

from selfserve.client import ChatRequest, InferenceClient
from selfserve.mock import MockConfig, MockServer


def test_health_and_models(mock: MockServer) -> None:
    client = InferenceClient(mock.url)
    ok, detail = client.health()
    assert ok and "200" in detail
    assert client.models() == ["mock-8b"]


def test_streaming_returns_text_and_timings(mock: MockServer) -> None:
    result = InferenceClient(mock.url).chat(ChatRequest.of("hello", max_tokens=8))
    assert result.ok and result.status == 200
    assert result.completion_tokens == 8
    assert len(result.token_times) == 8
    assert result.ttft_s is not None and result.ttft_s > 0
    assert result.text


def test_token_arrival_times_are_increasing(mock: MockServer) -> None:
    """Flat or out-of-order timings mean the stream was buffered, which is the
    classic way to produce a completely wrong ITL curve."""
    result = InferenceClient(mock.url).chat(ChatRequest.of("hello", max_tokens=10))
    assert result.token_times == sorted(result.token_times)
    assert all(gap > 0 for gap in result.inter_token_latencies)


def test_usage_comes_from_the_server_not_a_guess(mock: MockServer) -> None:
    result = InferenceClient(mock.url).chat(
        ChatRequest.of("a longer prompt with several words in it", max_tokens=6)
    )
    assert result.prompt_tokens > 0
    assert result.completion_tokens == 6


def test_non_streaming_has_no_first_token_moment(mock: MockServer) -> None:
    """Recording TTFT = total would make non-streaming numbers look
    comparable with streaming ones, which they are not."""
    result = InferenceClient(mock.url).chat(ChatRequest.of("hello", max_tokens=5, stream=False))
    assert result.ok and result.ttft_s is None
    assert result.completion_tokens == 5
    assert result.text


def test_a_server_error_is_a_measurement_not_an_exception() -> None:
    with MockServer(MockConfig(failure_rate=1.0)) as server:
        result = InferenceClient(server.url).chat(ChatRequest.of("hi"))
    assert not result.ok
    assert result.status == 500
    assert "simulated backend failure" in result.error


def test_an_unreachable_server_is_reported_not_raised() -> None:
    result = InferenceClient("http://127.0.0.1:9").chat(ChatRequest.of("hi"))
    assert not result.ok and "connection failed" in result.error


def test_health_of_an_unreachable_server() -> None:
    ok, detail = InferenceClient("http://127.0.0.1:9").health()
    assert not ok and "unreachable" in detail


def test_capacity_is_reported_as_503() -> None:
    """A real server with a full KV cache refuses rather than queueing."""
    import threading

    with MockServer(MockConfig(capacity=1, base_itl_s=0.02, base_ttft_s=0.02)) as server:
        client = InferenceClient(server.url)
        results = []

        def send() -> None:
            results.append(client.chat(ChatRequest.of("hi", max_tokens=12)))

        threads = [threading.Thread(target=send) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert any(r.status == 503 for r in results), "capacity 1 should reject concurrent requests"
    assert any(r.ok for r in results)


def test_request_payload_asks_for_usage_when_streaming() -> None:
    payload = ChatRequest.of("x").payload()
    assert payload["stream_options"] == {"include_usage": True}


def test_request_payload_omits_stream_options_when_not_streaming() -> None:
    assert "stream_options" not in ChatRequest.of("x", stream=False).payload()


def test_extra_fields_pass_through() -> None:
    payload = ChatRequest.of("x", extra={"top_p": 0.9, "seed": 1}).payload()
    assert payload["top_p"] == 0.9 and payload["seed"] == 1
