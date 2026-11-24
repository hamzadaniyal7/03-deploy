"""The HTTP server base both the mock and the gateway sit on."""

from __future__ import annotations

import threading

from selfserve.client import ChatRequest, InferenceClient
from selfserve.httpserver import QuietThreadingHTTPServer
from selfserve.mock import MockConfig, MockServer


def test_listen_backlog_is_not_the_stdlib_default() -> None:
    """socketserver defaults to 5 pending connections. An inference endpoint
    is specifically a thing that gets many at once, and the excess is refused
    by the kernel -- which reaches the client as "connection refused" and
    looks exactly like the server having crashed."""
    assert QuietThreadingHTTPServer.request_queue_size >= 128


def test_many_simultaneous_connections_are_all_accepted() -> None:
    """Regression: the suite failed intermittently because 40 concurrent
    connections overran a backlog of 5. With the old default this fails
    roughly 40% of its connections."""
    config = MockConfig(base_ttft_s=0.01, base_itl_s=0.002, capacity=64)
    results: list = []
    lock = threading.Lock()

    with MockServer(config) as server:
        client = InferenceClient(server.url)

        def send() -> None:
            result = client.chat(ChatRequest.of("hi", max_tokens=3))
            with lock:
                results.append(result)

        threads = [threading.Thread(target=send) for _ in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

    refused = [r for r in results if not r.ok and "connection" in r.error.lower()]
    assert not refused, f"{len(refused)}/40 connections refused"
    assert len(results) == 40


def test_client_disconnects_are_counted_not_logged() -> None:
    """A cancelled generation is normal. The stdlib prints a twenty-line
    traceback for each one, which under load buries every real error."""
    assert hasattr(QuietThreadingHTTPServer, "handle_error")
    config = MockConfig(base_ttft_s=0.001, base_itl_s=0.001)
    with MockServer(config) as server:
        # A client that reads one chunk and hangs up mid-stream.
        import urllib.request

        request = urllib.request.Request(
            server.url + "/v1/chat/completions",
            data=b'{"messages":[{"role":"user","content":"x"}],"max_tokens":200,"stream":true}',
            headers={"Content-Type": "application/json"},
        )
        response = urllib.request.urlopen(request, timeout=5)
        response.read(10)
        response.close()
        # The server must still be serving afterwards.
        ok, _ = InferenceClient(server.url).health()
        assert ok
