"""A threading HTTP server that does not shout about client disconnects.

`socketserver` prints a full traceback for any exception escaping a handler,
including the ones that are not errors. A client that hangs up -- a cancelled
generation, a browser tab closing, a benchmark worker that got what it needed
and moved on -- produces `ConnectionResetError` or `BrokenPipeError` from deep
inside `handle_one_request`, and the default behaviour is to print twenty lines
of stack trace per occurrence.

That is worse than noise. Under any real load the genuine errors are buried in
thousands of lines about clients doing something completely normal, which is
how a log stops being read at all. Disconnects are counted and dropped;
everything else still gets its traceback.
"""

from __future__ import annotations

import socket
import sys
import threading
from http.server import ThreadingHTTPServer
from typing import Any

DISCONNECTS = (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, socket.timeout)


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that treats a client hanging up as normal."""

    daemon_threads = True
    # Without this, restarting the server on the same port inside a test run
    # fails with EADDRINUSE while the previous socket sits in TIME_WAIT.
    allow_reuse_address = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._disconnect_lock = threading.Lock()
        self.disconnects = 0

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, DISCONNECTS):
            with self._disconnect_lock:
                self.disconnects += 1
            return
        super().handle_error(request, client_address)
