"""CPU-only HTTP tests for vLLM prefix-cache metrics."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import patch

import pytest

from inferenceguard.prefix_cache_metrics import (
    fetch_prefix_cache_metrics,
)


@contextmanager
def mock_metrics_server(body, status=200, redirect_location=None):
    """Run a local mock Prometheus endpoint."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/metrics":
                self.send_error(404)
                return

            payload = body.encode("utf-8")

            self.send_response(status)
            if redirect_location is not None:
                self.send_header("Location", redirect_location)
            self.send_header(
                "Content-Type",
                "text/plain; charset=utf-8",
            )
            self.send_header(
                "Content-Length",
                str(len(payload)),
            )
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


VALID_METRICS = """
# HELP vllm:prefix_cache_queries_total Prefix-cache query tokens
vllm:prefix_cache_queries_total 754
vllm:prefix_cache_hits_total 368
"""


def test_fetch_valid_prefix_cache_metrics():
    with mock_metrics_server(VALID_METRICS) as url:
        result = fetch_prefix_cache_metrics(url)

    assert result == {"queries": 754, "hits": 368}


def test_fetch_metrics_with_trailing_slash():
    with mock_metrics_server(VALID_METRICS) as url:
        result = fetch_prefix_cache_metrics(url + "/")

    assert result == {"queries": 754, "hits": 368}


def test_reject_missing_prefix_cache_metric():
    with mock_metrics_server("vllm:prefix_cache_queries_total 100") as url:
        with pytest.raises(ValueError):
            fetch_prefix_cache_metrics(url)


def test_reject_malformed_prefix_cache_metric():
    with mock_metrics_server(
        "vllm:prefix_cache_queries_total invalid\nvllm:prefix_cache_hits_total 50\n"
    ) as url:
        with pytest.raises(ValueError):
            fetch_prefix_cache_metrics(url)


def test_reject_http_server_error():
    with mock_metrics_server("server unavailable", status=503) as url:
        with pytest.raises(RuntimeError):
            fetch_prefix_cache_metrics(url)


def test_reject_connection_error():
    with patch(
        "inferenceguard.prefix_cache_metrics.urllib.request.OpenerDirector.open",
        side_effect=OSError("Connection refused"),
    ):
        with pytest.raises(RuntimeError):
            fetch_prefix_cache_metrics("http://127.0.0.1:8000")


def test_reject_http_timeout():
    with patch(
        "inferenceguard.prefix_cache_metrics.urllib.request.urlopen",
        side_effect=TimeoutError("Timed out"),
    ):
        with pytest.raises(RuntimeError):
            fetch_prefix_cache_metrics(
                "http://127.0.0.1:8000",
                timeout=0.1,
            )


def test_reject_redirect_response():
    """Redirects must not be followed to another destination."""
    with mock_metrics_server("", status=302) as url:
        with pytest.raises(RuntimeError):
            fetch_prefix_cache_metrics(url)


def test_reject_oversized_metrics_response():
    """A metrics response exceeding the limit must be rejected."""
    oversized_metrics = VALID_METRICS + "\n" + ("# padding\n" * 200000)

    with mock_metrics_server(oversized_metrics) as url:
        with pytest.raises(ValueError, match="size"):
            fetch_prefix_cache_metrics(url)


def test_reject_non_http_url():
    with pytest.raises(ValueError):
        fetch_prefix_cache_metrics("file:///etc/passwd")


def test_reject_redirect_with_valid_destination():
    """Never follow a redirect, even to a working metrics server."""
    with mock_metrics_server(VALID_METRICS) as destination:
        with mock_metrics_server(
            "",
            status=302,
            redirect_location=f"{destination}/metrics",
        ) as source:
            with pytest.raises(RuntimeError):
                fetch_prefix_cache_metrics(source)
