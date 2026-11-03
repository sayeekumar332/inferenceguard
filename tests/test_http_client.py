"""Self-checking HTTP adapter tests with a local mock inference endpoint."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from inferenceguard.http_client import (
    HttpInferenceClient,
    InferenceResponseError,
    InferenceTransportError,
)


@contextmanager
def mock_endpoint(body, status=200):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            request_body = json.loads(self.rfile.read(length))

            assert self.path == "/v1/completions"
            assert request_body["temperature"] == 0
            assert request_body["stream"] is False

            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            if isinstance(body, bytes):
                self.wfile.write(body)
            else:
                self.wfile.write(json.dumps(body).encode("utf-8"))

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/completions"
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def completion(text="Output A"):
    return {
        "id": "cmpl-test",
        "object": "text_completion",
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
    }


def test_valid_completion():
    with mock_endpoint(completion()) as endpoint:
        result = HttpInferenceClient(endpoint).generate("test-model", "Hello")
        assert result.text == "Output A"
        assert result.finish_reason == "stop"


def test_missing_choices_rejected():
    with mock_endpoint({"id": "invalid"}) as endpoint:
        with pytest.raises(InferenceResponseError):
            HttpInferenceClient(endpoint).generate("test-model", "Hello")


def test_invalid_json_rejected():
    with mock_endpoint(b"{invalid json") as endpoint:
        with pytest.raises(InferenceResponseError):
            HttpInferenceClient(endpoint).generate("test-model", "Hello")


def test_http_server_error():
    with mock_endpoint({"error": "unavailable"}, status=503) as endpoint:
        with pytest.raises(InferenceTransportError):
            HttpInferenceClient(endpoint).generate("test-model", "Hello")


def test_invalid_endpoint_rejected():
    with pytest.raises(ValueError):
        HttpInferenceClient("file:///etc/passwd")


def test_invalid_timeout_rejected():
    with pytest.raises(ValueError):
        HttpInferenceClient(
            "http://127.0.0.1:8000/v1/completions",
            timeout_seconds=-1,
        )
