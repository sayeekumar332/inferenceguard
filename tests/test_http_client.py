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


def test_vllm_token_ids_acquired():
    response = completion("Output A")
    response["choices"][0]["token_ids"] = [101, 202, 303]

    with mock_endpoint(response) as endpoint:
        client = HttpInferenceClient(
            endpoint,
            include_token_ids=True,
        )
        result = client.generate("test-model", "Hello")

    assert result.text == "Output A"
    assert result.token_ids == (101, 202, 303)


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


def test_vllm_token_request_flag_sent(monkeypatch):
    import urllib.request
    from io import BytesIO

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self, size=-1):
            return BytesIO(json.dumps(completion()).encode("utf-8")).read(size)

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    endpoint = "http://127.0.0.1:8000/v1/completions"

    HttpInferenceClient(
        endpoint,
        include_token_ids=True,
    ).generate("test-model", "Hello")

    assert captured["payload"]["return_token_ids"] is True

    HttpInferenceClient(endpoint).generate("test-model", "Hello")

    assert "return_token_ids" not in captured["payload"]


@pytest.mark.parametrize(
    "invalid_token_ids",
    [
        "101,202,303",
        123,
        {"tokens": [101, 202]},
        [101, "202", 303],
        [101, True, 303],
        [101, -1, 303],
        [101, 2.5, 303],
    ],
)
def test_vllm_malformed_token_ids_rejected(invalid_token_ids):
    response = completion("Output A")
    response["choices"][0]["token_ids"] = invalid_token_ids

    with mock_endpoint(response) as endpoint:
        client = HttpInferenceClient(
            endpoint,
            include_token_ids=True,
        )

        with pytest.raises(InferenceResponseError):
            client.generate("test-model", "Hello")
