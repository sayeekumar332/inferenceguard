"""Tests for the HTTP repeatability CLI command."""

import json

import pytest

from inferenceguard import cli
from inferenceguard.http_client import InferenceObservation, InferenceTransportError


@pytest.mark.parametrize(
    ("responses", "expected_verdict", "expected_exit"),
    [
        (["A", "A", "A"], "PASS", 0),
        (["A", "B", "B"], "FAIL", 1),
        (["A", InferenceTransportError("unavailable")], "INCONCLUSIVE", 2),
    ],
)
def test_repeat_cli_verdicts(monkeypatch, tmp_path, responses, expected_verdict, expected_exit):
    requests = []

    class StubClient:
        def __init__(self, endpoint, timeout_seconds, *, include_token_ids=False):
            assert endpoint == "http://127.0.0.1:8000/v1/completions"
            assert timeout_seconds == 30.0
            self.responses = iter(responses)

        def generate(self, model, prompt, max_tokens=32):
            requests.append((model, prompt, max_tokens))
            response = next(self.responses)
            if isinstance(response, Exception):
                raise response
            return InferenceObservation(text=response, finish_reason="stop")

    monkeypatch.setattr(cli, "HttpInferenceClient", StubClient)

    output = tmp_path / "repeat.json"
    args = [
        "repeat",
        "--endpoint",
        "http://127.0.0.1:8000/v1/completions",
        "--model",
        "mock-model",
        "--prompt",
        "Hello",
        "--repeats",
        "3",
        "--output",
        str(output),
    ]

    assert cli.main(args) == expected_exit

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["verdict"] == expected_verdict
    assert report["requested"] == 3
    assert report["completed"] == sum(not isinstance(response, Exception) for response in responses)
    assert len(requests) == min(len(responses), 3)
    assert len(set(requests)) == 1


def test_repeat_cli_rejects_invalid_repeats(tmp_path):
    output = tmp_path / "repeat.json"

    result = cli.main(
        [
            "repeat",
            "--endpoint",
            "http://127.0.0.1:8000/v1/completions",
            "--model",
            "mock-model",
            "--prompt",
            "Hello",
            "--repeats",
            "1",
            "--output",
            str(output),
        ]
    )

    assert result == 2
    assert not output.exists()


def test_repeat_cli_token_mode_without_token_ids_is_inconclusive(monkeypatch, tmp_path):
    class StubClient:
        def __init__(self, endpoint, timeout_seconds, *, include_token_ids=False):
            pass

        def generate(self, model, prompt, max_tokens=32):
            return InferenceObservation(
                text="identical",
                finish_reason="stop",
            )

    monkeypatch.setattr(cli, "HttpInferenceClient", StubClient)

    output = tmp_path / "tokens_missing.json"

    exit_code = cli.main(
        [
            "repeat",
            "--endpoint",
            "http://127.0.0.1:8000/v1/completions",
            "--model",
            "mock-model",
            "--prompt",
            "Hello",
            "--comparison-mode",
            "tokens",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 2
    assert report["verdict"] == "INCONCLUSIVE"
    assert report["completed"] == 3
    assert "Token IDs unavailable" in report["error"]


def test_repeat_cli_detects_token_mismatch(monkeypatch, tmp_path):
    class StubClient:
        def __init__(self, endpoint, timeout_seconds, *, include_token_ids=False):
            self.tokens = iter([(10, 20), (10, 21), (10, 20)])

        def generate(self, model, prompt, max_tokens=32):
            return InferenceObservation(
                text="identical",
                finish_reason="stop",
                token_ids=next(self.tokens),
            )

    monkeypatch.setattr(cli, "HttpInferenceClient", StubClient)

    output = tmp_path / "token_mismatch.json"

    exit_code = cli.main(
        [
            "repeat",
            "--endpoint",
            "http://127.0.0.1:8000/v1/completions",
            "--model",
            "mock-model",
            "--prompt",
            "Hello",
            "--comparison-mode",
            "tokens",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert report["verdict"] == "FAIL"
    assert report["mismatch_indices"] == [1]
    assert [item["token_ids"] for item in report["observations"]] == [[10, 20], [10, 21], [10, 20]]


def test_repeat_cli_default_report_preserves_text_schema(monkeypatch, tmp_path):
    class StubClient:
        def __init__(self, endpoint, timeout_seconds, *, include_token_ids=False):
            assert include_token_ids is False

        def generate(self, model, prompt, max_tokens=32):
            return InferenceObservation(
                text="A",
                finish_reason="stop",
            )

    monkeypatch.setattr(cli, "HttpInferenceClient", StubClient)

    output = tmp_path / "legacy.json"

    exit_code = cli.main(
        [
            "repeat",
            "--endpoint",
            "http://127.0.0.1:8000/v1/completions",
            "--model",
            "mock-model",
            "--prompt",
            "Hello",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["verdict"] == "PASS"
    assert report["observations"] == [{"text": "A", "finish_reason": "stop"}] * 3


@pytest.mark.parametrize(
    ("token_sequences", "expected_verdict", "expected_exit", "expected_mismatches"),
    [
        ([[10, 20], [10, 20], [10, 20]], "PASS", 0, []),
        ([[10, 20], [10, 21], [10, 20]], "FAIL", 1, [1]),
        ([[10, 20], None, [10, 20]], "INCONCLUSIVE", 2, []),
        ([[10, 20], [10, True], [10, 20]], "INCONCLUSIVE", 2, []),
    ],
)
def test_repeat_cli_real_http_token_integration(
    tmp_path,
    token_sequences,
    expected_verdict,
    expected_exit,
    expected_mismatches,
):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/completions"

            length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(length))
            requests.append(request)

            index = len(requests) - 1
            token_ids = token_sequences[index]

            choice = {
                "index": 0,
                "text": "identical",
                "finish_reason": "stop",
            }
            if token_ids is not None:
                choice["token_ids"] = token_ids

            response = json.dumps({"choices": [choice]}).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    output = tmp_path / "http_tokens.json"

    try:
        exit_code = cli.main(
            [
                "repeat",
                "--endpoint",
                f"http://127.0.0.1:{server.server_port}/v1/completions",
                "--model",
                "mock-model",
                "--prompt",
                "Hello",
                "--repeats",
                "3",
                "--comparison-mode",
                "tokens",
                "--output",
                str(output),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == expected_exit
    assert report["verdict"] == expected_verdict
    assert report["mismatch_indices"] == expected_mismatches

    assert len(requests) == (
        2
        if expected_verdict == "INCONCLUSIVE"
        and token_sequences[1] is not None
        and isinstance(token_sequences[1][1], bool)
        else 3
    )

    for request in requests:
        assert request["return_token_ids"] is True
        assert request["temperature"] == 0
        assert request["stream"] is False
        assert request["model"] == "mock-model"
        assert request["prompt"] == "Hello"


def test_repeat_cli_real_http_text_backward_compatibility(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(length))
            requests.append(request)

            response = json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": "stable output",
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    output = tmp_path / "text_compatibility.json"

    try:
        exit_code = cli.main(
            [
                "repeat",
                "--endpoint",
                f"http://127.0.0.1:{server.server_port}/v1/completions",
                "--model",
                "mock-model",
                "--prompt",
                "Hello",
                "--repeats",
                "3",
                "--comparison-mode",
                "text",
                "--output",
                str(output),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["verdict"] == "PASS"
    assert report["completed"] == 3
    assert report["mismatch_indices"] == []

    assert len(requests) == 3
    assert all("return_token_ids" not in req for req in requests)
    assert all(req["temperature"] == 0 for req in requests)
    assert all(req["stream"] is False for req in requests)

    assert all("token_ids" not in observation for observation in report["observations"])
