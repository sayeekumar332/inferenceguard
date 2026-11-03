"""End-to-end tests combining HTTP transport and repeatability verification."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from inferenceguard.cli import main
from inferenceguard.http_client import HttpInferenceClient
from inferenceguard.repeatability import run_repeatability


@contextmanager
def mock_inference_server(outputs):
    """Serve a controlled sequence of completion texts or HTTP errors."""
    state = {"requests": [], "count": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            raw = self.rfile.read(length)
            state["requests"].append(json.loads(raw))

            assert self.path == "/v1/completions"

            index = state["count"]
            state["count"] += 1
            output = outputs[index]

            if output is None:
                self.send_response(503)
                self.end_headers()
                return

            response = {
                "id": f"cmpl-{index}",
                "object": "text_completion",
                "choices": [
                    {
                        "index": 0,
                        "text": output,
                        "finish_reason": "stop",
                    }
                ],
            }

            payload = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/completions"
        yield endpoint, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def run_experiment(outputs):
    with mock_inference_server(outputs) as (endpoint, state):
        client = HttpInferenceClient(endpoint)
        result = run_repeatability(
            client,
            model="mock-model",
            prompt="Explain RISC-V in one sentence.",
            repeats=3,
            max_tokens=32,
        )
        return result, state["requests"]


def assert_identical_requests(requests):
    assert len(requests) == 3
    assert all(request == requests[0] for request in requests)
    assert requests[0]["temperature"] == 0
    assert requests[0]["stream"] is False


def test_http_identical_responses_pass():
    result, requests = run_experiment(["A", "A", "A"])

    assert result.verdict == "PASS"
    assert result.completed == 3
    assert result.mismatch_indices == ()
    assert_identical_requests(requests)


def test_http_first_request_divergence_detected():
    result, requests = run_experiment(["A", "B", "B"])

    assert result.verdict == "FAIL"
    assert result.completed == 3
    assert result.mismatch_indices == (1, 2)
    assert [item.text for item in result.observations] == ["A", "B", "B"]
    assert_identical_requests(requests)


def test_http_server_failure_is_inconclusive():
    result, requests = run_experiment(["A", None, "A"])

    assert result.verdict == "INCONCLUSIVE"
    assert result.completed == 1
    assert result.error == "Request 2: InferenceTransportError"
    assert len(requests) == 2
    assert requests[0] == requests[1]


@pytest.mark.parametrize(
    ("outputs", "verdict", "exit_code", "completed", "mismatches", "attempts"),
    [
        (["A", "A", "A"], "PASS", 0, 3, [], 3),
        (["A", "B", "B"], "FAIL", 1, 3, [1, 2], 3),
        (["A", None, "A"], "INCONCLUSIVE", 2, 1, [], 2),
    ],
)
def test_repeat_cli_end_to_end(
    tmp_path, outputs, verdict, exit_code, completed, mismatches, attempts
):
    """Verify CLI -> HTTP -> repeatability -> JSON report."""
    report_path = tmp_path / "repeatability.json"

    with mock_inference_server(outputs) as (endpoint, state):
        result_code = main(
            [
                "repeat",
                "--endpoint",
                endpoint,
                "--model",
                "mock-model",
                "--prompt",
                "Explain RISC-V in one sentence.",
                "--repeats",
                "3",
                "--max-tokens",
                "32",
                "--timeout-seconds",
                "5",
                "--output",
                str(report_path),
            ]
        )

    assert result_code == exit_code
    assert report_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["verdict"] == verdict
    assert report["requested"] == 3
    assert report["completed"] == completed
    assert report["mismatch_indices"] == mismatches

    requests = state["requests"]
    assert len(requests) == attempts
    assert all(request == requests[0] for request in requests)

    assert requests[0]["model"] == "mock-model"
    assert requests[0]["prompt"] == "Explain RISC-V in one sentence."
    assert requests[0]["max_tokens"] == 32
    assert requests[0]["temperature"] == 0
    assert requests[0]["stream"] is False

    if verdict == "FAIL":
        assert [item["text"] for item in report["observations"]] == ["A", "B", "B"]

    if verdict == "INCONCLUSIVE":
        assert report["error"] == "Request 2: InferenceTransportError"
