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
        def __init__(self, endpoint, timeout_seconds):
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
        def __init__(self, endpoint, timeout_seconds):
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
        def __init__(self, endpoint, timeout_seconds):
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
        def __init__(self, endpoint, timeout_seconds):
            pass

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
