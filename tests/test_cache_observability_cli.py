"""CPU-only tests for opt-in prefix-cache CLI integration."""

import json

from inferenceguard import cli, prefix_cache_metrics
from inferenceguard.http_client import InferenceObservation, InferenceTransportError


class FakeClient:
    def generate(self, model, prompt, max_tokens=32):
        return InferenceObservation(
            text="same output",
            finish_reason="stop",
            token_ids=(10, 20, 30),
        )


def run_cli(tmp_path, monkeypatch, *, observe=False, client=None):
    monkeypatch.setattr(
        cli,
        "HttpInferenceClient",
        lambda **kwargs: client if client is not None else FakeClient(),
    )

    output = tmp_path / "report.json"

    args = [
        "repeat",
        "--endpoint",
        "http://127.0.0.1:8000/v1/completions",
        "--model",
        "test-model",
        "--prompt",
        "test prompt",
        "--repeats",
        "3",
        "--comparison-mode",
        "tokens",
        "--output",
        str(output),
    ]

    if observe:
        args.append("--observe-prefix-cache")

    exit_code = cli.main(args)

    return exit_code, json.loads(output.read_text())


def test_default_cli_report_remains_unchanged(tmp_path, monkeypatch):
    def unexpected_metrics_request(*args, **kwargs):
        raise AssertionError("Metrics should not be requested")

    monkeypatch.setattr(
        prefix_cache_metrics,
        "fetch_prefix_cache_metrics",
        unexpected_metrics_request,
    )

    exit_code, report = run_cli(tmp_path, monkeypatch)

    assert exit_code == 0
    assert report["verdict"] == "PASS"
    assert report["completed"] == 3
    assert "cache_observability" not in report


def test_cli_records_cache_observations(tmp_path, monkeypatch):
    snapshots = iter(
        [
            {"queries": 0, "hits": 0},
            {"queries": 100, "hits": 0},
            {"queries": 200, "hits": 80},
            {"queries": 300, "hits": 180},
        ]
    )

    monkeypatch.setattr(
        prefix_cache_metrics,
        "fetch_prefix_cache_metrics",
        lambda *args, **kwargs: next(snapshots),
    )

    exit_code, report = run_cli(
        tmp_path,
        monkeypatch,
        observe=True,
    )

    assert exit_code == 0
    assert report["verdict"] == "PASS"

    cache = report["cache_observability"]

    assert cache["status"] == "AVAILABLE"
    assert cache["initial_error"] is None
    assert cache["errors"] == [None, None, None]

    assert [item["hits"] for item in cache["deltas"]] == [0, 80, 100]


def test_cli_preserves_verdict_when_metrics_fail(tmp_path, monkeypatch):
    def unavailable_metrics(*args, **kwargs):
        raise RuntimeError("Metrics unavailable")

    monkeypatch.setattr(
        prefix_cache_metrics,
        "fetch_prefix_cache_metrics",
        unavailable_metrics,
    )

    exit_code, report = run_cli(
        tmp_path,
        monkeypatch,
        observe=True,
    )

    assert exit_code == 0
    assert report["verdict"] == "PASS"
    assert report["completed"] == 3

    cache = report["cache_observability"]

    assert cache["status"] == "UNAVAILABLE"
    assert cache["deltas"] == [None, None, None]
    assert len(cache["errors"]) == 3
    assert cache["initial_error"] is not None


def test_cli_detects_token_divergence_with_cache_metrics(tmp_path, monkeypatch):
    class DivergentClient:
        def __init__(self):
            self.calls = 0

        def generate(self, model, prompt, max_tokens=32):
            self.calls += 1
            tokens = (10, 20, 30) if self.calls != 2 else (10, 21, 30)

            return InferenceObservation(
                text="generated",
                finish_reason="stop",
                token_ids=tokens,
            )

    snapshots = iter(
        [
            {"queries": 0, "hits": 0},
            {"queries": 100, "hits": 0},
            {"queries": 200, "hits": 80},
            {"queries": 300, "hits": 180},
        ]
    )

    monkeypatch.setattr(
        prefix_cache_metrics,
        "fetch_prefix_cache_metrics",
        lambda *args, **kwargs: next(snapshots),
    )

    exit_code, report = run_cli(
        tmp_path,
        monkeypatch,
        observe=True,
        client=DivergentClient(),
    )

    assert exit_code == 1
    assert report["verdict"] == "FAIL"
    assert report["mismatch_indices"] == [1]
    assert report["completed"] == 3

    cache = report["cache_observability"]
    assert cache["status"] == "AVAILABLE"
    assert len(cache["deltas"]) == 3
    assert cache["errors"] == [None, None, None]


def test_cli_reports_partial_cache_observability(tmp_path, monkeypatch):
    snapshots = iter(
        [
            {"queries": 0, "hits": 0},
            {"queries": 100, "hits": 0},
            OSError("Metrics unavailable"),
            {"queries": 300, "hits": 180},
        ]
    )

    def metrics_reader(*args, **kwargs):
        value = next(snapshots)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(
        prefix_cache_metrics,
        "fetch_prefix_cache_metrics",
        metrics_reader,
    )

    exit_code, report = run_cli(
        tmp_path,
        monkeypatch,
        observe=True,
    )

    assert exit_code == 0
    assert report["verdict"] == "PASS"

    cache = report["cache_observability"]

    assert cache["status"] == "PARTIAL"
    assert cache["deltas"][0]["queries"] == 100
    assert cache["deltas"][1] is None
    assert cache["deltas"][2] is None

    assert len(cache["errors"]) == 3
    assert cache["errors"][0] is None
    assert cache["errors"][1] is not None
    assert cache["errors"][2] is not None


def test_cli_incomplete_inference_remains_inconclusive(tmp_path, monkeypatch):
    class FailingAfterFirstRequest:
        def __init__(self):
            self.calls = 0

        def generate(self, model, prompt, max_tokens=32):
            self.calls += 1

            if self.calls == 2:
                raise InferenceTransportError("Inference server unavailable")

            return InferenceObservation(
                text="generated",
                finish_reason="stop",
                token_ids=(10, 20, 30),
            )

    snapshots = iter(
        [
            {"queries": 0, "hits": 0},
            {"queries": 100, "hits": 0},
        ]
    )

    monkeypatch.setattr(
        prefix_cache_metrics,
        "fetch_prefix_cache_metrics",
        lambda *args, **kwargs: next(snapshots),
    )

    exit_code, report = run_cli(
        tmp_path,
        monkeypatch,
        observe=True,
        client=FailingAfterFirstRequest(),
    )

    assert exit_code == 2
    assert report["verdict"] == "INCONCLUSIVE"
    assert report["requested"] == 3
    assert report["completed"] == 1

    cache = report["cache_observability"]

    assert cache["status"] == "PARTIAL"
    assert len(cache["deltas"]) == 1
    assert len(cache["errors"]) == 1
    assert cache["deltas"][0]["queries"] == 100
