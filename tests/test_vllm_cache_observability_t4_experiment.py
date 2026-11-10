"""CPU-only validation of planned real-vLLM cache evidence."""

from copy import deepcopy

import pytest

from experiments.vllm_cache_observability_t4.run_experiment import (
    validate_cache_report,
)


def make_report():
    return {
        "verdict": "PASS",
        "requested": 3,
        "completed": 3,
        "observations": [
            {
                "text": "same output",
                "finish_reason": "length",
                "token_ids": [10, 20, 30],
            }
            for _ in range(3)
        ],
        "mismatch_indices": [],
        "error": None,
        "cache_observability": {
            "status": "AVAILABLE",
            "initial_error": None,
            "deltas": [
                {"queries": 256, "hits": 0, "hit_rate": 0.0},
                {"queries": 256, "hits": 224, "hit_rate": 0.875},
                {"queries": 256, "hits": 224, "hit_rate": 0.875},
            ],
            "errors": [None, None, None],
        },
    }


def test_valid_combined_evidence_accepted():
    validate_cache_report(make_report(), exit_code=0)


def test_missing_cache_observability_rejected():
    report = make_report()
    del report["cache_observability"]

    with pytest.raises(RuntimeError):
        validate_cache_report(report, exit_code=0)


def test_unavailable_cache_metrics_rejected():
    report = make_report()
    report["cache_observability"]["status"] = "UNAVAILABLE"
    report["cache_observability"]["deltas"] = [None, None, None]

    with pytest.raises(RuntimeError):
        validate_cache_report(report, exit_code=0)


def test_token_divergence_rejected():
    report = deepcopy(make_report())
    report["observations"][1]["token_ids"] = [10, 99, 30]

    with pytest.raises(RuntimeError):
        validate_cache_report(report, exit_code=0)


def test_no_measured_cache_hits_rejected():
    report = make_report()
    report["cache_observability"]["deltas"] = [
        {"queries": 256, "hits": 0, "hit_rate": 0.0} for _ in range(3)
    ]

    with pytest.raises(RuntimeError):
        validate_cache_report(report, exit_code=0)


def test_incorrect_cache_hit_rate_rejected():
    report = make_report()
    report["cache_observability"]["deltas"][1]["hit_rate"] = 0.5

    with pytest.raises(RuntimeError, match="hit rate"):
        validate_cache_report(report, exit_code=0)


@pytest.mark.parametrize("invalid_counter", [-1, True, 2.5, None])
def test_invalid_cache_counter_rejected(invalid_counter):
    report = make_report()
    report["cache_observability"]["deltas"][1]["hits"] = invalid_counter

    with pytest.raises(RuntimeError):
        validate_cache_report(report, exit_code=0)


def test_first_request_cache_hits_do_not_prove_reuse():
    report = make_report()

    report["cache_observability"]["deltas"] = [
        {"queries": 256, "hits": 128, "hit_rate": 0.5},
        {"queries": 256, "hits": 0, "hit_rate": 0.0},
        {"queries": 256, "hits": 0, "hit_rate": 0.0},
    ]

    with pytest.raises(RuntimeError, match="repeated requests"):
        validate_cache_report(report, exit_code=0)


def test_vllm_server_command_preserves_t4_configuration():
    from experiments.vllm_cache_observability_t4.run_experiment import (
        build_server_command,
    )

    command = build_server_command("/usr/bin/vllm")

    assert command == [
        "/usr/bin/vllm",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--dtype",
        "float16",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.75",
        "--enable-prefix-caching",
    ]


def test_inferenceguard_command_enables_cache_observability():
    from experiments.vllm_cache_observability_t4.run_experiment import (
        build_inferenceguard_command,
    )

    command = build_inferenceguard_command(
        python_executable="/usr/bin/python3",
        output_path="evidence/repeatability.json",
    )

    assert command[:4] == [
        "/usr/bin/python3",
        "-m",
        "inferenceguard.cli",
        "repeat",
    ]
    assert "--observe-prefix-cache" in command
    assert command[command.index("--comparison-mode") + 1] == "tokens"
    assert command[command.index("--repeats") + 1] == "3"
    assert command[command.index("--max-tokens") + 1] == "24"


def test_inferenceguard_command_preserves_output_destination():
    from experiments.vllm_cache_observability_t4.run_experiment import (
        build_inferenceguard_command,
    )

    command = build_inferenceguard_command(
        python_executable="/usr/bin/python3",
        output_path="evidence/custom-report.json",
    )

    assert command[command.index("--output") + 1] == ("evidence/custom-report.json")
    assert command[command.index("--endpoint") + 1] == ("http://127.0.0.1:8000/v1/completions")


def test_metrics_preflight_accepts_valid_counters(monkeypatch):
    from experiments.vllm_cache_observability_t4 import run_experiment

    monkeypatch.setattr(
        run_experiment,
        "fetch_prefix_cache_metrics",
        lambda base_url: {"queries": 100, "hits": 40},
        raising=False,
    )

    snapshot = run_experiment.check_metrics_preflight("http://127.0.0.1:8000")

    assert snapshot == {"queries": 100, "hits": 40}


def test_metrics_preflight_rejects_invalid_counters(monkeypatch):
    from experiments.vllm_cache_observability_t4 import run_experiment

    monkeypatch.setattr(
        run_experiment,
        "fetch_prefix_cache_metrics",
        lambda base_url: {"queries": 20, "hits": 30},
        raising=False,
    )

    with pytest.raises(RuntimeError, match="Metrics preflight failed"):
        run_experiment.check_metrics_preflight("http://127.0.0.1:8000")


def test_metrics_preflight_rejects_unavailable_endpoint(monkeypatch):
    from experiments.vllm_cache_observability_t4 import run_experiment

    def unavailable(base_url):
        raise RuntimeError("Metrics endpoint unavailable")

    monkeypatch.setattr(
        run_experiment,
        "fetch_prefix_cache_metrics",
        unavailable,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="Metrics"):
        run_experiment.check_metrics_preflight("http://127.0.0.1:8000")


def test_server_lifecycle_starts_and_stops(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from experiments.vllm_cache_observability_t4 import run_experiment

    process = MagicMock()
    process.poll.return_value = None

    popen = MagicMock(return_value=process)
    monkeypatch.setattr(run_experiment.subprocess, "Popen", popen)

    with run_experiment.managed_vllm_server(
        ["/usr/bin/vllm", "serve", "test-model"],
        tmp_path / "server.log",
    ) as running:
        assert running is process

    popen.assert_called_once()
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=15)


def test_server_lifecycle_cleans_up_after_failure(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from experiments.vllm_cache_observability_t4 import run_experiment

    process = MagicMock()
    process.poll.return_value = None

    monkeypatch.setattr(
        run_experiment.subprocess,
        "Popen",
        MagicMock(return_value=process),
    )

    with pytest.raises(RuntimeError, match="Experiment failed"):
        with run_experiment.managed_vllm_server(
            ["/usr/bin/vllm", "serve", "test-model"],
            tmp_path / "server.log",
        ):
            raise RuntimeError("Experiment failed")

    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=15)


def test_server_lifecycle_kills_after_shutdown_timeout(monkeypatch, tmp_path):
    import subprocess
    from unittest.mock import MagicMock

    from experiments.vllm_cache_observability_t4 import run_experiment

    process = MagicMock()
    process.poll.return_value = None
    process.wait.side_effect = [
        subprocess.TimeoutExpired(cmd="vllm", timeout=15),
        0,
    ]

    monkeypatch.setattr(
        run_experiment.subprocess,
        "Popen",
        MagicMock(return_value=process),
    )

    with run_experiment.managed_vllm_server(
        ["/usr/bin/vllm", "serve", "test-model"],
        tmp_path / "server.log",
    ):
        pass

    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2


def test_server_preflight_accepts_healthy_server_and_metrics(monkeypatch):
    from unittest.mock import MagicMock

    from experiments.vllm_cache_observability_t4 import run_experiment

    server = MagicMock()
    calls = []

    def fake_wait(process):
        assert process is server
        calls.append("health")

    def fake_metrics(base_url):
        assert base_url == "http://127.0.0.1:8000"
        calls.append("metrics")
        return {"queries": 100, "hits": 40}

    monkeypatch.setattr(run_experiment, "wait_for_server", fake_wait, raising=False)
    monkeypatch.setattr(
        run_experiment,
        "check_metrics_preflight",
        fake_metrics,
    )

    snapshot = run_experiment.prepare_server_for_inference(server)

    assert snapshot == {"queries": 100, "hits": 40}
    assert calls == ["health", "metrics"]


def test_server_preflight_rejects_early_server_exit(monkeypatch):
    from unittest.mock import MagicMock

    from experiments.vllm_cache_observability_t4 import run_experiment

    def fake_wait(process):
        raise RuntimeError("vLLM server exited before becoming ready")

    metrics = MagicMock()

    monkeypatch.setattr(run_experiment, "wait_for_server", fake_wait, raising=False)
    monkeypatch.setattr(
        run_experiment,
        "check_metrics_preflight",
        metrics,
    )

    with pytest.raises(RuntimeError, match="exited"):
        run_experiment.prepare_server_for_inference(MagicMock())

    metrics.assert_not_called()


def test_server_preflight_rejects_unavailable_metrics(monkeypatch):
    from unittest.mock import MagicMock

    from experiments.vllm_cache_observability_t4 import run_experiment

    def fake_wait(process):
        return None

    def fake_metrics(base_url):
        raise RuntimeError("Metrics preflight failed")

    monkeypatch.setattr(run_experiment, "wait_for_server", fake_wait, raising=False)
    monkeypatch.setattr(
        run_experiment,
        "check_metrics_preflight",
        fake_metrics,
    )

    with pytest.raises(RuntimeError, match="Metrics preflight failed"):
        run_experiment.prepare_server_for_inference(MagicMock())


def test_execute_inferenceguard_accepts_valid_evidence(monkeypatch, tmp_path):
    import json
    from subprocess import CompletedProcess

    from experiments.vllm_cache_observability_t4 import run_experiment

    report_path = tmp_path / "repeatability.json"
    report = make_report()

    def fake_run(command, check):
        assert check is False
        assert "--observe-prefix-cache" in command
        assert command[command.index("--output") + 1] == str(report_path)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return CompletedProcess(command, 0)

    monkeypatch.setattr(run_experiment.subprocess, "run", fake_run)

    result = run_experiment.execute_inferenceguard("/usr/bin/python3", report_path)

    assert result["verdict"] == "PASS"
    assert result["cache_status"] == "AVAILABLE"
    assert result["repeated_hits"] > 0


def test_execute_inferenceguard_rejects_missing_report(monkeypatch, tmp_path):
    from subprocess import CompletedProcess

    from experiments.vllm_cache_observability_t4 import run_experiment

    report_path = tmp_path / "missing.json"

    monkeypatch.setattr(
        run_experiment.subprocess,
        "run",
        lambda command, check: CompletedProcess(command, 0),
    )

    with pytest.raises(RuntimeError, match="not generated"):
        run_experiment.execute_inferenceguard("/usr/bin/python3", report_path)


def test_execute_inferenceguard_rejects_invalid_evidence(monkeypatch, tmp_path):
    import json
    from subprocess import CompletedProcess

    from experiments.vllm_cache_observability_t4 import run_experiment

    report_path = tmp_path / "repeatability.json"
    report = make_report()
    report["observations"][1]["token_ids"] = [10, 99, 30]

    def fake_run(command, check):
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return CompletedProcess(command, 0)

    monkeypatch.setattr(run_experiment.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError):
        run_experiment.execute_inferenceguard("/usr/bin/python3", report_path)


def test_controlled_experiment_runs_all_stages(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    from experiments.vllm_cache_observability_t4 import run_experiment

    server = MagicMock()
    calls = []

    @contextmanager
    def fake_server(command, log_path):
        assert "--enable-prefix-caching" in command
        assert log_path == tmp_path / "server.log"
        calls.append("start")
        try:
            yield server
        finally:
            calls.append("stop")

    def fake_preflight(process):
        assert process is server
        calls.append("preflight")
        return {"queries": 100, "hits": 40}

    def fake_execute(python_executable, report_path):
        assert python_executable == "/usr/bin/python3"
        assert report_path == tmp_path / "repeatability.json"
        calls.append("inference")
        return {"verdict": "PASS", "cache_status": "AVAILABLE"}

    monkeypatch.setattr(run_experiment, "managed_vllm_server", fake_server)
    monkeypatch.setattr(run_experiment, "prepare_server_for_inference", fake_preflight)
    monkeypatch.setattr(run_experiment, "execute_inferenceguard", fake_execute)

    result = run_experiment.run_controlled_experiment("/usr/bin/vllm", "/usr/bin/python3", tmp_path)

    assert result["verdict"] == "PASS"
    assert calls == ["start", "preflight", "inference", "stop"]


def test_controlled_experiment_stops_on_preflight_failure(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    from experiments.vllm_cache_observability_t4 import run_experiment

    calls = []

    @contextmanager
    def fake_server(command, log_path):
        calls.append("start")
        try:
            yield MagicMock()
        finally:
            calls.append("stop")

    def failing_preflight(process):
        calls.append("preflight")
        raise RuntimeError("Metrics preflight failed")

    execute = MagicMock()

    monkeypatch.setattr(run_experiment, "managed_vllm_server", fake_server)
    monkeypatch.setattr(run_experiment, "prepare_server_for_inference", failing_preflight)
    monkeypatch.setattr(run_experiment, "execute_inferenceguard", execute)

    with pytest.raises(RuntimeError, match="Metrics preflight failed"):
        run_experiment.run_controlled_experiment("/usr/bin/vllm", "/usr/bin/python3", tmp_path)

    assert calls == ["start", "preflight", "stop"]
    execute.assert_not_called()


def test_controlled_experiment_stops_on_inference_failure(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    from experiments.vllm_cache_observability_t4 import run_experiment

    calls = []

    @contextmanager
    def fake_server(command, log_path):
        calls.append("start")
        try:
            yield MagicMock()
        finally:
            calls.append("stop")

    def fake_preflight(process):
        calls.append("preflight")
        return {"queries": 100, "hits": 40}

    def failing_execute(python_executable, report_path):
        calls.append("inference")
        raise RuntimeError("Invalid inference evidence")

    monkeypatch.setattr(run_experiment, "managed_vllm_server", fake_server)
    monkeypatch.setattr(run_experiment, "prepare_server_for_inference", fake_preflight)
    monkeypatch.setattr(run_experiment, "execute_inferenceguard", failing_execute)

    with pytest.raises(RuntimeError, match="Invalid inference evidence"):
        run_experiment.run_controlled_experiment("/usr/bin/vllm", "/usr/bin/python3", tmp_path)

    assert calls == ["start", "preflight", "inference", "stop"]


def test_environment_metadata_records_gpu_and_vllm_version(monkeypatch):
    from experiments.vllm_cache_observability_t4 import run_experiment

    monkeypatch.setattr(
        run_experiment,
        "gpu_name",
        lambda: "NVIDIA Tesla T4",
        raising=False,
    )
    monkeypatch.setattr(
        run_experiment,
        "version",
        lambda package: "0.33.0",
        raising=False,
    )

    metadata = run_experiment.collect_environment_metadata()

    assert metadata["gpu"] == "NVIDIA Tesla T4"
    assert metadata["vllm_version"] == "0.33.0"
    assert metadata["model"] == "Qwen/Qwen3-0.6B"
    assert metadata["precision"] == "float16"
    assert metadata["prefix_caching"] == "enabled"


def test_environment_metadata_records_experiment_configuration(monkeypatch):
    from experiments.vllm_cache_observability_t4 import run_experiment

    monkeypatch.setattr(
        run_experiment,
        "gpu_name",
        lambda: "NVIDIA Tesla T4",
        raising=False,
    )
    monkeypatch.setattr(
        run_experiment,
        "version",
        lambda package: "0.33.0",
        raising=False,
    )

    metadata = run_experiment.collect_environment_metadata()

    assert metadata["temperature"] == 0
    assert metadata["repeats"] == 3
    assert metadata["comparison_mode"] == "tokens"
    assert metadata["max_tokens"] == 24
    assert "--enable-prefix-caching" in metadata["server_command"]
