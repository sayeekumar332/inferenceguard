"""Validate combined real-vLLM token and prefix-cache evidence.

P4.1 implements offline evidence validation only.
GPU experiment execution will be added in a later milestone.
"""

import json
import math
import shutil
import subprocess
import sys
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

from experiments.vllm_http_t4.run_experiment import (
    gpu_name,
    validate_report,
    wait_for_server,
)
from inferenceguard.prefix_cache_metrics import (
    calculate_cache_delta,
    fetch_prefix_cache_metrics,
)

OUTPUT_DIR = Path("experiments/vllm_cache_observability_t4/evidence")

CACHE_TEST_PROMPT = (
    "Inference regression testing compares generated token sequences "
    "across repeated requests using identical model configuration, "
    "sampling parameters, input prompts, and inference server settings. "
    "Prefix caching reuses previously computed attention states, "
    "so verification must distinguish cold requests from warmed requests. "
) * 12 + "\nSummarize the importance of deterministic inference verification."


def validate_cache_report(report, exit_code):
    """Independently validate token repeatability and cache evidence."""

    if not isinstance(report, dict):
        raise RuntimeError("Experiment report must be a dictionary")

    # Reuse the independently tested token-evidence validator.
    validate_report(report, exit_code)

    if report.get("verdict") != "PASS" or exit_code != 0:
        raise RuntimeError("P4 requires a passing token-verification result")

    cache = report.get("cache_observability")

    if not isinstance(cache, dict):
        raise RuntimeError("Missing cache-observability evidence")

    if cache.get("status") != "AVAILABLE":
        raise RuntimeError("Complete cache-observability evidence required")

    if cache.get("initial_error") is not None:
        raise RuntimeError("Unexpected initial metrics error")

    deltas = cache.get("deltas")
    errors = cache.get("errors")

    if not isinstance(deltas, list) or len(deltas) != 3:
        raise RuntimeError("Expected three cache delta measurements")

    if not isinstance(errors, list) or len(errors) != 3:
        raise RuntimeError("Expected three cache error entries")

    if any(error is not None for error in errors):
        raise RuntimeError("Cache measurement errors are present")

    total_queries = 0
    total_hits = 0
    repeated_hits = 0

    for index, delta in enumerate(deltas):
        if not isinstance(delta, dict):
            raise RuntimeError(f"Missing cache delta for request {index + 1}")

        queries = delta.get("queries")
        hits = delta.get("hits")
        hit_rate = delta.get("hit_rate")

        for name, value in (("queries", queries), ("hits", hits)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"Invalid {name} counter for request {index + 1}")

        if hits > queries:
            raise RuntimeError("Cache-hit delta exceeds query delta")

        if queries == 0:
            if hit_rate is not None:
                raise RuntimeError("Zero-query interval must have a null hit rate")
        else:
            if (
                isinstance(hit_rate, bool)
                or not isinstance(hit_rate, (int, float))
                or not math.isfinite(hit_rate)
                or not math.isclose(
                    hit_rate,
                    hits / queries,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ):
                raise RuntimeError(f"Invalid cache-hit rate for request {index + 1}")

        total_queries += queries
        total_hits += hits

        # Cache reuse must be measured on a repeated request,
        # not inferred from an earlier unrelated cache hit.
        if index > 0:
            repeated_hits += hits

    if total_queries == 0:
        raise RuntimeError("No cache queries were measured")

    if repeated_hits == 0:
        raise RuntimeError("No prefix-cache hits measured on repeated requests")

    print("Combined evidence validation: PASS")
    print("Token-verification verdict:", report["verdict"])
    print("Total cache-query tokens:", total_queries)
    print("Total cache-hit tokens:", total_hits)
    print("Repeated-request cache-hit tokens:", repeated_hits)

    return {
        "verdict": report["verdict"],
        "cache_status": cache["status"],
        "total_queries": total_queries,
        "total_hits": total_hits,
        "repeated_hits": repeated_hits,
    }


def build_server_command(vllm_executable: str) -> list[str]:
    """Construct the validated Tesla T4 vLLM server command."""
    if not isinstance(vllm_executable, str) or not vllm_executable.strip():
        raise ValueError("A valid vLLM executable is required")

    return [
        vllm_executable,
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


def build_inferenceguard_command(
    python_executable: str,
    output_path: str,
) -> list[str]:
    """Construct the token-verification and cache-observation command."""
    if not isinstance(python_executable, str) or not python_executable.strip():
        raise ValueError("A valid Python executable is required")

    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError("A valid evidence output path is required")

    return [
        python_executable,
        "-m",
        "inferenceguard.cli",
        "repeat",
        "--endpoint",
        "http://127.0.0.1:8000/v1/completions",
        "--model",
        "Qwen/Qwen3-0.6B",
        "--prompt",
        CACHE_TEST_PROMPT,
        "--repeats",
        "3",
        "--comparison-mode",
        "tokens",
        "--max-tokens",
        "24",
        "--observe-prefix-cache",
        "--output",
        output_path,
    ]


def check_metrics_preflight(base_url: str) -> dict[str, int]:
    """Verify that vLLM provides valid prefix-cache metrics.

    Execute this check after the inference server becomes healthy,
    but before running the repeated inference experiment.
    """

    try:
        snapshot = fetch_prefix_cache_metrics(base_url)

        # Validate individual counters and their relationship
        # using the existing, tested cache-delta implementation.
        calculate_cache_delta(snapshot, snapshot)

    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Metrics preflight failed: {exc}") from exc

    print("Metrics preflight: PASS")
    print("Prefix-cache query tokens:", snapshot["queries"])
    print("Prefix-cache hit tokens:", snapshot["hits"])

    return snapshot


@contextmanager
def managed_vllm_server(command: list[str], log_path: Path):
    """Start vLLM and guarantee subprocess cleanup."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(  # noqa: S603
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            else:
                process.wait()


def prepare_server_for_inference(process):
    """Require server readiness and valid cache metrics before inference."""
    wait_for_server(process)
    return check_metrics_preflight("http://127.0.0.1:8000")


def execute_inferenceguard(
    python_executable: str,
    report_path: Path,
) -> dict:
    """Execute InferenceGuard and independently validate its evidence."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_inferenceguard_command(
        python_executable,
        str(report_path),
    )

    # Never allow an earlier report to masquerade as a new result.
    report_path.unlink(missing_ok=True)

    result = subprocess.run(command, check=False)  # noqa: S603

    if not report_path.is_file():
        raise RuntimeError("InferenceGuard report was not generated")

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Invalid InferenceGuard JSON report") from exc

    return validate_cache_report(report, result.returncode)


def run_controlled_experiment(
    vllm_executable: str,
    python_executable: str,
    output_dir: Path,
) -> dict:
    """Run a controlled vLLM experiment with guaranteed server cleanup."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    server_command = build_server_command(vllm_executable)
    server_log = output_dir / "server.log"
    report_path = output_dir / "repeatability.json"

    with managed_vllm_server(server_command, server_log) as process:
        # Inference must not begin until health and metrics checks pass.
        prepare_server_for_inference(process)

        # This function also independently validates token and cache evidence.
        result = execute_inferenceguard(
            python_executable,
            report_path,
        )

    # The context manager has already shut down the server here.
    return result


def collect_environment_metadata() -> dict:
    """Collect reproducible GPU experiment environment metadata."""
    server_command = build_server_command("vllm")

    return {
        "gpu": gpu_name(),
        "model": "Qwen/Qwen3-0.6B",
        "vllm_version": version("vllm"),
        "python_version": sys.version.split()[0],
        "precision": "float16",
        "temperature": 0,
        "repeats": 3,
        "comparison_mode": "tokens",
        "max_tokens": 24,
        "prefix_caching": "enabled",
        "server_command": server_command,
    }


def run_experiment_entrypoint(
    vllm_executable: str,
    python_executable: str,
    output_dir: Path,
) -> dict:
    """Save experiment metadata, run validation, and preserve failures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    environment_path = output_dir / "environment.json"
    failure_path = output_dir / "failure.json"

    # Prevent an old failure report from being mistaken for a new one.
    failure_path.unlink(missing_ok=True)

    try:
        metadata = collect_environment_metadata()

        environment_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        return run_controlled_experiment(
            vllm_executable,
            python_executable,
            output_dir,
        )

    except Exception as exc:
        failure = {
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

        failure_path.write_text(
            json.dumps(failure, indent=2) + "\n",
            encoding="utf-8",
        )

        raise


def main() -> int:
    """Execute the reproducible vLLM cache-observability experiment."""
    vllm_executable = shutil.which("vllm")

    if vllm_executable is None:
        print("ERROR: vLLM executable not found", file=sys.stderr)
        return 1

    try:
        result = run_experiment_entrypoint(
            vllm_executable,
            sys.executable,
            OUTPUT_DIR,
        )
    except Exception as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 1

    print("Controlled experiment: PASS")
    print("Token verdict:", result["verdict"])
    print("Cache observability:", result["cache_status"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
