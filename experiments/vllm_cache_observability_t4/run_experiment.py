"""Validate combined real-vLLM token and prefix-cache evidence.

P4.1 implements offline evidence validation only.
GPU experiment execution will be added in a later milestone.
"""

import math
import subprocess
from contextlib import contextmanager
from pathlib import Path

from experiments.vllm_http_t4.run_experiment import validate_report
from inferenceguard.prefix_cache_metrics import (
    calculate_cache_delta,
    fetch_prefix_cache_metrics,
)


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
        "Explain why deterministic inference verification is important.",
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
