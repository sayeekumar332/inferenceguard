"""Validate combined real-vLLM token and prefix-cache evidence.

P4.1 implements offline evidence validation only.
GPU experiment execution will be added in a later milestone.
"""

import math

from experiments.vllm_http_t4.run_experiment import validate_report


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
