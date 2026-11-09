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
