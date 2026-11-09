"""CPU-only verification of the vLLM HTTP experiment validator."""

from copy import deepcopy

import pytest

from experiments.vllm_http_t4.run_experiment import validate_report


def make_report(token_sequences):
    """Construct a valid InferenceGuard token comparison report."""
    mismatches = [
        index
        for index in range(1, len(token_sequences))
        if token_sequences[index] != token_sequences[0]
    ]

    return {
        "verdict": "FAIL" if mismatches else "PASS",
        "requested": 3,
        "completed": 3,
        "observations": [
            {
                "text": "generated output",
                "finish_reason": "stop",
                "token_ids": tokens,
            }
            for tokens in token_sequences
        ],
        "mismatch_indices": mismatches,
        "error": None,
    }


def test_identical_token_sequences_accepted():
    report = make_report([[10, 20], [10, 20], [10, 20]])
    validate_report(report, exit_code=0)


def test_detected_token_divergence_accepted():
    report = make_report([[10, 20], [10, 21], [10, 20]])
    validate_report(report, exit_code=1)


@pytest.mark.parametrize(
    "invalid_value",
    [
        None,
        "10,20",
        [10, True],
        [10, -1],
        [10, 2.5],
    ],
)
def test_invalid_token_evidence_rejected(invalid_value):
    report = make_report([[10, 20], [10, 20], [10, 20]])
    report = deepcopy(report)
    report["observations"][1]["token_ids"] = invalid_value

    with pytest.raises(RuntimeError):
        validate_report(report, exit_code=0)


def test_incorrect_verdict_rejected():
    report = make_report([[10, 20], [10, 21], [10, 20]])
    report["verdict"] = "PASS"

    with pytest.raises(RuntimeError):
        validate_report(report, exit_code=1)


def test_incorrect_exit_code_rejected():
    report = make_report([[10, 20], [10, 20], [10, 20]])

    with pytest.raises(RuntimeError):
        validate_report(report, exit_code=1)


def test_incorrect_mismatch_indices_rejected():
    report = make_report([[10, 20], [10, 21], [10, 20]])
    report["mismatch_indices"] = []

    with pytest.raises(RuntimeError):
        validate_report(report, exit_code=1)


def test_incomplete_requests_rejected():
    report = make_report([[10, 20], [10, 20], [10, 20]])
    report["completed"] = 2

    with pytest.raises(RuntimeError):
        validate_report(report, exit_code=0)
