import copy
import json

import pytest

from inferenceguard.cli import main, render_html
from inferenceguard.core import Policy, ValidationError, compare_runs, load_run


def sample():
    return {
        "schema_version": "1.0",
        "metadata": {
            "model_id": "fixture-model",
            "model_revision": "abc123",
            "tokenizer_id": "fixture-tokenizer",
            "tokenizer_revision": "abc123",
            "sampling": {"temperature": 0, "max_tokens": 2, "seed": 42},
        },
        "cases": [
            {
                "id": "case-1",
                "prompt_hash": "sha256:123",
                "tokens": [7, 8],
                "logprobs": [-0.1, -0.2],
            }
        ],
    }


def test_identical_pass():
    a = sample()
    assert compare_runs(a, copy.deepcopy(a), Policy())["verdict"] == "PASS"


def test_corrupted_tokens_fail():
    a, b = sample(), sample()
    b["cases"][0]["tokens"] = [7, 9]
    report = compare_runs(a, b, Policy())
    assert report["verdict"] == "FAIL"
    assert "Generated token IDs differ" in report["cases"][0]["reasons"]


def test_metadata_mismatch_incompatible():
    a, b = sample(), sample()
    b["metadata"]["model_revision"] = "different"
    assert compare_runs(a, b, Policy())["verdict"] == "INCOMPATIBLE"


def test_prompt_fingerprint_fail():
    a, b = sample(), sample()
    b["cases"][0]["prompt_hash"] = "sha256:other"
    assert compare_runs(a, b, Policy())["verdict"] == "FAIL"


def test_logprobs_tolerance():
    a, b = sample(), sample()
    b["cases"][0]["logprobs"][0] += 0.01
    assert compare_runs(a, b, Policy(logprob_abs_tolerance=0.001))["verdict"] == "FAIL"
    assert compare_runs(a, b, Policy(logprob_abs_tolerance=0.02))["verdict"] == "PASS"


def test_invalid_nonfinite_policy():
    with pytest.raises(ValidationError):
        Policy(logprob_abs_tolerance=float("nan"))


def test_duplicate_case_rejected():
    a = sample()
    a["cases"].append(copy.deepcopy(a["cases"][0]))
    with pytest.raises(ValidationError):
        compare_runs(a, sample(), Policy())


def test_different_case_sets_are_incompatible():
    a, b = sample(), sample()
    b["cases"][0]["id"] = "other"
    assert compare_runs(a, b, Policy())["verdict"] == "INCOMPATIBLE"


def test_html_escapes_case_identifiers():
    a = sample()
    a["cases"][0]["id"] = "<script>alert(1)</script>"
    report = compare_runs(a, a, Policy())
    rendered = render_html(report)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_cli_exit_codes(tmp_path):
    a = sample()
    path_a, path_b, report = (tmp_path / x for x in ("a.json", "b.json", "report.json"))
    path_a.write_text(json.dumps(a))
    path_b.write_text(json.dumps(a))
    args = [
        "compare",
        "--reference",
        str(path_a),
        "--candidate",
        str(path_b),
        "--output",
        str(report),
    ]
    assert main(args) == 0
    assert json.loads(report.read_text())["verdict"] == "PASS"
    a["cases"][0]["tokens"] = [5, 6]
    path_b.write_text(json.dumps(a))
    assert main(args) == 1
    a["metadata"]["model_id"] = "another-model"
    path_b.write_text(json.dumps(a))
    assert main(args) == 2


def test_nan_json_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"value": NaN}')
    with pytest.raises(ValidationError):
        load_run(bad)
