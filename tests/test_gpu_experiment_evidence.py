"""Offline verification of recorded Tesla T4 inference evidence."""

import hashlib
import json
from pathlib import Path

import pytest

EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "experiments" / "t4_qwen3_0_6b"


@pytest.fixture(scope="module")
def evidence():
    names = {
        "off": "long_prefix_cache_off.json",
        "on": "long_prefix_cache_on.json",
        "comparison": "long_prefix_comparison.json",
    }
    return {
        key: json.loads((EVIDENCE_DIR / name).read_text(encoding="utf-8"))
        for key, name in names.items()
    }


def test_experiment_configuration(evidence):
    off = evidence["off"]
    on = evidence["on"]

    for report, cache_enabled in ((off, False), (on, True)):
        assert report["model"] == "Qwen/Qwen3-0.6B"
        assert report["vllm_version"] == "0.33.0"
        assert report["gpu"] == "Tesla T4"
        assert report["dtype"] == "float16"
        assert report["prefix_caching"] is cache_enabled
        assert report["temperature"] == 0.0
        assert report["max_tokens"] == 24
        assert report["max_model_len"] == 1024
        assert len(report["observations"]) == 3

    assert off["prompt_sha256"] == on["prompt_sha256"]
    assert off["prompt"] == on["prompt"]

    digest = hashlib.sha256(off["prompt"].encode("utf-8")).hexdigest()
    assert digest == off["prompt_sha256"]


def test_cache_disabled_has_no_hits(evidence):
    observations = evidence["off"]["observations"]

    assert [x["request"] for x in observations] == [1, 2, 3]
    assert all(x["prompt_tokens"] > 0 for x in observations)
    assert all(x["cached_tokens"] == 0 for x in observations)


def test_cache_enabled_reuses_prefix(evidence):
    observations = evidence["on"]["observations"]

    assert [x["request"] for x in observations] == [1, 2, 3]
    assert observations[0]["cached_tokens"] == 0

    for observation in observations[1:]:
        assert observation["cached_tokens"] > 0
        assert observation["cached_tokens"] <= observation["prompt_tokens"]


def test_generated_tokens_are_identical(evidence):
    off = evidence["off"]
    on = evidence["on"]

    for report in (off, on):
        observations = report["observations"]
        baseline = observations[0]["token_ids"]

        assert baseline
        assert report["all_token_ids_identical"] is True

        for observation in observations:
            assert observation["token_ids"] == baseline
            assert isinstance(observation["text"], str)

    assert [x["token_ids"] for x in off["observations"]] == [
        x["token_ids"] for x in on["observations"]
    ]

    assert [x["text"] for x in off["observations"]] == [x["text"] for x in on["observations"]]


def test_comparison_matches_original_evidence(evidence):
    off = evidence["off"]
    on = evidence["on"]
    comparison = evidence["comparison"]

    assert comparison["same_prompt"] is True
    assert comparison["cache_off_repeatable"] == off["all_token_ids_identical"]
    assert comparison["cache_on_repeatable"] == on["all_token_ids_identical"]

    assert comparison["cross_configuration_identical"] is True

    assert comparison["cache_off_hits"] == [x["cached_tokens"] for x in off["observations"]]
    assert comparison["cache_on_hits"] == [x["cached_tokens"] for x in on["observations"]]
