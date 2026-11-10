"""CPU-only tests for vLLM prefix-cache metrics parsing."""

import pytest

from inferenceguard.prefix_cache_metrics import (
    calculate_cache_delta,
    parse_prefix_cache_metrics,
)


def test_parse_unlabelled_metrics():
    metrics = """
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries_total 377
vllm:prefix_cache_hits_total 368
"""

    result = parse_prefix_cache_metrics(metrics)

    assert result == {"queries": 377, "hits": 368}


def test_parse_labelled_metrics():
    metrics = """
vllm:prefix_cache_queries_total{model_name="Qwen",engine="0"} 754
vllm:prefix_cache_hits_total{model_name="Qwen",engine="0"} 368
"""

    result = parse_prefix_cache_metrics(metrics)

    assert result == {"queries": 754, "hits": 368}


def test_accept_metric_names_without_total_suffix():
    metrics = """
vllm:prefix_cache_queries 100
vllm:prefix_cache_hits 80
"""

    assert parse_prefix_cache_metrics(metrics) == {
        "queries": 100,
        "hits": 80,
    }


def test_ignore_unrelated_metrics():
    metrics = """
vllm:prompt_tokens_total 1000
vllm:prefix_cache_queries_total 300
vllm:prefix_cache_hits_total 200
"""

    assert parse_prefix_cache_metrics(metrics) == {
        "queries": 300,
        "hits": 200,
    }


@pytest.mark.parametrize(
    "metrics",
    [
        "vllm:prefix_cache_queries_total 100",
        "vllm:prefix_cache_hits_total 80",
        "",
    ],
)
def test_reject_missing_metrics(metrics):
    with pytest.raises(ValueError):
        parse_prefix_cache_metrics(metrics)


@pytest.mark.parametrize(
    "invalid_value",
    ["-1", "NaN", "+Inf", "12.5", "invalid"],
)
def test_reject_invalid_counter_values(invalid_value):
    metrics = f"""
vllm:prefix_cache_queries_total {invalid_value}
vllm:prefix_cache_hits_total 10
"""

    with pytest.raises(ValueError):
        parse_prefix_cache_metrics(metrics)


def test_reject_duplicate_time_series():
    metrics = """
vllm:prefix_cache_queries_total{engine="0"} 100
vllm:prefix_cache_queries_total{engine="1"} 200
vllm:prefix_cache_hits_total{engine="0"} 50
vllm:prefix_cache_hits_total{engine="1"} 80
"""

    with pytest.raises(ValueError):
        parse_prefix_cache_metrics(metrics)


def test_calculate_cache_delta():
    before = {"queries": 377, "hits": 0}
    after = {"queries": 754, "hits": 368}

    result = calculate_cache_delta(before, after)

    assert result["queries"] == 377
    assert result["hits"] == 368
    assert result["hit_rate"] == pytest.approx(368 / 377)


def test_zero_query_interval():
    before = {"queries": 100, "hits": 50}
    after = {"queries": 100, "hits": 50}

    result = calculate_cache_delta(before, after)

    assert result == {
        "queries": 0,
        "hits": 0,
        "hit_rate": None,
    }


def test_zero_cache_hits():
    before = {"queries": 100, "hits": 20}
    after = {"queries": 200, "hits": 20}

    result = calculate_cache_delta(before, after)

    assert result == {
        "queries": 100,
        "hits": 0,
        "hit_rate": 0.0,
    }


def test_full_cache_hit_rate():
    before = {"queries": 100, "hits": 50}
    after = {"queries": 200, "hits": 150}

    result = calculate_cache_delta(before, after)

    assert result["hit_rate"] == 1.0


def test_reject_counter_reset():
    before = {"queries": 500, "hits": 300}
    after = {"queries": 100, "hits": 50}

    with pytest.raises(ValueError, match="reset"):
        calculate_cache_delta(before, after)


def test_reject_hit_counter_reset():
    before = {"queries": 500, "hits": 300}
    after = {"queries": 600, "hits": 100}

    with pytest.raises(ValueError, match="reset"):
        calculate_cache_delta(before, after)


def test_reject_hits_exceeding_queries():
    before = {"queries": 100, "hits": 50}
    after = {"queries": 150, "hits": 120}

    with pytest.raises(ValueError):
        calculate_cache_delta(before, after)


@pytest.mark.parametrize(
    "invalid_value",
    [-1, True, 2.5, "100", None],
)
def test_reject_invalid_snapshot_values(invalid_value):
    before = {"queries": invalid_value, "hits": 0}
    after = {"queries": 100, "hits": 50}

    with pytest.raises(ValueError):
        calculate_cache_delta(before, after)


def test_parser_accepts_real_vllm_float_counters():
    from inferenceguard.prefix_cache_metrics import parse_prefix_cache_metrics

    metrics = """
vllm:prefix_cache_queries_total{engine="0",model_name="Qwen/Qwen3-0.6B"} 256.0
vllm:prefix_cache_queries_created{engine="0",model_name="Qwen/Qwen3-0.6B"} 1794300852.0
vllm:prefix_cache_hits_total{engine="0",model_name="Qwen/Qwen3-0.6B"} 224.0
vllm:prefix_cache_hits_created{engine="0",model_name="Qwen/Qwen3-0.6B"} 1794300852.0
"""

    assert parse_prefix_cache_metrics(metrics) == {
        "queries": 256,
        "hits": 224,
    }


def test_parser_rejects_fractional_prometheus_counters():
    import pytest

    from inferenceguard.prefix_cache_metrics import parse_prefix_cache_metrics

    metrics = """
vllm:prefix_cache_queries_total 10.5
vllm:prefix_cache_hits_total 4.0
"""

    with pytest.raises(ValueError):
        parse_prefix_cache_metrics(metrics)
