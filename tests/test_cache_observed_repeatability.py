"""CPU-only tests for cache-observed inference repeatability."""

from inferenceguard.cache_observation import CacheObservedClient
from inferenceguard.http_client import InferenceObservation, InferenceTransportError
from inferenceguard.repeatability import run_repeatability


class FakeClient:
    def __init__(self, token_sequences):
        self.sequences = iter(token_sequences)

    def generate(self, model, prompt, max_tokens=32):
        return InferenceObservation(
            text="generated",
            finish_reason="stop",
            token_ids=tuple(next(self.sequences)),
        )


def make_metrics_reader(snapshots):
    values = iter(snapshots)
    return lambda: next(values)


def test_observe_cache_activity_for_three_requests():
    client = FakeClient([[10, 20]] * 3)

    metrics_reader = make_metrics_reader(
        [
            {"queries": 0, "hits": 0},
            {"queries": 100, "hits": 0},
            {"queries": 200, "hits": 80},
            {"queries": 300, "hits": 180},
        ]
    )

    observed = CacheObservedClient(client, metrics_reader)

    result = run_repeatability(
        observed,
        model="test-model",
        prompt="test prompt",
        repeats=3,
        comparison_mode="tokens",
    )

    assert result.verdict == "PASS"
    assert result.completed == 3

    assert len(observed.deltas) == 3
    assert [item["queries"] for item in observed.deltas] == [100, 100, 100]
    assert [item["hits"] for item in observed.deltas] == [0, 80, 100]


def test_cache_activity_does_not_hide_token_divergence():
    client = FakeClient(
        [
            [10, 20],
            [10, 21],
            [10, 20],
        ]
    )

    metrics_reader = make_metrics_reader(
        [
            {"queries": 0, "hits": 0},
            {"queries": 100, "hits": 0},
            {"queries": 200, "hits": 80},
            {"queries": 300, "hits": 180},
        ]
    )

    observed = CacheObservedClient(client, metrics_reader)

    result = run_repeatability(
        observed,
        model="test-model",
        prompt="test prompt",
        repeats=3,
        comparison_mode="tokens",
    )

    assert result.verdict == "FAIL"
    assert result.mismatch_indices == (1,)
    assert len(observed.deltas) == 3


def test_zero_cache_queries_are_not_reported_as_zero_rate():
    client = FakeClient([[10], [10], [10]])

    metrics_reader = make_metrics_reader(
        [
            {"queries": 50, "hits": 20},
            {"queries": 50, "hits": 20},
            {"queries": 50, "hits": 20},
            {"queries": 50, "hits": 20},
        ]
    )

    observed = CacheObservedClient(client, metrics_reader)

    result = run_repeatability(
        observed,
        model="test-model",
        prompt="test prompt",
        repeats=3,
        comparison_mode="tokens",
    )

    assert result.verdict == "PASS"
    assert all(delta["hit_rate"] is None for delta in observed.deltas)


def test_metrics_unavailable_before_first_request():
    client = FakeClient([[10], [10], [10]])

    snapshots = iter(
        [
            OSError("Metrics server unavailable"),
            {"queries": 100, "hits": 0},
            {"queries": 200, "hits": 80},
            {"queries": 300, "hits": 180},
        ]
    )

    def reader():
        value = next(snapshots)
        if isinstance(value, Exception):
            raise value
        return value

    observed = CacheObservedClient(client, reader)

    result = run_repeatability(
        observed,
        model="test-model",
        prompt="test",
        repeats=3,
        comparison_mode="tokens",
    )

    assert result.verdict == "PASS"
    assert result.completed == 3
    assert observed.deltas[0] is None
    assert observed.deltas[1]["hits"] == 80
    assert observed.deltas[2]["hits"] == 100
    assert observed.errors[0] is not None


def test_metrics_failure_after_inference():
    client = FakeClient([[10], [10], [10]])

    snapshots = iter(
        [
            {"queries": 0, "hits": 0},
            {"queries": 100, "hits": 0},
            OSError("Metrics HTTP 503"),
            {"queries": 300, "hits": 180},
        ]
    )

    def reader():
        value = next(snapshots)
        if isinstance(value, Exception):
            raise value
        return value

    observed = CacheObservedClient(client, reader)

    result = run_repeatability(
        observed,
        model="test-model",
        prompt="test",
        repeats=3,
        comparison_mode="tokens",
    )

    assert result.verdict == "PASS"
    assert result.completed == 3
    assert observed.deltas[0]["queries"] == 100
    assert observed.deltas[1] is None
    assert observed.deltas[2] is None
    assert observed.errors[1] is not None
    assert len(observed.deltas) == result.completed
    assert len(observed.errors) == result.completed

    assert observed.errors[0] is None
    assert observed.errors[1] is not None
    assert observed.errors[2] is not None


def test_counter_reset_does_not_invalidate_tokens():
    client = FakeClient([[10], [10], [10]])

    observed = CacheObservedClient(
        client,
        make_metrics_reader(
            [
                {"queries": 500, "hits": 300},
                {"queries": 600, "hits": 400},
                {"queries": 50, "hits": 10},
                {"queries": 150, "hits": 90},
            ]
        ),
    )

    result = run_repeatability(
        observed,
        model="test-model",
        prompt="test",
        repeats=3,
        comparison_mode="tokens",
    )

    assert result.verdict == "PASS"
    assert observed.deltas[0]["hits"] == 100
    assert observed.deltas[1] is None
    assert observed.deltas[2]["hits"] == 80
    assert observed.errors[1] is not None
    assert len(observed.deltas) == result.completed
    assert len(observed.errors) == result.completed

    assert observed.errors[0] is None
    assert observed.errors[1] is not None
    assert observed.errors[2] is None


def test_inference_failure_remains_inconclusive():
    class FailingClient:
        def generate(self, model, prompt, max_tokens=32):
            raise InferenceTransportError("Inference unavailable")

    observed = CacheObservedClient(
        FailingClient(),
        lambda: {"queries": 0, "hits": 0},
    )

    result = run_repeatability(
        observed,
        model="test-model",
        prompt="test",
        repeats=3,
        comparison_mode="tokens",
    )

    assert result.verdict == "INCONCLUSIVE"
    assert result.completed == 0
    assert observed.deltas == []
