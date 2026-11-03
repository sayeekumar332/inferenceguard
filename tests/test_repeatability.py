"""Tests for repeated inference response comparison."""

import pytest

from inferenceguard.http_client import (
    InferenceObservation,
    InferenceTransportError,
)
from inferenceguard.repeatability import run_repeatability


class FakeClient:
    """Controlled mock client that returns a sequence of observations."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def generate(self, model, prompt, max_tokens=32):
        self.requests.append((model, prompt, max_tokens))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return InferenceObservation(text=response, finish_reason="stop")


def test_identical_responses_pass():
    client = FakeClient(["A", "A", "A"])
    result = run_repeatability(client, model="test", prompt="hello")

    assert result.verdict == "PASS"
    assert result.completed == 3
    assert result.mismatch_indices == ()
    assert len(client.requests) == 3
    assert len(set(client.requests)) == 1


def test_first_request_differs():
    client = FakeClient(["A", "B", "B"])
    result = run_repeatability(client, model="test", prompt="hello")

    assert result.verdict == "FAIL"
    assert result.mismatch_indices == (1, 2)


def test_last_request_differs():
    client = FakeClient(["A", "A", "B"])
    result = run_repeatability(client, model="test", prompt="hello")

    assert result.verdict == "FAIL"
    assert result.mismatch_indices == (2,)


def test_transport_failure_is_inconclusive():
    client = FakeClient(["A", InferenceTransportError("server unavailable"), "A"])
    result = run_repeatability(client, model="test", prompt="hello")

    assert result.verdict == "INCONCLUSIVE"
    assert result.completed == 1
    assert result.error == "Request 2: InferenceTransportError"
    assert len(client.requests) == 2


def test_failure_not_misclassified_as_pass():
    client = FakeClient(["A", "B", InferenceTransportError("connection lost")])
    result = run_repeatability(client, model="test", prompt="hello")

    assert result.verdict == "INCONCLUSIVE"
    assert result.mismatch_indices == (1,)
    assert result.completed == 2


@pytest.mark.parametrize("count", [0, 1, -1, 1001, True, 2.5])
def test_invalid_repeat_count_rejected(count):
    client = FakeClient(["A", "A"])

    with pytest.raises(ValueError):
        run_repeatability(client, model="test", prompt="hello", repeats=count)
