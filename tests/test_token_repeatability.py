"""Token-ID repeatability verification using synthetic observations."""

import pytest

from inferenceguard.http_client import InferenceObservation
from inferenceguard.repeatability import run_repeatability


class FakeTokenClient:
    def __init__(self, observations):
        self.observations = iter(observations)

    def generate(self, model, prompt, max_tokens=32):
        return next(self.observations)


def observation(text, token_ids=None):
    return InferenceObservation(
        text=text,
        finish_reason="stop",
        token_ids=token_ids,
    )


def run_tokens(*observations):
    return run_repeatability(
        FakeTokenClient(observations),
        model="test-model",
        prompt="test-prompt",
        comparison_mode="tokens",
    )


def test_identical_token_ids_pass():
    result = run_tokens(
        observation("A", (10, 20, 30)),
        observation("A", (10, 20, 30)),
        observation("A", (10, 20, 30)),
    )

    assert result.verdict == "PASS"
    assert result.mismatch_indices == ()


def test_different_token_ids_fail_even_when_text_matches():
    result = run_tokens(
        observation("Same text", (10, 20, 30)),
        observation("Same text", (10, 21, 30)),
        observation("Same text", (10, 20, 30)),
    )

    assert result.verdict == "FAIL"
    assert result.mismatch_indices == (1,)


def test_different_token_sequence_lengths_fail():
    result = run_tokens(
        observation("A", (10, 20)),
        observation("A", (10, 20, 30)),
        observation("A", (10, 20)),
    )

    assert result.verdict == "FAIL"
    assert result.mismatch_indices == (1,)


def test_missing_token_ids_are_inconclusive():
    result = run_tokens(
        observation("A", (10, 20)),
        observation("A"),
        observation("A", (10, 20)),
    )

    assert result.verdict == "INCONCLUSIVE"
    assert result.error is not None


def test_identical_tokens_with_different_text_pass_in_token_mode():
    result = run_tokens(
        observation("A", (10, 20)),
        observation("B", (10, 20)),
        observation("A", (10, 20)),
    )

    assert result.verdict == "PASS"


def test_default_text_mode_preserves_existing_behavior():
    client = FakeTokenClient(
        [
            observation("A", (10, 20)),
            observation("A", (10, 21)),
            observation("A", (10, 20)),
        ]
    )

    result = run_repeatability(
        client,
        model="test-model",
        prompt="test-prompt",
    )

    assert result.verdict == "PASS"


@pytest.mark.parametrize("mode", ["invalid", "", None])
def test_invalid_comparison_mode_rejected(mode):
    client = FakeTokenClient([])

    with pytest.raises(ValueError):
        run_repeatability(
            client,
            model="test-model",
            prompt="test-prompt",
            comparison_mode=mode,
        )
