"""Repeated-request verification for deterministic inference experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from inferenceguard.http_client import (
    InferenceObservation,
    InferenceResponseError,
    InferenceTransportError,
)


class CompletionClient(Protocol):
    """Interface implemented by HttpInferenceClient."""

    def generate(self, model: str, prompt: str, max_tokens: int = 32) -> InferenceObservation: ...


@dataclass(frozen=True)
class RepeatabilityResult:
    """Evidence collected from one repeated-request experiment."""

    verdict: Literal["PASS", "FAIL", "INCONCLUSIVE"]
    requested: int
    completed: int
    observations: tuple[InferenceObservation, ...]
    mismatch_indices: tuple[int, ...]
    error: str | None


def run_repeatability(
    client: CompletionClient,
    *,
    model: str,
    prompt: str,
    repeats: int = 3,
    max_tokens: int = 32,
    comparison_mode: Literal["text", "tokens"] = "text",
) -> RepeatabilityResult:
    """Compare repeated requests using text or generated token IDs.

    The first successful response is the baseline.
    Missing token IDs make token comparison inconclusive.
    Request failures preserve the existing inconclusive behavior.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise ValueError("repeats must be an integer")
    if not 2 <= repeats <= 1000:
        raise ValueError("repeats must be between 2 and 1000")

    if comparison_mode not in ("text", "tokens"):
        raise ValueError("comparison_mode must be 'text' or 'tokens'")

    observations: list[InferenceObservation] = []
    mismatches: list[int] = []

    for index in range(repeats):
        try:
            observation = client.generate(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
            )
        except (InferenceTransportError, InferenceResponseError) as exc:
            return RepeatabilityResult(
                verdict="INCONCLUSIVE",
                requested=repeats,
                completed=len(observations),
                observations=tuple(observations),
                mismatch_indices=tuple(mismatches),
                error=f"Request {index + 1}: {type(exc).__name__}",
            )

        if observations:
            baseline = observations[0]

            if comparison_mode == "text":
                if observation.text != baseline.text:
                    mismatches.append(index)

            elif (
                baseline.token_ids is not None
                and observation.token_ids is not None
                and observation.token_ids != baseline.token_ids
            ):
                mismatches.append(index)

        observations.append(observation)

    if comparison_mode == "tokens":
        missing = [
            index + 1
            for index, observation in enumerate(observations)
            if observation.token_ids is None
        ]

        if missing:
            return RepeatabilityResult(
                verdict="INCONCLUSIVE",
                requested=repeats,
                completed=len(observations),
                observations=tuple(observations),
                mismatch_indices=tuple(mismatches),
                error=f"Token IDs unavailable for requests: {missing}",
            )

    return RepeatabilityResult(
        verdict="FAIL" if mismatches else "PASS",
        requested=repeats,
        completed=len(observations),
        observations=tuple(observations),
        mismatch_indices=tuple(mismatches),
        error=None,
    )
