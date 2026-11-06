"""Minimal OpenAI-compatible completion client for verification experiments."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit


class InferenceTransportError(Exception):
    """The inference request failed at the transport or HTTP layer."""


class InferenceResponseError(Exception):
    """The endpoint returned an invalid completion response."""


@dataclass(frozen=True)
class InferenceObservation:
    text: str
    finish_reason: str | None
    token_ids: tuple[int, ...] | None = None


class HttpInferenceClient:
    """Client for the OpenAI-compatible /v1/completions endpoint."""

    MAX_RESPONSE_BYTES = 1024 * 1024

    def __init__(self, endpoint: str, timeout_seconds: float = 30.0) -> None:
        if not isinstance(endpoint, str):
            raise ValueError("endpoint must be a URL string")

        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path != "/v1/completions"
        ):
            raise ValueError("endpoint must be an HTTP(S) /v1/completions URL")

        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("invalid endpoint port") from exc

        if port is not None and port == 0:
            raise ValueError("invalid endpoint port")

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")

        self.endpoint = endpoint
        self.timeout_seconds = float(timeout_seconds)

    def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 32,
    ) -> InferenceObservation:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
            }
        ).encode("utf-8")

        # Only HTTP(S) endpoints with the expected path are accepted by __init__.
        request = urllib.request.Request(  # noqa: S310
            self.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=self.timeout_seconds
            ) as response:
                raw = response.read(self.MAX_RESPONSE_BYTES + 1)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            raise InferenceTransportError("Inference HTTP request failed") from exc

        if len(raw) > self.MAX_RESPONSE_BYTES:
            raise InferenceResponseError("Response exceeds maximum size")

        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise InferenceResponseError("Response is not valid JSON") from exc

        if not isinstance(result, dict):
            raise InferenceResponseError("Response must be a JSON object")

        choices = result.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise InferenceResponseError("Expected exactly one choice")

        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("index") != 0:
            raise InferenceResponseError("Invalid completion choice")

        text = choice.get("text")
        reason = choice.get("finish_reason")

        if not isinstance(text, str):
            raise InferenceResponseError("Completion text must be a string")
        if reason is not None and not isinstance(reason, str):
            raise InferenceResponseError("Invalid finish_reason")

        return InferenceObservation(text=text, finish_reason=reason)
