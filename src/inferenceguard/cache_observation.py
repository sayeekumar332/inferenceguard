"""Observe prefix-cache activity during repeated inference requests."""

from collections.abc import Callable

from inferenceguard.http_client import InferenceObservation
from inferenceguard.prefix_cache_metrics import calculate_cache_delta
from inferenceguard.repeatability import CompletionClient


class CacheObservedClient:
    """Wrap inference requests with independent cache observability.

    Each completed inference request has exactly one delta entry
    and one corresponding error entry.

    Metrics failures do not alter the inference result.
    """

    def __init__(
        self,
        client: CompletionClient,
        metrics_reader: Callable[[], dict[str, int]],
    ) -> None:
        self.client = client
        self.metrics_reader = metrics_reader

        self.deltas: list[dict[str, int | float | None] | None] = []
        self.errors: list[str | None] = []

        self.initial_error: str | None = None
        self._previous: dict[str, int] | None = None

        try:
            self._previous = self.metrics_reader()
        except (OSError, RuntimeError, ValueError) as exc:
            self.initial_error = f"Initial metrics unavailable: {type(exc).__name__}"

    def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 32,
    ) -> InferenceObservation:
        """Execute inference and independently measure cache activity."""

        # Inference errors propagate to run_repeatability().
        observation = self.client.generate(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
        )

        try:
            current = self.metrics_reader()

        except (OSError, RuntimeError, ValueError) as exc:
            self.deltas.append(None)
            self.errors.append(f"Metrics unavailable: {type(exc).__name__}")
            self._previous = None
            return observation

        if self._previous is None:
            # Validate the new snapshot before accepting it as a baseline.
            try:
                calculate_cache_delta(current, current)
            except ValueError as exc:
                self.deltas.append(None)
                self.errors.append(f"Invalid metrics snapshot: {exc}")
                return observation

            self.deltas.append(None)
            self.errors.append(self.initial_error or "Previous metrics snapshot unavailable")

            self.initial_error = None
            self._previous = current
            return observation

        try:
            delta = calculate_cache_delta(self._previous, current)

        except ValueError as exc:
            self.deltas.append(None)
            self.errors.append(f"Invalid cache interval: {exc}")

            # A valid counter reset establishes a new baseline.
            # Other invalid snapshots must not become baselines.
            if "reset" in str(exc).lower():
                self._previous = current
            else:
                self._previous = None

            return observation

        self.deltas.append(delta)
        self.errors.append(None)
        self._previous = current

        return observation
