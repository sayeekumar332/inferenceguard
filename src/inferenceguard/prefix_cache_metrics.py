"""Parse vLLM prefix-cache metrics from Prometheus text."""

import re
import urllib.error
import urllib.request
from urllib.parse import urlsplit

METRIC_PATTERN = re.compile(
    r"^(vllm:prefix_cache_(queries|hits)(?:_total)?)"
    r"(?:\{[^}]*\})?\s+(\S+)$"
)

METRIC_PREFIX = "vllm:prefix_cache_"


def parse_prefix_cache_metrics(metrics_text: str) -> dict[str, int]:
    """Extract one non-negative integer counter for queries and hits.

    Reject missing, invalid, or duplicate matching metric series.
    Metrics with and without the Prometheus '_total' suffix are accepted.
    """
    if not isinstance(metrics_text, str):
        raise ValueError("Metrics response must be text")

    counters: dict[str, int] = {}

    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if not line.startswith(METRIC_PREFIX):
            continue

        match = METRIC_PATTERN.fullmatch(line)

        if match is None:
            raise ValueError(f"Malformed prefix-cache metric: {line}")

        metric_type = match.group(2)
        raw_value = match.group(3)

        if metric_type in counters:
            raise ValueError(f"Duplicate prefix-cache metric: {metric_type}")

        if not raw_value.isdecimal():
            raise ValueError(f"Invalid prefix-cache counter: {raw_value}")

        counters[metric_type] = int(raw_value)

    if "queries" not in counters or "hits" not in counters:
        raise ValueError("Missing required prefix-cache query or hit counter")

    return counters


def calculate_cache_delta(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, int | float | None]:
    """Calculate prefix-cache activity between two counter snapshots.

    Returns query and hit token deltas, plus a hit-rate fraction
    between 0.0 and 1.0. A zero-query interval has no defined rate.

    Rejects invalid counters, counter resets, and impossible deltas.
    """

    for snapshot_name, snapshot in (
        ("before", before),
        ("after", after),
    ):
        if not isinstance(snapshot, dict):
            raise ValueError(f"Invalid {snapshot_name} snapshot")

        for counter in ("queries", "hits"):
            value = snapshot.get(counter)

            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid {counter} counter in {snapshot_name} snapshot")

        if snapshot["hits"] > snapshot["queries"]:
            raise ValueError(f"Cache hits exceed queries in {snapshot_name} snapshot")

    query_delta = after["queries"] - before["queries"]
    hit_delta = after["hits"] - before["hits"]

    if query_delta < 0 or hit_delta < 0:
        raise ValueError("Prefix-cache counter reset detected")

    if hit_delta > query_delta:
        raise ValueError("Cache hit delta exceeds query delta")

    hit_rate = hit_delta / query_delta if query_delta > 0 else None

    return {
        "queries": query_delta,
        "hits": hit_delta,
        "hit_rate": hit_rate,
    }


MAX_METRICS_RESPONSE_BYTES = 1024 * 1024  # 1 MiB


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Prevent the metrics client from following HTTP redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_prefix_cache_metrics(
    base_url: str,
    timeout: float = 5.0,
) -> dict[str, int]:
    """Retrieve and validate prefix-cache metrics from a vLLM server.

    Only HTTP(S) base URLs are accepted.
    Redirects are rejected.
    Responses are limited to 1 MiB.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("Invalid metrics server URL")

    parsed = urlsplit(base_url)

    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Expected an HTTP(S) server base URL")

    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not (0 < timeout <= 60)
    ):
        raise ValueError("Timeout must be between 0 and 60 seconds")

    endpoint = base_url.rstrip("/") + "/metrics"

    # Use a dedicated opener; do not change global urllib settings.
    opener = urllib.request.build_opener(_RejectRedirects())

    try:
        with opener.open(  # noqa: S310
            endpoint, timeout=timeout
        ) as response:
            payload = response.read(MAX_METRICS_RESPONSE_BYTES + 1)

    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise RuntimeError(f"Unable to retrieve prefix-cache metrics: {exc}") from exc

    if len(payload) > MAX_METRICS_RESPONSE_BYTES:
        raise ValueError("Prefix-cache metrics response exceeds maximum size")

    try:
        metrics_text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("Metrics response is not UTF-8") from exc

    return parse_prefix_cache_metrics(metrics_text)
