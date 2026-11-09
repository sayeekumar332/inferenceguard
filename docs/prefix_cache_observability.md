# Prefix-Cache Observability

## Overview

InferenceGuard supports opt-in prefix-cache observability during repeated HTTP inference requests.

The feature compares generated outputs while independently recording prefix-cache activity from the inference server.

## Usage

```bash
python -m inferenceguard.cli repeat \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model Qwen/Qwen3-0.6B \
  --prompt "Explain deterministic inference verification." \
  --repeats 3 \
  --comparison-mode tokens \
  --observe-prefix-cache \
  --output report.json
```

## Metrics

InferenceGuard reads the following vLLM counters:

- `vllm:prefix_cache_queries`
- `vllm:prefix_cache_hits`

Prometheus `_total` suffixes are supported.

Cache-hit rate is calculated as:

`delta_hits / delta_queries`

When the query delta is zero, the hit rate is undefined and reported as `null`.

## JSON Report

When enabled, the report includes `cache_observability` with:

- `status`: AVAILABLE, PARTIAL, or UNAVAILABLE
- `initial_error`: Initial metrics collection failure, if any
- `deltas`: Per-request cache counter differences
- `errors`: Per-request metrics collection errors

Without `--observe-prefix-cache`, the original report format is preserved.

## Failure Handling

Metrics acquisition failures do not change the inference-verification verdict.

Missing measurements are recorded as `null`, not as zero cache hits.

Counter resets invalidate the affected measurement interval.

## Security

The metrics client:

- Accepts HTTP and HTTPS base URLs.
- Rejects HTTP redirects.
- Limits metrics responses to 1 MiB.
- Applies an HTTP timeout.
- Rejects malformed metrics.

Only trusted and authorized inference servers should be used.

## Limitations

Prefix-cache counters are server-wide measurements. Other inference workloads may contribute to the observed activity.

Per-request attribution is most reliable when the experiment runs without unrelated inference traffic.

The parser currently expects one matching time series for each required counter.

The integrated functionality has been validated with CPU-only automated tests. A real-GPU experiment using the combined feature has not yet been performed.
