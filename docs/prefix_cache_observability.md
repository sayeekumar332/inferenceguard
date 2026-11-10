# InferenceGuard — Prefix-Cache Observability

## Purpose

InferenceGuard can observe vLLM prefix-cache activity **while independently checking inference repeatability**. Metrics are optional: failure to collect metrics is not silently treated as a zero cache-hit measurement and does not change the inference comparison verdict.

## Usage

Point InferenceGuard at a trusted, reachable OpenAI-compatible completion server which exposes its Prometheus metrics on the corresponding `/metrics` endpoint:

```bash
inferenceguard repeat \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model Qwen/Qwen3-0.6B \
  --prompt "Explain deterministic inference verification." \
  --repeats 3 \
  --comparison-mode tokens \
  --observe-prefix-cache \
  --output reports/token_cache_report.json
```

Token comparison additionally requires the serving backend to supply genuine token IDs through the supported completion response extension. Text mode can be used if that extension is unavailable, but it does not establish token-level identity.

## Implementation

The `repeat` CLI constructs an `HttpInferenceClient` for completions. If `--observe-prefix-cache` is enabled, `CacheObservedClient` wraps it and retrieves a metric snapshot before the run and after each completed request.

`prefix_cache_metrics.py` parses one matching series each of:

- `vllm:prefix_cache_queries`
- `vllm:prefix_cache_hits`

The accepted Prometheus counter names may carry the `_total` suffix. The parser rejects missing, malformed, negative, nonintegral or duplicate counters; the delta calculator rejects reset/invalid intervals.

For each valid interval:

`query_delta = query_after - query_before`

`hit_delta = hit_after - hit_before`

`hit_rate = hit_delta / query_delta` when `query_delta > 0`; otherwise the rate is `null`.

These are token-counter-based quantities, **not estimates based on latency**.

## Report schema and status

When observation is requested, the JSON report adds:

```json
{
  "cache_observability": {
    "status": "AVAILABLE",
    "initial_error": null,
    "deltas": [
      {"queries": 563, "hits": 0, "hit_rate": 0.0},
      {"queries": 563, "hits": 560, "hit_rate": 0.9946714031971581},
      {"queries": 563, "hits": 560, "hit_rate": 0.9946714031971581}
    ],
    "errors": [null, null, null]
  }
}
```

This illustrates the **shape and the recorded Tesla T4 cache-counter values**; consult the archived JSON for the authoritative experiment data. Each completed request has a corresponding delta entry or `null`, plus an error entry.

- `AVAILABLE`: every requested inference call completed and has a valid metrics delta.
- `PARTIAL`: some usable cache intervals are present, but the run has missing intervals.
- `UNAVAILABLE`: no usable per-request cache interval is available.
- `initial_error`: error categorization when the first snapshot was unavailable.
- `errors`: categorization for failed post-request measurement or an invalid interval.

If the initial snapshot is missing, the next valid snapshot may establish a baseline without yielding a delta for that request. A reset or invalid snapshot can invalidate that interval. Missing measurements are represented as `null`, not zero.

**The cache status does not override the inference verdict.** A report may contain `verdict: PASS` with `cache_observability.status: UNAVAILABLE` and CLI exit code 0. A pipeline that requires cache observability must examine both report fields.

Without the observation flag, the HTTP repeat report omits the cache section to retain compatibility with prior consumers.

## Verified real GPU experiment

The integrated managed-server workflow was exercised on **NVIDIA Tesla T4**, **vLLM 0.33.0**, and **Qwen/Qwen3-0.6B**, FP16 and temperature 0, with three sequential token-level requests.

| Request | Cache query tokens | Cache-hit tokens | Cache-hit rate |
|---|---:|---:|---:|
| 1 (cold) | 563 | 0 | 0.00% |
| 2 (warm) | 563 | 560 | 99.47% |
| 3 (warm) | 563 | 560 | 99.47% |
| **Total** | **1,689** | **1,120** | **66.31%** |

All three responses contained identical generated token IDs; there were zero mismatches. This demonstrates integration with **actual model inference and actual vLLM metrics** in the reported environment.

[Full Tesla T4 experiment report](../experiments/vllm_cache_observability_t4/T4_EXPERIMENT_REPORT.md) · [Archived results](../experiments/vllm_cache_observability_t4/evidence/repeatability.json)

The evidence report verifies internal consistency of the archived counter deltas. Raw pre/post Prometheus snapshots were not preserved, so the deltas cannot be independently recomputed from raw samples. The reported 66.31% is a token-level cache-hit ratio, **not a measured latency or cost reduction**. The experiment did not reproduce vLLM issue #40896.

## Simulated metrics-unavailable demonstration

A CPU-only local HTTP server supplies simulated completion responses and no valid `/metrics` endpoint:

```bash
python -m demos.customer_demo.metrics_unavailable_demo \
  --output /tmp/inferenceguard_metrics_unavailable.json
```

Expected report: inference `PASS`, cache status `UNAVAILABLE`, `deltas: [null, null, null]`, and explicit initial and per-request metric errors. The simulated demonstration is not a second real-GPU experiment.

[Full customer demonstration guide](../demos/customer_demo/README.md)

## Security, attribution and limitations

- The metrics fetcher accepts HTTP(S) base URLs without credentials, query or fragment, rejects HTTP redirects, imposes a timeout and 1 MiB response limit, and rejects invalid metric text.
- The completion URL is user-supplied; restrict use to authorized endpoints. Input URL validation is **not** SSRF prevention.
- Prefix-cache counters may be server-wide; unrelated concurrent requests can confound per-request attribution.
- Parsing currently requires a single matching time series per counter. Ambiguous/duplicated series are rejected.
- Missing data and counter resets are diagnostic conditions, not evidence of zero reuse.
- Hardware/model/kernel differences and short deterministic runs limit generalization.
