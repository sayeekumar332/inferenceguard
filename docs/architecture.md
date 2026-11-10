# InferenceGuard — Architecture and Verification Specification

**Version:** `0.1.0a0` · **Scope:** evidence-based AI inference regression detection · **Milestone:** P5.2 (170 passing tests)

## 1. Engineering objective and implemented solution

InferenceGuard detects observable differences between reference and candidate inference evidence, and between repeated requests sent to the same inference endpoint. It produces explicit verdicts and machine-readable reports, helping engineering teams identify the request that diverged and the evidence supporting that result.

The related [vLLM issue #40896](https://github.com/vllm-project/vllm/issues/40896) illustrates the value of first-request/warm-request verification. InferenceGuard provides an independent regression-detection and cache-observability layer, with verification boundaries and potential extensions explained later in this document.

## 2. Implemented verification workflows

### 2.1 Offline differential verification: `inferenceguard compare`

`reference.json` + `candidate.json` → `core.load_run` and validation → `core.compare_runs` → JSON report (optional HTML).

- Validates versioned schema, numeric constraints, case IDs, prompt fingerprints, model/tokenizer identifiers and revisions, and sampling metadata.
- Checks generated token-ID equality and optional aligned per-token log-probabilities with a configurable absolute tolerance.
- Returns `PASS`, `FAIL`, or `INCOMPATIBLE`; invalid inputs yield exit code 2.
- Reports canonical SHA-256 content fingerprints, which detect content changes but do not authenticate the collector.
- Escapes externally provided values in HTML.

### 2.2 Repeated HTTP verification: `inferenceguard repeat`

User-configured `/v1/completions` endpoint → `HttpInferenceClient.generate` → `InferenceObservation` → `run_repeatability` → JSON report.

- Issues sequential non-streaming requests using temperature 0.
- Supports `--comparison-mode text` (default) and `--comparison-mode tokens`.
- In **token mode**, sends `return_token_ids: true`, validates genuine nonnegative integer token IDs in the `choices[0].token_ids` response field, and compares sequences against the first completed response. The capability depends on server compatibility.
- Does **not** infer missing token IDs by retokenizing response text. Missing token observations produce `INCONCLUSIVE`.
- Reports zero-based request mismatch indices. Transport, invalid-response, or required-evidence failures can result in `INCONCLUSIVE`.
- Emits JSON for HTTP runs; HTML output remains specific to offline `compare`.

### 2.3 Optional prefix-cache observation: `--observe-prefix-cache`

`HttpInferenceClient` → `CacheObservedClient` → requests plus snapshots from `prefix_cache_metrics.fetch_prefix_cache_metrics` → `cache_observability` report section.

- Reads vLLM Prometheus counters `vllm:prefix_cache_queries` and `vllm:prefix_cache_hits` (including supported `_total` forms).
- Attempts an initial metric snapshot, then a post-request snapshot for each completed request; computes interval query/hit deltas and hit-rate fractions.
- Enforces nonnegative integral counters, rejects duplicate or malformed series, and records invalid intervals/counter resets.
- Sets observation status to `AVAILABLE`, `PARTIAL`, or `UNAVAILABLE`, including per-request `deltas`, `errors`, and `initial_error`.
- Missing metrics are represented by `null`, not zero. Cache status is **independent of the inference verdict**.
- Counters may be server-wide: isolated workloads are necessary for defensible per-request attribution.

## 3. Principal implementation components

| Component | Responsibility |
|---|---|
| `src/inferenceguard/core.py` | Offline schema validation, compatibility, comparison policy, provenance digests |
| `src/inferenceguard/http_client.py` | Bounded HTTP completion requests, optional token-ID request and strict response parsing |
| `src/inferenceguard/repeatability.py` | Sequential repeated requests, text/token comparison, mismatch and inconclusive verdicts |
| `src/inferenceguard/prefix_cache_metrics.py` | Prometheus counter parsing, secure bounded metrics retrieval, delta calculation |
| `src/inferenceguard/cache_observation.py` | Metrics snapshots around inference calls and independent observability errors |
| `src/inferenceguard/cli.py` | CLI argument handling, orchestration, verdict/exit-code mapping, JSON and offline HTML |
| `demos/customer_demo/` | Three HTTP-backed simulated demonstration scenarios |
| `experiments/vllm_cache_observability_t4/` | Managed real-vLLM GPU experiment runner, report and archived JSON evidence |
| `experiments/t4_qwen3_0_6b/` | Earlier separate GPU experiment with cache-enabled/cache-disabled comparisons |
| `tests/` | Unit, integration-style mock HTTP, negative-path, demonstration and evidence tests |

### Data flow

```text
             Offline reference/candidate files
                           |
                  validation + compare
                           |
                    evidence report

/v1/completions -- HttpInferenceClient --> repeatability engine
                        |                     |
               optional CacheObservedClient  |
                        |                     |
                     /metrics                 |
                        |                     |
               cache observations --------> JSON report
```

## 4. Offline input contract (`schema_version: "1.0"`)

A run has exactly `schema_version`, `metadata`, and `cases`.

- Metadata includes nonempty model/tokenizer identifiers and revisions, plus a nonempty sampling object.
- Cases have unique IDs, nonempty prompt fingerprints, up to 8,192 nonnegative token IDs each, and optional finite aligned log-probabilities.
- Limits: 10,000 cases and an 8 MiB input file.
- Prompt normalization and hashing policy belong to the evidence collector; a prompt hash does not attest to the collector's trustworthiness.
- Experiment-specific T4 JSON is **not** automatically compatible with the offline input schema without conversion.

## 5. Verdict and observability contracts

| Workflow | Outcome | Exit code | Meaning |
|---|---|---:|---|
| Offline `compare` | `PASS` / `FAIL` / `INCOMPATIBLE` | 0 / 1 / 2 | Matching evidence / selected difference / incompatible run metadata |
| HTTP `repeat` | `PASS` / `FAIL` / `INCONCLUSIVE` | 0 / 1 / 2 | Matching observations / divergence / insufficient evidence |
| Either | Input or execution error | 2 | Verification could not be completed |

`cache_observability.status` is a **separate** report field, not a verdict that overrides `PASS` or `FAIL`. CI policies that require cache readings must enforce this separately.

## 6. Real GPU experiments — separate, noninterchangeable evidence

### 6.1 Earlier cache ON/OFF Tesla T4 comparison

The initial experiment used the vLLM Python API and Qwen3-0.6B on Tesla T4, observing 368/377 cached prompt tokens on two warm cache-enabled requests, with equal generated token-ID sequences in the recorded cache-on/cache-off trials. See [earlier experiment methodology](../experiments/t4_qwen3_0_6b/README.md).

### 6.2 Newer managed-server Tesla T4 integration

A separate controlled experiment ran **vLLM 0.33.0**, **Qwen/Qwen3-0.6B**, **FP16**, temperature 0, three calls, with prefix caching enabled.

| Request | Query tokens | Cache-hit tokens |
|---|---:|---:|
| Cold request | 563 | 0 |
| Warm request 1 | 563 | 560 |
| Warm request 2 | 563 | 560 |
| **Totals** | **1,689** | **1,120** |

All three requests completed and their generated **24-token ID sequences matched**. The aggregate cache-hit ratio from the recorded counters was **66.31%**. This is **not** a latency improvement measurement. Archived results passed independent internal consistency checks; raw before/after Prometheus samples were not saved. See [managed-server T4 experiment report](../experiments/vllm_cache_observability_t4/T4_EXPERIMENT_REPORT.md).

Neither T4 study reproduces the H100-specific report from vLLM issue #40896. The T4 runtime also recorded a FlashAttention 2 compatibility fallback and a shutdown semaphore warning, documented in the experiment report.

## 7. Automated tests and customer demos

Following P5.2, **170 tests passed**, with Ruff lint and formatting checks clean. Three CPU-runnable local HTTP demonstrations exercise the InferenceGuard CLI with **simulated** token IDs:

1. Healthy repeated responses → inference `PASS`.
2. A deliberately changed token ID in response 2 → inference `FAIL`, index `[1]`, even though response text stays unchanged.
3. Unavailable `/metrics` → inference `PASS`, cache observability `UNAVAILABLE`, null deltas and explicit errors.

These are controlled simulated cases, **not real model inference**. See [customer demo guide](../demos/customer_demo/README.md).

## 8. Next engineering extensions

Implemented capabilities include real HTTP token-ID acquisition, opt-in metrics observation, a managed-server GPU experiment, and reproducible simulated failure demonstrations. Future work includes:

- Broader model/backend compatibility tests and explicit capability negotiation.
- Controlled comparisons of independently configured cache ON/OFF **HTTP servers**.
- High-concurrency/distributed workload evaluation and robust per-request metrics attribution.
- Stochastic-output statistical policies and false-positive characterization.
- Stronger provenance, authenticated evidence, and operational hardening.

Validate each expansion with new targeted tests and representative runtime evidence before claiming support.
