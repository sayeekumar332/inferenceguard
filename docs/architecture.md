# InferenceGuard — Architecture and Verification Specification

**Version:** 0.1.0a0 · **Project focus:** evidence-based AI inference regression detection

## 1. Engineering problem

Inference-serving changes can introduce observable differences between a baseline and a candidate run, or between repeated identical requests. InferenceGuard makes selected differences measurable and reviewable through strict comparison rules, explicit verdicts, and persisted reports.

The project is motivated in part by [vLLM issue #40896](https://github.com/vllm-project/vllm/issues/40896), which describes first-request versus subsequent-request divergence with prefix caching enabled at temperature zero. InferenceGuard addresses regression **detection and evidence collection**; the upstream issue is separate from any fix to vLLM internals.

## 2. Implemented workflows

### 2.1 Offline differential verification (`inferenceguard compare`)

`reference.json` + `candidate.json` → `core.load_run` / `core.validate_run` → `core.compare_runs` → JSON report and optional escaped HTML report.

- Validates versioned input schema, numeric constraints, case identifiers, prompt fingerprints, model/tokenizer IDs and revisions, and sampling metadata.
- Checks token-ID equality under the default policy and optional per-token log-probabilities within a configurable absolute tolerance.
- Returns `PASS`, `FAIL`, or `INCOMPATIBLE`. Invalid input is reported as an execution/validation error (exit code 2).
- Adds SHA-256 canonical content digests to the CLI report as useful local fingerprints, not producer authentication.

### 2.2 HTTP repeated-request verification (`inferenceguard repeat`)

User-supplied `/v1/completions` endpoint → `HttpInferenceClient.generate` → `InferenceObservation` → `run_repeatability` → JSON report.

- Sends identical sequential HTTP POST requests to an OpenAI-compatible completion endpoint, with temperature 0 and nonstreaming completion requests.
- Supports `--comparison-mode text` (default) and `--comparison-mode tokens`.
- Compares successful responses against the first response; mismatch indices are zero-based.
- Returns `PASS`, `FAIL`, or `INCONCLUSIVE` when required observations are missing or a request fails.
- `InferenceObservation` carries `text`, `finish_reason`, and optional original `token_ids`.
- **Live acquisition boundary:** the current HTTP client extracts text and finish reason but does not request or parse token IDs from server-specific extensions. Therefore token-mode requests through this adapter are `INCONCLUSIVE` when IDs are absent. Token-mode comparison logic has been tested with synthetic token-capable clients.
- HTTP repeat reports are JSON; HTML report generation currently belongs to the offline `compare` workflow.

## 3. Principal modules

| Module | Responsibility |
|---|---|
| `core.py` | Strict offline schema validation, comparison policy, canonical content digest |
| `http_client.py` | Bounded HTTP completion client, response validation, observation structure |
| `repeatability.py` | Sequential repeat requests, text/token comparison, mismatch and inconclusive results |
| `cli.py` | `compare`/`repeat` commands, exit-code mapping, JSON and offline HTML output |
| `tests/` | Unit, mock-HTTP, CLI, security-focused, repeatability, and experimental evidence checks |
| `experiments/t4_qwen3_0_6b/` | Standalone real-GPU experiment script, recorded data, comparison and methodology |

## 4. Offline input contract (`schema_version: "1.0"`)

An offline run contains exactly `schema_version`, `metadata`, and `cases`.

- `metadata`: nonempty `model_id`, `model_revision`, `tokenizer_id`, `tokenizer_revision`, and nonempty `sampling` object.
- Each case: unique `id`, nonempty `prompt_hash`, `tokens` (up to 8,192 nonnegative integer IDs), and `logprobs` (null or finite numeric values aligned with tokens).
- Maximum case count: 10,000; maximum input file size: 8 MiB.
- A prompt fingerprint is supplied by the collection workflow. Collectors should document how prompt bytes are normalized and hashed. InferenceGuard does not establish that a collector is authentic.
- The offline T4 experimental JSON has a **separate experiment-specific format**; it is not automatically an input to `inferenceguard compare` without conversion to the versioned offline schema.

## 5. Verdict contracts

| Workflow | Verdict | Exit | Interpretation |
|---|---|---:|---|
| Offline `compare` | `PASS` | 0 | Selected invariants match |
| Offline `compare` | `FAIL` | 1 | Selected invariant differs |
| Offline `compare` | `INCOMPATIBLE` | 2 | Metadata/case sets do not support a defensible comparison |
| HTTP `repeat` | `PASS` | 0 | Selected available comparison dimension is repeatable |
| HTTP `repeat` | `FAIL` | 1 | Selected comparison dimension differs |
| HTTP `repeat` | `INCONCLUSIVE` | 2 | Response failure or required token data unavailable |
| Either | Input/operational error | 2 | Command could not complete a valid comparison |

A `PASS` is evidence about selected measurements, not proof of model semantic correctness or universal determinism.

## 6. Real-GPU verification reference

An independent vLLM Python API experiment ran on a free NVIDIA Tesla T4 with `Qwen/Qwen3-0.6B`, vLLM 0.33.0, FP16, temperature 0, and three sequential requests per prefix-cache configuration.

| Request | Cache disabled | Cache enabled |
|---|---:|---:|
| 1 | 0/377 | 0/377 |
| 2 | 0/377 | 368/377 |
| 3 | 0/377 | 368/377 |

All six recorded generated token-ID sequences matched. This demonstrates measured cache reuse and repeatability for the tested T4 configuration, not reproduction or remediation of the issue reported on NVIDIA H100. The exact model commit revision was not pinned. Consult [experiment methodology](../experiments/t4_qwen3_0_6b/README.md).

Tesla T4 was selected because it was accessible via Google Colab Free and sufficient to exercise the small-model inference and actual prefix reuse at zero GPU expenditure. H100-specific behavior requires an appropriately matched H100 experiment.

## 7. Acceptance criteria and demonstrated testing

- Identical offline runs pass; injected token or log-probability mismatches fail.
- Incompatible model/tokenizer/sampling metadata and case sets produce `INCOMPATIBLE`.
- Malformed and oversized offline evidence is rejected; HTML escapes supplied values.
- HTTP repeated text matches pass; first-versus-warm mismatch scenarios fail with identified indices; transport failures are inconclusive.
- Token-ID equality/difference, missing IDs, and CLI report compatibility are covered by synthetic tests.
- T4 JSON observations and the comparison report are checked offline.
- Recorded project checkpoint: **55 local tests passing** with Ruff checks; use current CI runs as authoritative evidence of branch-specific checks.

## 8. Next integration boundary

**Live vLLM token-ID acquisition:** opt-in request for a vLLM-supported token-ID response extension, strict parsing of actual returned token IDs, correct handling of absent/unsupported data, and mock-server tests. Real vLLM HTTP integration may be verified in a single resource-limited Tesla T4 session if free access is available.

**Automated cache-ON/OFF orchestration:** compare independently controlled server configurations with documented model revision, tokenizer, sampling, hardware, cache status, and effective kernel/backend settings. Do not equate enabling prefix caching with observed cache reuse.

**Future verification research:** concurrency perturbation, statistical evaluation for stochastic sampling, false-positive characterization, and stronger evidence provenance. These are separate milestones from the implemented comparison engine.
