# InferenceGuard — Threat Model and Evidence Integrity

**Scope:** offline differential comparison, live HTTP completion repeatability, optional prefix-cache metrics retrieval, JSON/HTML reporting, GPU experiments and simulated customer demonstrations.

## 1. Security and reliability objectives

Protect the integrity of reported comparison outcomes and reduce the risk that untrusted data, network endpoints, or published evidence mislead users or compromise the environment running the CLI.

Verification claims must match the **specific data available** and must not turn unobserved metrics into fabricated measurements.

## 2. Assets

- Comparison verdicts, mismatch indices, cache-status indicators, per-request metric deltas, and generated reports.
- Model/tokenizer/sampling, hardware and execution configuration metadata.
- Prompt fingerprints, generated text, authentic token IDs, log-probabilities and experiment files.
- Workstation network reachability, CI secrets, inference-server endpoints and monitoring interfaces.

## 3. Trust boundaries

### 3.1 Offline files and evidence producers

Offline evidence is untrusted. `compare` validates a versioned schema, bounded sizes, numeric values, case alignment and model/tokenizer compatibility. Canonical SHA-256 digests support content comparison but **do not authenticate** the collector or establish that its metadata is truthful.

The T4 experiment's archived JSON uses a different evidence format from offline `schema_version: "1.0"` comparison input.

### 3.2 HTTP completion requests and token-ID extensions

The HTTP client accepts configured HTTP(S) URLs targeting `/v1/completions`, disallows embedded credentials, queries and fragments, and applies timeout and response-size bounds. It validates the returned completion structure.

In token mode the adapter sets `return_token_ids: true` and validates `choices[0].token_ids` as nonnegative integers. It does not reconstruct IDs by retokenizing text. **Missing token IDs yield inconclusive token evidence**, not a token-level PASS.

**SSRF exposure:** URL structure checks do not prevent a malicious or mistaken endpoint from pointing to a reachable internal resource. Isolate runners, restrict allowed destinations, and use only trusted servers. HTTP carries no transport encryption; HTTPS certificate validation does not itself authorize the destination.

The client does not currently provide a managed credential store, configurable authenticated headers, or comprehensive network isolation.

### 3.3 Prefix-cache metrics

Opt-in monitoring also makes requests to `/metrics` at the server origin. This introduces an additional network-read and input-parsing surface.

Implemented mitigations include bounded 1 MiB response reads, finite timeouts, redirect rejection, HTTP(S) base URL validation and strict counter parsing. The metrics collector rejects missing or ambiguous counter series, malformed values, negative deltas and reset/invalid intervals.

Cache values may be **server-wide**. Per-request interval changes can include unrelated traffic unless the environment is isolated. Metrics failures are recorded as unavailable, independently of inference `PASS`/`FAIL`.

### 3.4 Reports, demos and published evidence

JSON/HTML reports and archived observations may expose model responses, prompt content or identifying environmental details. The offline HTML renderer escapes externally provided text, but publication still requires a deliberate secrets/privacy review.

The three P5.2 customer demonstrations return **simulated** responses and token IDs. Claims about those demos must not imply the IDs came from an actual model. Their role is reproducible positive and negative-path testing.

## 4. Implemented safeguards

- Strict bounded offline schema validation and compatibility rules.
- Finite HTTP timeouts, bounded response reads, endpoint structure checks and completion response validation.
- Explicit opt-in request for genuine server-generated token IDs.
- `INCONCLUSIVE` for missing required token observations rather than fabricated IDs.
- Independent cache observability `AVAILABLE` / `PARTIAL` / `UNAVAILABLE` statuses, null missing intervals and categorized errors.
- Validated cache-counter deltas, malformed-series rejection and rejected metrics redirects.
- HTML escaping for human-readable offline reports.
- Automated positive and negative-path tests, protected-branch PR workflow, lint/format checks.

## 5. Remaining risks and candidate controls

| Risk | Current boundary / proposed improvement |
|---|---|
| Private-network SSRF or unauthorized server access | Run with explicit network restrictions; future destination allowlisting |
| Sensitive prompts, outputs, metrics or logs | Data-minimization/redaction rules, controlled artifact retention and publication review |
| Untrusted evidence provenance | Future signed/verified producer manifests; content hashes alone are insufficient |
| Unsupported token-ID extensions | Explicit backend capability detection and robust cross-backend compatibility tests |
| Missing metrics / false zero-hit claims | Keep independent cache status, per-request error details and explicit `null` values |
| Server-wide metric contamination | Dedicated inference environment, workload isolation and attribution validation |
| Backend/configuration drift | Pin and record model revisions, software, CUDA/kernel settings, sampling and cache options |
| Resource exhaustion | Preserve bounded parsing; consider rate limits, retries and broader resource policies |
| CI or supply-chain compromise | Dependency auditing, least-privilege automation, evaluate action pinning and SBOM |
| Overgeneralizing deterministic results | Add cross-platform, high-concurrency and stochastic verification research |

## 6. Hardware-backed evidence and its limits

**Earlier T4 Python API study:** a cache-ON/OFF comparison observed 368/377 cached prompt tokens for warm cache-enabled calls and matching generated token-ID sequences in its recorded trials. [Study](../experiments/t4_qwen3_0_6b/README.md).

**Newer managed-server T4 integration study:** vLLM 0.33.0 and Qwen/Qwen3-0.6B yielded 3/3 matching 24-token responses and recorded cache-query/hit totals of 1,689 / 1,120, a 66.31% token-level cache-hit ratio. Archived results passed internal consistency validation. [Study](../experiments/vllm_cache_observability_t4/T4_EXPERIMENT_REPORT.md).

These are **different experiments**. The newer report explicitly notes the absence of retained raw before-and-after Prometheus snapshots and records attention-backend and shutdown diagnostics. Neither study reproduces or resolves [vLLM issue #40896](https://github.com/vllm-project/vllm/issues/40896), originally reported on a different GPU environment. None establishes universal determinism, measured speedup, cost reduction or security certification.

The P5.2 local demonstrations are deliberately simulated; they are not additional hardware-backed evidence.

## 7. Reporting discipline

A `PASS` states that the **selected observable comparison invariants** held. It does not certify model semantics or reliability across untested configurations. A separate cache status must be checked whenever cache observability is required by an operational gate.

At the completed P5.2 checkpoint, the repository recorded **170 passing automated tests** and clean Ruff checks; ongoing CI and future milestone measurements must be assessed independently.
