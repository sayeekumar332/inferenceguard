# InferenceGuard — Threat Model and Evidence Integrity

**Scope:** offline differential verification, HTTP inference repeatability, JSON/HTML reporting, and published experiment evidence.

## 1. Security and reliability objectives

Protect the trustworthiness of comparison results and prevent untrusted inputs, network endpoints, or published evidence from misleading users or compromising the environment running the CLI.

## 2. Assets

- Comparison verdicts, mismatch indices, input policies, and generated reports.
- Model, tokenizer, sampling, prompt fingerprint, and hardware metadata used to interpret observations.
- Captured generated text, token IDs, log-probabilities, and experiment files.
- CI credentials, user workstation/network access, and inference endpoints used by `repeat`.

## 3. Trust boundaries

### Offline inputs and provenance

User-supplied JSON and collector-produced metadata are untrusted. The `compare` workflow validates structure, values, sizes, and cross-run compatibility. SHA-256 content fingerprints can reveal changed bytes, but do **not** authenticate an artifact producer or establish truthful collection. Prompt fingerprints and case IDs are not attestations.

### HTTP inference endpoint

The `repeat` command makes outbound requests to a user-supplied HTTP(S) `/v1/completions` URL. The client validates the endpoint URL form, rejects embedded credentials and URL query/fragment components, uses a finite positive request timeout, and bounds the bytes read from the response. Responses are untrusted and require validation before conversion to observations.

**Network-access consideration:** a URL supplied by an untrusted user can target reachable internal or local services. Path and scheme validation alone are not SSRF prevention; deployment environments should restrict destinations appropriately. Plain HTTP provides no transport encryption. TLS server authentication depends on a correctly configured HTTPS environment; it is not a substitute for endpoint authorization policy.

### Output and publication

Machine-readable JSON and human-readable HTML reports can contain sensitive generated content or identifying experiment metadata. HTML report text must remain escaped. GitHub publication of experimental data requires review for secrets and private prompts.

## 4. Implemented safeguards

- Strict versioned offline schema with constraints on file size, case count, token lengths, numeric finiteness and metadata fields.
- Explicit comparison compatibility and evidence-limited verdicts.
- HTML escaping of externally supplied case identifiers and reasons.
- HTTP(S) URL structure checks and rejection of URL userinfo, query and fragment; timeout and response-size limits.
- Validation of the expected completion response shape before accepting text observations.
- Separation of unavailable token evidence (`INCONCLUSIVE`) from successful token verification (`PASS`).
- Minimal runtime dependencies and automated regression tests for key negative cases.

The current HTTP adapter accepts HTTP as well as HTTPS and does not provide a built-in server allowlist, authentication credential management, or comprehensive endpoint isolation. These are integration and deployment considerations rather than controls already provided.

## 5. Risks and next engineering controls

| Risk or requirement | Current approach / next control |
|---|---|
| Malicious or private-network URL | Validate configured server destinations; add opt-in hostname/IP policy and documented deployment restrictions where appropriate |
| Sensitive prompts and outputs | Redaction/retention rules and explicit publication review |
| Collector falsification | Authenticated or signed manifests and trusted collector provenance in future work |
| Unsupported token-ID extensions | Strictly validate genuine server-generated token-ID arrays; return `INCONCLUSIVE` if missing rather than reconstructing IDs from text |
| Credential exposure | Dedicated secret handling and headers only after defining an authenticated-endpoint security design |
| Backend drift | Pin model revisions and record effective serving, tokenizer, sampler, kernel, and cache configuration |
| Response / resource exhaustion | Continue bounded parsing; consider retry policy, rate limiting, and resource controls for expanded workflows |
| Supply chain and CI | Continue dependency auditing; evaluate GitHub Actions SHA pinning and an SBOM when appropriate |
| Overconfident verification verdicts | Report exact supported invariants, test conditions and evidence coverage |
| Broader statistical behavior | Design stochastic sampling and false-positive tests before using token identity as a general release criterion |

## 6. What the real-GPU experiment demonstrates

A Tesla T4 experiment with vLLM 0.33.0 and Qwen3-0.6B recorded real prefix-cache reuse (368/377 cached prompt tokens on two warm requests) and identical generated token IDs across six requests. The scripts and observations are published with CPU-only verification tests.

The result supports repeatability claims for that specific configuration, but is not an H100-specific reproduction of [vLLM issue #40896](https://github.com/vllm-project/vllm/issues/40896). The upstream issue concerns vLLM behavior; InferenceGuard addresses its regression detection and evidence-collection aspects.

## 7. Reporting discipline

Describe what each test or experiment measured, and keep results distinct from universal guarantees. A successful comparison establishes that the selected observed invariants held; it is not a proof of semantic correctness, security certification, or absence of every regression.
