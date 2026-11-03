# InferenceGuard v0.1 Architecture and Acceptance Criteria

## Customer problem

Model-serving deployments can yield apparently valid but incorrect outputs after runtime or configuration changes. The initial scope is a locally executable *differential observation checker*, not a complete inference certification system.

## Modules

- `core.validate_run`: strict, versioned JSON schema and finite-numeric validation.
- `core.compare_runs`: comparator and explicit comparison policy.
- `core.canonical_digest`: provenance fingerprint over normalized input content.
- `cli.main`: operational exit codes and reports.
- `cli.render_html`: escaped static evidence report.
- `tests/`: self-checking success, fault-injection, invalid-input, injection-resistance tests.

## Input schema

`schema_version: '1.0'`; `metadata`: model ID/revision, tokenizer ID/revision, sampling object. `cases` array contains unique `id`, `prompt_hash`, `tokens` list, and `logprobs` list or null.

`prompt_hash` is a user-supplied fingerprint; collectors must calculate it from canonical prompt bytes and document normalization. `sampling` must describe all effective generation settings and should not use defaults that differ across runtimes. Do not compare unrelated prompt sets.

## MVP acceptance

1. Identical runs pass.
2. Injected token mismatch fails.
3. Altered prompt fingerprint fails.
4. Model/tokenizer/sampling mismatch is incompatible.
5. Log-probability change beyond configured tolerance fails.
6. Malformed JSON and nonfinite metrics fail closed.
7. HTML output escapes external content.
8. CI checks code and tests across supported Python versions.

## Next module / Day 2

Add a versioned `InferenceAdapter` protocol and a deterministic fixture adapter. Do not claim vLLM interoperability until exercised against a real vLLM endpoint. Before implementing concurrency/cache comparisons, specify whether those comparisons should preserve identical token sequences, logits tolerance, or task-level invariants.
