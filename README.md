# InferenceGuard

**AI Inference Regression Detection & Evidence-Based Verification**

**Version:** 0.1.0a0 | **Maturity:** Alpha — Actively Developed

InferenceGuard is an open-source Python verification tool that helps AI infrastructure engineers detect inference-output regressions, compare reference and candidate observations, and generate reproducible verification evidence.

Designed for inference regression testing, debugging, and CI workflows, InferenceGuard combines offline differential verification, HTTP-based repeatability testing, token-level comparison logic, and documented real-GPU experiments.

## Key Capabilities

- **Offline differential verification:** Compare reference and candidate inference observations, including generated token IDs and optional log-probabilities.
- **HTTP repeatability testing:** Send identical requests to OpenAI-compatible completion endpoints and detect output differences.
- **Token-ID comparison:** Detect token-sequence changes even when decoded output text is identical.
- **Structured verification reports:** Generate machine-readable JSON reports and HTML reports for offline comparisons.
- **Automated verification:** Run regression tests and CI checks using pytest, Ruff, and GitHub Actions.
- **Experimental evidence:** Inspect recorded NVIDIA Tesla T4 inference experiments with verified prefix-cache reuse.

## Quick Start

### Requirements

Python 3.11 or later.

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

### Run Automated Tests

```bash
python -m pytest -q
```

### Offline Inference Comparison

```bash
inferenceguard compare \
  --reference examples/reference.json \
  --candidate examples/candidate_fault.json \
  --output reports/result.json \
  --html reports/result.html
```

The example contains an intentionally injected token mismatch and should produce `FAIL` with exit code `1`.

For a passing comparison, use the reference file as both the reference and candidate.

### HTTP Inference Repeatability

Requires a running OpenAI-compatible `/v1/completions` endpoint.

```bash
inferenceguard repeat \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model example-model \
  --prompt "Hello" \
  --repeats 3 \
  --comparison-mode text \
  --output reports/repeatability.json
```

This command sends three identical inference requests sequentially with temperature set to zero and compares their generated text.

## Token-ID Repeatability Comparison

InferenceGuard provides two comparison modes:

| Mode | Description |
|---|---|
| `text` | Compare decoded completion text (default) |
| `tokens` | Compare genuine generated token-ID sequences |

The token-ID comparison algorithm is implemented and tested.

**Current adapter limitation:** The standard HTTP client does not extract generated token IDs from completion responses. Consequently, selecting token mode with this adapter returns `INCONCLUSIVE` when token IDs are unavailable.

Token comparison has been verified using synthetic token-capable clients. The offline `compare` command separately supports token-ID comparison from recorded observations.

## Real-GPU Experimental Evidence

InferenceGuard includes a documented inference-repeatability experiment executed on an NVIDIA Tesla T4 GPU using vLLM.

| Component | Configuration |
|---|---|
| GPU | NVIDIA Tesla T4 |
| Model | Qwen/Qwen3-0.6B |
| Inference engine | vLLM 0.33.0 |
| Precision | FP16 |
| Temperature | 0.0 |
| Total requests | 6 |

### Experimental Results

| Request | Prefix Cache OFF | Prefix Cache ON |
|---|---:|---:|
| 1 | 0/377 cached tokens | 0/377 cached tokens |
| 2 | 0/377 cached tokens | 368/377 cached tokens |
| 3 | 0/377 cached tokens | 368/377 cached tokens |

**Observed outcome:** All six requests generated identical token-ID sequences despite confirmed prefix-cache reuse.

The experiment demonstrates output consistency for the tested configuration. It does not reproduce or resolve the H100-specific vLLM issue #40896.

The experiment was conducted through vLLM's Python API, not the InferenceGuard HTTP CLI.

[View experiment methodology, original observations and verification evidence](experiments/t4_qwen3_0_6b/README.md).

## Verification Outcomes

| Verdict | Exit Code | Meaning |
|---|---:|---|
| `PASS` | 0 | Selected comparison checks passed |
| `FAIL` | 1 | A selected comparison check failed |
| `INCOMPATIBLE` | 2 | Offline observations cannot be defensibly compared |
| `INCONCLUSIVE` | 2 | Repeatability verification lacks sufficient evidence or encountered a request failure |
| Invalid input | 2 | Validation or execution could not complete |

These verdicts describe the implemented comparison checks, not the semantic correctness of generated responses.

## Engineering and Quality Practices

InferenceGuard is developed using:

- Incremental feature development and regression testing
- GitHub feature branches and pull requests
- Automated pytest test suites
- Ruff lint and formatting validation
- GitHub Actions CI workflows
- Package-build and dependency-audit checks
- Preserved experiment scripts and original evidence
- Documented verification limitations

## Roadmap

| Milestone | Status |
|---|---|
| Offline differential verification | Implemented |
| OpenAI-compatible HTTP inference client | Implemented |
| Repeated-request text comparison | Implemented |
| Token-ID comparison logic | Implemented |
| Explicit CLI comparison modes | Implemented |
| Tesla T4 prefix-cache experiment | Completed |
| Automated offline evidence verification | Implemented |
| Automated cache-ON/OFF inference workflow | Planned |
| Live token-ID acquisition from supported backends | Planned |
| Concurrency-based regression testing | Planned |
| Historical regression reproduction and false-positive evaluation | Planned |
| Evidence provenance enhancements and integrations | Future |

## Documentation

- [Architecture and specification](docs/architecture.md)
- [GitHub repository protections](docs/github-governance.md)
- [Threat model and limitations](docs/threat-model.md)
- [Tesla T4 experimental evidence](experiments/t4_qwen3_0_6b/README.md)

## Scope and Limitations

InferenceGuard currently verifies selected inference-output invariants under specified test conditions.

It does not prove model semantic correctness, guarantee deterministic execution across different hardware or software configurations, or provide cryptographic attestation.

Stochastic inference requires appropriate statistical testing before results can be used as reliable release gates.

**Project maturity:** Alpha. The implemented capabilities are supported by automated tests and documented experiments, but broader production reliability and compatibility have not yet been established.

## License

MIT License.