# InferenceGuard

**AI Inference Regression Detection & Evidence-Based Verification**

**Version:** `0.1.0a0` | **Development stage:** Alpha | **License:** MIT

InferenceGuard is an open-source Python verification tool for detecting inference-output regressions, evaluating repeatability, and generating reproducible engineering evidence. It helps AI infrastructure engineers determine **what changed, which requests diverged, and which comparison invariants were violated**—using automated checks rather than manual inspection of generated responses.

The project combines **offline differential verification**, **live HTTP completion repeatability testing**, **generated token-ID comparison logic**, and **documented NVIDIA GPU experimentation**. Its design emphasizes measurable results, inspectable evidence, and incremental verification suitable for engineering and CI workflows.

## Why InferenceGuard?

Identical inference requests can yield different outputs when model revisions, serving configurations, numerical execution paths, or caching behavior change. Detecting such differences is only the first step: engineers also need to identify the affected requests, retain their observations, and separate demonstrable regressions from missing or incompatible evidence.

InferenceGuard provides a focused workflow for these tasks:

- **Detect** differences in repeated responses or reference/candidate observations.
- **Compare** generated text, token IDs, and supported log-probability measurements.
- **Classify** results as `PASS`, `FAIL`, or an evidence/compatibility outcome, as appropriate to the command.
- **Preserve** machine-readable results and documented experimental observations for subsequent investigation.

## Implemented Capabilities

| Capability | Current implementation |
|---|---|
| Offline reference/candidate comparison | Schema and metadata validation; aligned cases and prompt fingerprints |
| Token-level differential verification | Generated token-ID identity checks |
| Log-probability comparison | Optional per-token values with configurable absolute tolerance |
| HTTP repeatability testing | Sequential identical requests to OpenAI-compatible `/v1/completions` endpoints |
| Text comparison | Default repeated-response comparison with mismatch indices |
| Token-ID repeatability logic | Explicit token comparison mode using genuine token IDs from a capable observation source |
| Evidence handling | JSON verification reports and optional HTML reports for offline comparison |
| Automated verification | 55 passing local tests at the documented milestone; Ruff checks and GitHub Actions CI workflows |
| Hardware experiment | Preserved NVIDIA Tesla T4 / vLLM prefix-cache observations and CPU-only evidence checks |

### Verification outcomes

| Outcome | Exit code | Interpretation |
|---|---:|---|
| `PASS` | `0` | All selected checks completed successfully for the supplied evidence |
| `FAIL` | `1` | At least one selected comparison invariant was violated |
| `INCOMPATIBLE` | `2` | Offline runs cannot be compared defensibly because of incompatible metadata or cases |
| `INCONCLUSIVE` | `2` | Repeatability evidence is incomplete, unavailable, or interrupted by a request failure |
| Invalid input or execution error | `2` | Validation or command execution did not complete |

These verdicts concern the selected **observable invariants**, not a judgment of the semantic quality of the model's answers.

## Quick Start

**Requirements:** Python 3.11 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

### 1. Offline differential comparison

```bash
inferenceguard compare \
  --reference examples/reference.json \
  --candidate examples/candidate_fault.json \
  --output reports/result.json \
  --html reports/result.html
```

The example contains a deliberate token mismatch and is expected to return `FAIL` (exit code `1`). Use the reference file for both inputs to exercise a matching comparison (exit code `0`).

The offline checker validates model and tokenizer identifiers/revisions, sampling metadata, case alignment, prompt fingerprints, generated token IDs, and optional per-token log-probabilities.

### 2. Repeated HTTP inference: text mode

Requires an accessible OpenAI-compatible `/v1/completions` endpoint and a valid model name served by that endpoint.

```bash
inferenceguard repeat \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model example-model \
  --prompt "Hello" \
  --repeats 3 \
  --comparison-mode text \
  --output reports/repeatability.json
```

InferenceGuard sends repeated requests with temperature set to zero, compares generated completion text against the first response, and records any mismatching request indices.

### 3. Token-ID repeatability mode

```bash
inferenceguard repeat \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model example-model \
  --prompt "Hello" \
  --repeats 3 \
  --comparison-mode tokens \
  --output reports/token_report.json
```

The underlying comparator supports genuine generated token-ID sequences and can detect a token difference even when the decoded text is identical. The comparison engine and CLI behavior have been verified with synthetic token-capable clients.

**Live adapter integration:** The current HTTP client obtains completion text and finish reasons but does not yet request and parse server-specific generated token-ID fields. Consequently, this mode reports `INCONCLUSIVE` with the current HTTP adapter when token IDs are absent. An upcoming vLLM-compatible adapter enhancement will connect live token acquisition to the existing comparator.

## Engineering Case Study: vLLM Prefix-Cache Repeatability

**Upstream reference:** [vLLM issue #40896 — First request differs from subsequent identical requests at temperature=0](https://github.com/vllm-project/vllm/issues/40896)

The issue describes a repeatability defect in which the first inference result differs from later identical requests when prefix caching is enabled, despite deterministic sampling settings. The report includes an NVIDIA H100 environment.

InferenceGuard addresses **the regression-verification challenges illustrated by this issue** through several implemented mechanisms:

| Engineering challenge | InferenceGuard response | Supporting evidence |
|---|---|---|
| First request differs from later requests | First-response baseline and request-indexed mismatch detection | Automated `A → B → B` tests |
| Decoded output may conceal token-sequence changes | Explicit token-ID comparison mode | Token equality, mismatch, and missing-data tests |
| Cache reuse must be observed, not assumed | Record cached prompt-token counts in the controlled experiment | Real Tesla T4 results |
| Findings must be inspectable | Preserve source observations, configuration and comparison reports | Versioned JSON reports and evidence tests |
| Regression investigation needs reproducible checks | Run repeatability and differential checks with defined conditions | CLI, pytest suite, and documented methodology |

**Engineering outcome:** InferenceGuard implements mechanisms for identifying repeated-request divergence, comparing token-level observations, and producing structured evidence relevant to this class of inference regression. These capabilities address detection and diagnosis; changes to vLLM's internal caching or compute kernels remain an upstream responsibility.

### Why we selected NVIDIA Tesla T4 instead of H100

The original report used NVIDIA H100 hardware. For InferenceGuard's first controlled hardware experiment, we selected the **NVIDIA Tesla T4 available through Google Colab Free**, for three concrete reasons:

1. **Real inference execution:** T4 supports CUDA inference and was sufficient to run Qwen3-0.6B with FP16 and exercise vLLM prefix caching.
2. **A focused test objective:** The initial goal was to validate controlled request repetition, actual cache reuse, token-ID capture, and evidence preservation—not to claim hardware-identical reproduction of an H100-specific failure.
3. **Resource-efficient reproducibility:** The experiment used ₹0 GPU expenditure, with scripts and JSON observations saved persistently so further verification could run on a CPU.

The GPUs differ in architecture (Turing versus Hopper), compute capability, precision support, and available kernels. These differences matter for exact bug reproduction; accordingly, the T4 result is a **verified hardware baseline for our methodology**, while evaluating the original H100 failure requires a matching environment and test configuration.

### Real-GPU experiment: setup and results

| Parameter | Recorded configuration |
|---|---|
| GPU | NVIDIA Tesla T4 |
| Model | `Qwen/Qwen3-0.6B` |
| vLLM | `0.33.0` |
| Precision | FP16 |
| Temperature | `0.0` |
| Prompt length | 377 tokens |
| Output limit | 24 generated tokens per request |
| Runs | Three sequential requests per cache configuration |

| Request | Prefix cache OFF | Prefix cache ON |
|---|---:|---:|
| 1 (cold) | 0 / 377 tokens cached | 0 / 377 tokens cached |
| 2 (warm) | 0 / 377 tokens cached | **368 / 377** tokens cached |
| 3 (warm) | 0 / 377 tokens cached | **368 / 377** tokens cached |

**Observed findings:**

- Prefix caching was genuinely exercised: both cache-enabled warm requests reported 368 cached prompt tokens.
- All three cache-disabled requests produced identical generated token IDs.
- All three cache-enabled requests produced identical generated token IDs.
- Token-ID sequences were identical across both configurations.
- The original observations, experiment script, and comparison report passed offline evidence-verification tests.

The experiment established **output consistency alongside measured cache reuse on the tested T4 configuration**. It did not reproduce the reported H100 divergence and therefore does not establish whether that upstream failure is fixed.

**Evidence:** [Tesla T4 experiment, method, raw observations, and validation](experiments/t4_qwen3_0_6b/README.md).

## Engineering Progress and Next Integrations

| Engineering area | Achieved | Next integration |
|---|---|---|
| Differential comparison | Token and log-probability checks with metadata validation | Additional supported evidence types |
| Repeated HTTP inference | Live sequential text acquisition and comparison | Server-specific genuine token-ID acquisition |
| Token-level checks | Tested token-ID comparator and CLI mode | End-to-end vLLM HTTP token verification |
| Prefix-cache investigation | Measured real cache reuse and compared six GPU observations | Automated cache-ON/OFF server configuration workflows |
| Reports | JSON and offline HTML evidence | Broader CI-oriented reporting and workflows |
| Verification | Automated tests, CI checks, preserved experiment evidence | Historical regression cases and cross-backend coverage |
| Provenance | Prompt fingerprints and recorded configuration metadata | Stronger evidence provenance and optional signing |

### The specific bridge to live token-level verification

The remaining integration requirement is **observability through the serving API**, not a new token comparison algorithm. Generated token IDs must be returned by the inference server and correctly requested, parsed, validated, and passed to InferenceGuard.

An initial integration can target vLLM's supported token-ID response extension while keeping the default OpenAI-compatible text workflow unchanged. A compatible server can then supply real token observations to the existing repeatability engine. Other servers will require explicit capability checks rather than assumptions that all OpenAI-compatible APIs expose the same extensions.

Automated cache-configuration comparisons additionally require controlled server settings and recorded run metadata so that results remain meaningful.

## Quality and Reproducibility

Development follows incremental feature branches, pull requests, automated pytest coverage, Ruff lint/format checks, GitHub Actions CI, package-build verification, and dependency auditing.

At the documented milestone, **55 automated tests passed locally**. The test suite includes repeatability scenarios, token-level comparisons, HTTP mock-server checks, CLI behavior, and offline verification of preserved Tesla T4 experimental evidence.

The project uses a deliberately scoped verification contract:

- Temperature zero is a useful controlled setting, but does not universally guarantee identical outputs across serving environments.
- Stochastic generation needs an appropriately designed statistical verification policy; exact token identity can be unsuitable as an unconditional release gate.
- Model outputs are compared against specified invariants rather than judged for semantic correctness.
- Prompt hashes and case IDs assist alignment and provenance, but are not cryptographic attestations.
- GPU experiment results describe the recorded hardware, model, and software configuration, not every possible deployment.

## Documentation

- [Architecture and specification](docs/architecture.md)
- [GitHub repository governance](docs/github-governance.md)
- [Threat model and verification boundaries](docs/threat-model.md)
- [NVIDIA Tesla T4 experimental evidence](experiments/t4_qwen3_0_6b/README.md)
- [Related upstream vLLM issue #40896](https://github.com/vllm-project/vllm/issues/40896)

## License

MIT License.

**InferenceGuard is built to turn inference-output differences into measurable, repeatable, machine-checkable engineering evidence.**
