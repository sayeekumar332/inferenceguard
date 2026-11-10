# InferenceGuard

**AI Inference Regression Detection & Evidence-Based Verification**

**Version:** `0.1.0a0` | **Stage:** Alpha | **License:** MIT | **Python:** 3.11+

InferenceGuard is an open-source Python verification tool for detecting inference-output regressions, evaluating repeatability, and producing inspectable engineering evidence. It provides **offline reference-versus-candidate comparison**, **live HTTP completion checks**, **generated token-ID verification**, and **optional vLLM prefix-cache observability**.

It helps inference engineers compare outputs across model versions and serving configurations, pinpoint divergent requests, and retain evidence they can review or use in CI. Its focus is repeatable, measurable verification of inference behavior.

## What is implemented

| Capability | Status |
|---|---|
| Offline differential verification | Reference/candidate schema and metadata validation; aligned cases; prompt fingerprints |
| Token-level comparison | Generated token-ID identity and mismatch detection |
| Log-probability comparison | Optional per-token comparison with configurable absolute tolerance |
| Live HTTP repeatability | Sequential repeated calls to OpenAI-compatible `/v1/completions` endpoints |
| Response comparison modes | `text` or `tokens`; genuine token IDs required for token comparison |
| Prefix-cache observability | Optional `--observe-prefix-cache`, reading vLLM Prometheus `/metrics` counters around requests |
| Evidence | Machine-readable JSON reports; optional offline HTML report |
| Customer demonstrations | Three locally reproducible scenarios using simulated HTTP completion responses |
| Hardware verification | Real NVIDIA Tesla T4 / vLLM experiments with archived evidence |
| Automated quality checks | **170 tests passed** on the completed P5.2 milestone; Ruff lint and formatting; GitHub Actions CI |

### Interpretation of outcomes

| Outcome | Exit code | Meaning |
|---|---:|---|
| `PASS` | `0` | Selected inference comparisons passed |
| `FAIL` | `1` | A selected comparison invariant was violated |
| `INCOMPATIBLE` | `2` | Offline evidence or metadata cannot be defensibly compared |
| `INCONCLUSIVE` | `2` | Repeatability evidence was incomplete or unavailable |
| Input/execution error | `2` | Command or validation did not complete |

**Important:** When cache observability is requested, it is a *separate result* within the JSON report. The `cache_observability.status` value can be `AVAILABLE`, `PARTIAL`, or `UNAVAILABLE`. An inference `PASS` can coexist with `UNAVAILABLE` cache metrics; automation that requires metrics must check this status independently.

## Quick start

Requires Python 3.11 or later. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

### 1. Offline reference/candidate comparison

```bash
inferenceguard compare \
  --reference examples/reference.json \
  --candidate examples/candidate_fault.json \
  --output reports/result.json \
  --html reports/result.html
```

The deliberately faulty candidate should produce `FAIL` (exit code `1`). Comparing the reference file against itself should produce `PASS`. The checker validates compatible model/tokenizer and sampling metadata, case alignment, prompt fingerprints, generated token IDs, and optional log-probability observations.

### 2. Live completion repeatability: text

Requires an accessible OpenAI-compatible completion endpoint:

```bash
inferenceguard repeat \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model example-model \
  --prompt "Hello" \
  --repeats 3 \
  --comparison-mode text \
  --output reports/repeatability.json
```

Repeated requests use temperature zero. InferenceGuard compares generated text against the first response and records mismatching request indices.

### 3. Live token-ID repeatability and prefix-cache observability

For a compatible inference endpoint that provides **genuine generated token IDs**:

```bash
inferenceguard repeat \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model example-model \
  --prompt "Hello" \
  --repeats 3 \
  --comparison-mode tokens \
  --observe-prefix-cache \
  --output reports/token_cache_report.json
```

In token mode the HTTP adapter requests and parses supported token-ID responses; token IDs must actually be supplied by the server. Not every OpenAI-compatible backend supports this. Missing token observations result in inconclusive token-level evidence rather than invented IDs.

With `--observe-prefix-cache`, InferenceGuard also attempts to obtain cache-counter deltas from the server's Prometheus `/metrics` endpoint. Failure to read the metrics is recorded explicitly and does not silently change the inference comparison verdict.

## Real NVIDIA Tesla T4 verification

InferenceGuard has been exercised against an actual GPU-backed vLLM server. The **newer managed-server experiment** used:

| Parameter | Recorded value |
|---|---|
| GPU | NVIDIA Tesla T4 |
| Inference server | vLLM `0.33.0` |
| Model | `Qwen/Qwen3-0.6B` |
| Precision | FP16 |
| Temperature | `0` |
| Request count | 3 |
| Generated tokens | 24 per request |
| Prefix caching | Enabled |

### Observed repeatability and cache reuse

| Request | Prefix-cache query tokens | Cache-hit tokens |
|---|---:|---:|
| 1 (cold) | 563 | 0 |
| 2 (warm) | 563 | 560 |
| 3 (warm) | 563 | 560 |
| **Total** | **1,689** | **1,120** |

All three calls completed and returned **identical generated token-ID sequences**: zero mismatches. The recorded aggregate prefix-cache token-hit ratio was **66.31%**. These are Prometheus counter-derived observations, **not** measurements of latency improvements or cost savings.

The experiment report describes setup, managed server lifecycle, source evidence, validation, diagnostics, and limitations. The recorded counters were independently checked for internal consistency; raw before-and-after Prometheus metric snapshots were not retained.

**Primary evidence:** [vLLM 0.33.0 / Tesla T4 experiment report](experiments/vllm_cache_observability_t4/T4_EXPERIMENT_REPORT.md).

**Earlier case study:** [Tesla T4 cache-enabled and cache-disabled comparison](experiments/t4_qwen3_0_6b/README.md) documents a distinct controlled experiment, including cache-on/cache-off output comparison. Do not conflate the measurements from these separate runs.

### Engineering relevance to vLLM regression investigations

[vLLM issue #40896](https://github.com/vllm-project/vllm/issues/40896) illustrates why comparing the first response against later identical requests is valuable. InferenceGuard implements that comparison, detects token-ID differences, and records cache observations to support investigation. Our Tesla T4 experiments demonstrate these verification methods with actual inference and cache reuse under the recorded configurations. The upstream H100 issue is a motivating scenario rather than a failure we claim to have reproduced.

## Reproducible customer demonstrations (P5.2)

Three local scenarios send actual HTTP requests to a **simulated** completion server and exercise the real InferenceGuard CLI. **These scripts do not execute a language model; their token IDs are simulated.** They complement, but do not replace, the real-GPU evidence above.

| Demonstration | Expected inference verdict | Additional observation | Exit |
|---|---|---|---:|
| Healthy inference | `PASS` | Three matching simulated token-ID sequences | `0` |
| Injected token regression | `FAIL` | One changed token ID in response 2, despite unchanged text; mismatch index `[1]` | `1` |
| Unavailable cache metrics | `PASS` | Cache status `UNAVAILABLE`, explicit errors and null deltas | `0` |

Run from the repository root:

```bash
python -m demos.customer_demo.healthy_demo \
  --output /tmp/inferenceguard_healthy_demo.json

python -m demos.customer_demo.regression_demo \
  --output /tmp/inferenceguard_regression_demo.json

python -m demos.customer_demo.metrics_unavailable_demo \
  --output /tmp/inferenceguard_metrics_unavailable.json
```

The regression example **intentionally exits with code 1**. For prerequisites, expected reports, verification commands, and limitations, see the [Customer Demonstration Guide](demos/customer_demo/README.md).

## Verification and development

As of the merged P5.2 milestone, **170 automated tests passed**, along with `ruff check .` and `ruff format --check .`. Development uses feature branches, pull requests, protected `main`, GitHub Actions, and incremental test-driven verification.

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m build
```

## Scope and next validation steps

InferenceGuard is an **independent verification layer** for inference outputs and cache observations. It is not a patch to vLLM internals. The current evidence demonstrates the implemented checks and real-server integration under the documented Tesla T4 configurations.

Future validation priorities include additional model/backend compatibility, high-concurrency and distributed workloads, and cross-GPU repeatability. Token-ID checks require genuine server-provided IDs; missing cache measurements must be treated as unavailable, not as zero reuse. The observed 66.31% cache-hit ratio is a measured token-counter ratio, not a measured speedup or cost saving. Temperature-zero matching on the tested workloads does not establish universal determinism.

## Documentation

- [Architecture and specification](docs/architecture.md)
- [Threat model and verification boundaries](docs/threat-model.md)
- [GitHub repository governance](docs/github-governance.md)
- [Managed-server NVIDIA Tesla T4 experiment](experiments/vllm_cache_observability_t4/T4_EXPERIMENT_REPORT.md)
- [Earlier Tesla T4 experiment](experiments/t4_qwen3_0_6b/README.md)
- [Customer demonstrations](demos/customer_demo/README.md)
- [Related vLLM issue #40896](https://github.com/vllm-project/vllm/issues/40896)

## License

MIT License.

**InferenceGuard turns inference-output differences into measurable, repeatable, machine-checkable engineering evidence.**
