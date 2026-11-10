# Real vLLM HTTP Token-ID Validation — NVIDIA Tesla T4

## Objective

Validate InferenceGuard's complete HTTP token-verification workflow against a real vLLM inference server running on an NVIDIA Tesla T4.

This experiment extends the earlier CPU-only mock HTTP integration tests by acquiring genuine generated token IDs from a GPU-hosted model.

## Experimental Environment

| Parameter | Configuration |
|---|---|
| GPU | NVIDIA Tesla T4 |
| Execution platform | Google Colab |
| Python | 3.13.16 |
| vLLM | 0.33.0 |
| Model | Qwen/Qwen3-0.6B |
| Precision | FP16 |
| Temperature | 0 |
| Maximum generated tokens | 24 |
| Repeated requests | 3 |
| Comparison mode | Generated token IDs |
| Prefix caching | Enabled |
| API | OpenAI-compatible `/v1/completions` |

The inference server was bound to the local loopback interface. The experiment used InferenceGuard's `return_token_ids` HTTP integration.

## Experimental Method

1. Start the vLLM server with Qwen3-0.6B on the Tesla T4.
2. Wait for the server's HTTP health endpoint.
3. Execute three sequential identical inference requests using InferenceGuard.
4. Request genuine generated token IDs from the server.
5. Compare the first generated token sequence against subsequent responses.
6. Generate the structured verification report.
7. Independently validate the reported verdict, observations, token IDs and mismatch indices.
8. Preserve the experimental evidence.

The experiment runner is `run_experiment.py`.

## Observed Results

| Measurement | Result |
|---|---|
| Requests submitted | 3 |
| Requests completed | 3 |
| Generated tokens per request | 24 |
| Token-sequence mismatches | 0 |
| InferenceGuard verdict | PASS |
| Runner exit code | 0 |
| Independent evidence validation | PASS |

The first eight generated token IDs were identical across all three responses:

`[3555, 525, 279, 7567, 315, 1667, 72349, 44378]`

All complete 24-token sequences were identical.

**Engineering result:** InferenceGuard successfully acquired genuine token-ID observations through the vLLM HTTP interface, compared repeated generations, produced a PASS verdict, and preserved independently verifiable evidence.

## Preserved Evidence

The `evidence/` directory contains:

- `environment.json` — Recorded GPU, model, vLLM version and experiment configuration.
- `repeatability.json` — Generated completion observations, token IDs and comparison result.
- `execution_summary.json` — Experiment execution status.

Full operational logs were retained separately in Google Drive.

## Offline Evidence Verification

From the InferenceGuard repository root:

```bash
python -m pytest tests/test_vllm_http_t4_experiment.py -q
```

The preserved evidence can be checked without CUDA, vLLM or GPU access.

To run the complete repository verification:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

The recorded development milestone is **81 passing automated tests**.

## Reproduction

On a compatible NVIDIA GPU environment with vLLM 0.33.0 and InferenceGuard installed, execute from the repository root:

```bash
python experiments/vllm_http_t4/run_experiment.py
```

The runner launches the local vLLM server, executes the repeatability check and writes results to `experiments/vllm_http_t4/evidence/`.

Model downloads, sufficient GPU resources and a working CUDA environment are required. Preserve the outputs externally before terminating an ephemeral runtime.

## Engineering outcome and next validation

This experiment establishes successful **real-server HTTP token-ID acquisition and repeatability verification** for the recorded configuration. It advances InferenceGuard from simulated HTTP testing to actual GPU-backed model responses.

Prefix caching was enabled in this run, while cache-hit measurements were added in the later [managed-server T4 cache experiment](../vllm_cache_observability_t4/T4_EXPERIMENT_REPORT.md). More model/server configurations and a matched H100 investigation of [vLLM issue #40896](https://github.com/vllm-project/vllm/issues/40896) are future validation opportunities. The PASS verdict applies to the three observed requests rather than every possible deployment.
