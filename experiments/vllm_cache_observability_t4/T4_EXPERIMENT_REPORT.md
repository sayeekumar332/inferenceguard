# InferenceGuard — NVIDIA Tesla T4 Experiment Report

## 1. Objective

Evaluate InferenceGuard against a real vLLM inference server to verify:

1. Repeatability of generated token IDs across identical deterministic inference requests.
2. Observability of vLLM prefix-cache queries and hits.
3. Consistency of recorded inference and cache evidence.

This experiment demonstrates the combined token-repeatability and prefix-cache measurement workflow on a real NVIDIA Tesla T4 inference server.

## 2. Experimental Environment

| Parameter | Configuration |
|---|---|
| GPU | NVIDIA Tesla T4 |
| Python | 3.13.16 |
| vLLM | 0.33.0 |
| Model | Qwen/Qwen3-0.6B |
| Precision | FP16 |
| Temperature | 0 |
| Maximum generated tokens | 24 |
| Maximum model context | 2048 |
| Requests | 3 |
| Comparison mode | Token IDs |
| Prefix caching | Enabled |
| GPU memory utilization setting | 0.75 |

The server was launched locally on `127.0.0.1:8000` through the controlled experiment runner.

## 3. Methodology

The experiment runner:

1. Starts the vLLM server with a reproducible configuration.
2. Checks the server's `/health` endpoint.
3. Verifies the availability of prefix-cache metrics from `/metrics`.
4. Issues three identical inference requests using deterministic sampling.
5. Records returned generated token IDs.
6. Records prefix-cache query and hit counter deltas for each request.
7. Validates token-level repeatability and cache observability.
8. Preserves experiment metadata and results.
9. Shuts down the managed server.

The reported cache-hit values are derived from the vLLM Prometheus counters and are not estimates based on response latency.

## 4. Token Repeatability Results

| Measurement | Observed result |
|---|---|
| Requested inference calls | 3 |
| Completed inference calls | 3 |
| Generated tokens per call | 24 |
| Token-ID mismatches | 0 |
| Experiment verdict | PASS |

All three recorded generated token-ID sequences were identical.

The requests terminated at the configured 24-token generation limit.

**Conclusion:** Token-level repeatability was observed across the three requests in this configuration.

## 5. Prefix-Cache Observability Results

| Request | Query tokens | Cache-hit tokens | Cache-hit rate |
|---|---:|---:|---:|
| First | 563 | 0 | 0.00% |
| Second | 563 | 560 | 99.47% |
| Third | 563 | 560 | 99.47% |
| **Total** | **1,689** | **1,120** | **66.31%** |

The first request recorded no prefix-cache hits. Both subsequent requests recorded 560 cache-hit tokens.

**Conclusion:** The experiment observed positive prefix-cache reuse on repeated identical requests.

The overall 66.31% cache-hit ratio is a token-level counter ratio, not a measured latency improvement or computing-cost reduction.

## 6. Independent Evidence Validation

The recorded JSON artifacts were separately parsed and checked.

The validation confirmed:

- Three requests were completed.
- All three generated token-ID sequences contained 24 tokens.
- The sequences matched exactly.
- No mismatch indices were reported.
- Cache-query deltas were `[563, 563, 563]`.
- Cache-hit deltas were `[0, 560, 560]`.
- Total query tokens equaled 1,689.
- Total hit tokens equaled 1,120.
- The aggregate cache-hit ratio was 66.31%.

This establishes internal consistency of the archived results. Raw before-and-after Prometheus snapshots were not retained, so the reported cache-counter deltas cannot be independently reconstructed from raw metric samples.

## 7. Runtime Diagnostics

The vLLM server log recorded two notable diagnostics:

**FlashAttention 2 compatibility:** The NVIDIA Tesla T4 does not meet the compute capability requirement of the FlashAttention 2 implementation considered by vLLM. The server continued and successfully completed the inference requests using an alternative attention backend.

**Shutdown resource warning:** Python reported one leaked semaphore during server shutdown. The inference experiment completed successfully, but cleanup behavior warrants further investigation.

These diagnostics are not evidence of token mismatch or invalid cache measurements, but they remain relevant operational limitations.

## 8. Evidence Artifacts

The publication includes:

- `evidence/environment.json` — recorded hardware, software, and experiment configuration.
- `evidence/repeatability.json` — inference observations, token IDs, verdict, and cache-counter deltas.

The raw `server.log` is retained separately and excluded from the public repository to avoid unnecessary exposure of environment-specific logs.

## 9. Scope and future experiments

The recorded achievement is **three completed real-model inference calls with identical generated token IDs, together with directly observed cache-counter changes**. This result provides a baseline for expanding verification to additional hardware, models, concurrency patterns, and serving configurations.

The current run used one Tesla T4, one model, temperature zero, and three sequential requests. It did not measure latency improvement or cost savings, test stochastic decoding or distributed workloads, or reproduce the separate H100-based [vLLM issue #40896](https://github.com/vllm-project/vllm/issues/40896). A future matched-environment experiment would be required to investigate that upstream report. The recorded 66.31% figure is a token-counter hit ratio, not a performance improvement.

## 10. Reproduction

From the repository root, in a compatible NVIDIA GPU environment with the project and vLLM installed:

```bash
python -m experiments.vllm_cache_observability_t4.run_experiment
```

The runner manages server startup, readiness checks, inference execution, evidence validation, and server shutdown.

Reproduction requires a compatible Python, PyTorch, CUDA, and vLLM environment and sufficient GPU memory. Results may differ across hardware and software configurations.

## 11. Conclusion

InferenceGuard successfully verified identical generated token IDs across three real vLLM requests and observed 1,120 prefix-cache-hit tokens during the controlled Tesla T4 experiment.

Together with the passing automated regression suite, these results demonstrate functional integration with a real GPU-backed inference server under the tested configuration.

Broader workloads, additional inference backends, and operational hardening remain subjects for subsequent validation.
