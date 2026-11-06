# InferenceGuard — NVIDIA Tesla T4 Prefix-Cache Experiment

## 1. Objective

Evaluate whether repeated inference requests produce identical generated token sequences when vLLM prefix caching is enabled or disabled.

This experiment provides real-GPU baseline evidence for InferenceGuard's inference-regression verification work.

## 2. Experimental environment

| Component | Configuration |
|---|---|
| Cloud platform | Google Colab Free |
| GPU | NVIDIA Tesla T4 |
| Compute capability | 7.5 |
| Model | Qwen/Qwen3-0.6B |
| Inference engine | vLLM 0.33.0 |
| Precision | FP16 |
| Maximum model length | 1024 tokens |
| GPU memory utilization | 0.65 |
| Execution mode | Eager |
| Sampling temperature | 0.0 |
| Maximum generated tokens | 24 |
| Requests per configuration | 3 |

The experiment uses separate vLLM engine processes for prefix-cache-enabled and prefix-cache-disabled configurations.

## 3. Methodology

A common 377-token input prompt was submitted three times sequentially to each vLLM engine.

The experiment recorded the input prompt, its SHA-256 hash, generated text, generated token IDs, prompt-token counts and number of cached prompt tokens.

Two independent configurations were evaluated:

- **Cache OFF:** `enable_prefix_caching=False`
- **Cache ON:** `enable_prefix_caching=True`

The generated token sequences were compared within each configuration and across both configurations.

## 4. Results

| Request | Cache OFF | Cache ON |
|---|---:|---:|
| 1 — Cold | 0/377 cached tokens | 0/377 cached tokens |
| 2 — Warm | 0/377 cached tokens | 368/377 cached tokens |
| 3 — Warm | 0/377 cached tokens | 368/377 cached tokens |

**Observed findings:**

- All three cache-disabled requests generated identical token IDs.
- All three cache-enabled requests generated identical token IDs.
- Generated token sequences were identical across both configurations.
- Warm requests reused 368 of 377 prompt tokens when prefix caching was enabled.
- No generated-token mismatches were detected.

## 5. Reproduction

The experiment requires a compatible Linux NVIDIA GPU, vLLM 0.33.0, and access to the Qwen3-0.6B model.

Run the experiment using an environment with vLLM installed:

```bash
python run_prefix_cache_experiment.py --cache off --output long_prefix_cache_off.json
python run_prefix_cache_experiment.py --cache on --output long_prefix_cache_on.json
```

The included JSON reports are the captured observations from the original Tesla T4 execution.

For offline verification from the InferenceGuard repository root:

```bash
python -m pytest tests/test_gpu_experiment_evidence.py -v
```

No GPU is required to validate these saved reports.

## 6. Evidence files

| File | Purpose |
|---|---|
| `run_prefix_cache_experiment.py` | Reproducible experiment script |
| `long_prefix_cache_off.json` | Cache-disabled observations |
| `long_prefix_cache_on.json` | Cache-enabled observations |
| `long_prefix_comparison.json` | Cross-configuration comparison |

## 7. Scope and limitations

This experiment demonstrates output consistency under the tested Tesla T4 configuration. It does not establish universal deterministic inference behavior.

The experiment does not reproduce or resolve the reported H100-specific vLLM issue #40896.

The model identifier was recorded, but the exact Hugging Face model commit revision was not explicitly pinned. This limits strict reproduction of the environment.

The experiment used vLLM's standalone Python API, not InferenceGuard's HTTP CLI. The saved evidence is validated using separate offline pytest checks.

The reported cached-token measurements demonstrate actual prompt-prefix reuse during the cache-enabled runs. They do not guarantee identical cache behavior across other models, GPUs or versions.

## 8. Conclusion

The experiment demonstrated that prefix-cache reuse did not change generated token IDs across six controlled inference requests on the NVIDIA Tesla T4.

This provides a real-GPU reference dataset for future InferenceGuard development and cross-platform regression investigations.
