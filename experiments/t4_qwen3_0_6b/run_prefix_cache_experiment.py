import argparse
import hashlib
import json
import multiprocessing
from datetime import UTC, datetime
from pathlib import Path

import torch
import vllm
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", choices=["on", "off"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache_enabled = args.cache == "on"

    prefix = " ".join(
        f"Technical observation {i}: "
        "A deterministic inference experiment compares "
        "identical model inputs and generated token sequences."
        for i in range(1, 21)
    )
    prompt = prefix + "\nConclusion: The central observation is"

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        dtype="float16",
        max_model_len=1024,
        gpu_memory_utilization=0.65,
        enforce_eager=True,
        enable_prefix_caching=cache_enabled,
    )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=24,
    )

    observations = []

    for index in range(3):
        response = llm.generate([prompt], sampling, use_tqdm=False)[0]

        output = response.outputs[0]

        observation = {
            "request": index + 1,
            "prompt_tokens": len(response.prompt_token_ids),
            "cached_tokens": response.num_cached_tokens,
            "text": output.text,
            "token_ids": list(output.token_ids),
        }
        observations.append(observation)

        print(
            f"Request {index + 1}: "
            f"{observation['cached_tokens']}/"
            f"{observation['prompt_tokens']} cached tokens",
            flush=True,
        )

    sequences = [x["token_ids"] for x in observations]

    report = {
        "experiment": "InferenceGuard T4 long-prefix repeatability",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": "Qwen/Qwen3-0.6B",
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "dtype": "float16",
        "prefix_caching": cache_enabled,
        "enforce_eager": True,
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.65,
        "temperature": 0.0,
        "max_tokens": 24,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "observations": observations,
        "all_token_ids_identical": all(sequence == sequences[0] for sequence in sequences[1:]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "All token IDs identical:",
        report["all_token_ids_identical"],
        flush=True,
    )
    print("Report:", args.output, flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
