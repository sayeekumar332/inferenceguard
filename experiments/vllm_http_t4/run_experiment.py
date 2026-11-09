"""Real vLLM HTTP token-ID integration experiment.

Run from the InferenceGuard repository root on a Colab GPU runtime.
"""

import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from importlib.metadata import version
from pathlib import Path

MODEL = "Qwen/Qwen3-0.6B"
BASE_URL = "http://127.0.0.1:8000"
OUTPUT_DIR = Path("experiments/vllm_http_t4/evidence")


def gpu_name():
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi executable not found")

    result = subprocess.run(  # noqa: S603
        [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def wait_for_server(process):
    for _ in range(180):
        if process.poll() is not None:
            raise RuntimeError("vLLM server exited before becoming ready")

        try:
            # BASE_URL is a fixed HTTP loopback address, not user-provided input.
            with urllib.request.urlopen(  # noqa: S310
                f"{BASE_URL}/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass

        time.sleep(2)

    raise RuntimeError("vLLM server did not become ready")


def validate_report(report, exit_code):
    """Validate genuine token-level experiment evidence."""
    verdict = report.get("verdict")
    observations = report.get("observations")

    if report.get("requested") != 3 or report.get("completed") != 3:
        raise RuntimeError("Expected three completed inference requests")

    if not isinstance(observations, list) or len(observations) != 3:
        raise RuntimeError("Expected three inference observations")

    sequences = []

    for index, observation in enumerate(observations, start=1):
        if not isinstance(observation, dict):
            raise RuntimeError(f"Invalid observation {index}")

        token_ids = observation.get("token_ids")

        if not isinstance(token_ids, list):
            raise RuntimeError(f"Missing token IDs in observation {index}")

        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in token_ids
        ):
            raise RuntimeError(f"Invalid token IDs in observation {index}")

        sequences.append(token_ids)

    expected_mismatches = [
        index for index in range(1, len(sequences)) if sequences[index] != sequences[0]
    ]

    expected_verdict = "FAIL" if expected_mismatches else "PASS"
    expected_exit = 1 if expected_mismatches else 0

    if verdict != expected_verdict or exit_code != expected_exit:
        raise RuntimeError("Verdict or exit code disagrees with token evidence")

    if report.get("mismatch_indices") != expected_mismatches:
        raise RuntimeError("Mismatch indices disagree with token evidence")

    if report.get("error") is not None:
        raise RuntimeError("Unexpected inference error in experiment report")

    print("Evidence validation: PASS")
    print("Token comparison verdict:", verdict)
    print("Generated token counts:", [len(ids) for ids in sequences])
    print("Mismatch indices:", expected_mismatches)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "gpu": gpu_name(),
        "model": MODEL,
        "vllm_version": version("vllm"),
        "precision": "float16",
        "temperature": 0,
        "repeats": 3,
        "comparison_mode": "tokens",
        "prefix_caching": "enabled",
    }

    (OUTPUT_DIR / "environment.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    vllm_executable = shutil.which("vllm")
    if vllm_executable is None:
        raise RuntimeError("vLLM CLI executable not found")

    command = [
        vllm_executable,
        "serve",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--dtype",
        "float16",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.75",
        "--enable-prefix-caching",
    ]

    with (OUTPUT_DIR / "server.log").open("w") as log:
        server = subprocess.Popen(  # noqa: S603
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

        try:
            wait_for_server(server)

            inferenceguard_command = [
                sys.executable,
                "-m",
                "inferenceguard.cli",
                "repeat",
                "--endpoint",
                f"{BASE_URL}/v1/completions",
                "--model",
                MODEL,
                "--prompt",
                "Explain why deterministic inference verification is important.",
                "--repeats",
                "3",
                "--comparison-mode",
                "tokens",
                "--max-tokens",
                "24",
                "--output",
                str(OUTPUT_DIR / "repeatability.json"),
            ]

            result = subprocess.run(  # noqa: S603
                inferenceguard_command,
                check=False,
            )

            print("InferenceGuard exit code:", result.returncode)

            if result.returncode not in (0, 1, 2):
                raise RuntimeError("Unexpected InferenceGuard exit code")

            report_path = OUTPUT_DIR / "repeatability.json"
            if not report_path.is_file():
                raise RuntimeError("InferenceGuard report was not generated")

            report = json.loads(report_path.read_text(encoding="utf-8"))
            print("Verdict:", report["verdict"])
            print("Completed requests:", report["completed"])
            validate_report(report, result.returncode)

        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
