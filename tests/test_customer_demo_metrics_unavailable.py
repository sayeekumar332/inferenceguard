"""End-to-end customer demo for unavailable prefix-cache metrics."""

import json
import subprocess
import sys
from pathlib import Path


def test_metrics_unavailable_customer_demo(tmp_path):
    report_path = tmp_path / "metrics_unavailable.json"
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(  # noqa: S603 - trusted interpreter and fixed module
        [
            sys.executable,
            "-m",
            "demos.customer_demo.metrics_unavailable_demo",
            "--output",
            str(report_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert report_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))

    # Inference correctness remains independently verified.
    assert report["verdict"] == "PASS"
    assert report["requested"] == 3
    assert report["completed"] == 3
    assert report["mismatch_indices"] == []
    assert report["error"] is None

    observations = report["observations"]
    assert len(observations) == 3

    token_ids = [item["token_ids"] for item in observations]
    assert token_ids[0] == token_ids[1] == token_ids[2]

    # Cache observability is explicitly unavailable.
    cache = report["cache_observability"]

    assert cache["status"] == "UNAVAILABLE"
    assert cache["initial_error"] is not None
    assert cache["deltas"] == [None, None, None]
    assert len(cache["errors"]) == 3
    assert all(error is not None for error in cache["errors"])
