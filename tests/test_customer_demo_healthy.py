"""Tests for the CPU-only healthy inference customer demonstration."""

import json
import subprocess
import sys
from pathlib import Path


def test_healthy_customer_demo(tmp_path):
    report_path = tmp_path / "healthy_report.json"
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(  # noqa: S603 - trusted interpreter and fixed module
        [
            sys.executable,
            "-m",
            "demos.customer_demo.healthy_demo",
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

    assert report["verdict"] == "PASS"
    assert report["requested"] == 3
    assert report["completed"] == 3
    assert report["mismatch_indices"] == []

    observations = report["observations"]
    assert len(observations) == 3

    token_ids = [obs["token_ids"] for obs in observations]

    assert all(len(ids) > 0 for ids in token_ids)
    assert token_ids[0] == token_ids[1] == token_ids[2]
