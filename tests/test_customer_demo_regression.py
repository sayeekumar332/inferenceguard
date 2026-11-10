"""Tests for the CPU-only inference regression demonstration."""

import json
import subprocess
import sys
from pathlib import Path


def test_regression_customer_demo(tmp_path):
    report_path = tmp_path / "regression_report.json"
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(  # noqa: S603 - trusted interpreter and fixed module
        [
            sys.executable,
            "-m",
            "demos.customer_demo.regression_demo",
            "--output",
            str(report_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 1, result.stderr
    assert report_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["verdict"] == "FAIL"
    assert report["requested"] == 3
    assert report["completed"] == 3
    assert report["mismatch_indices"] == [1]
    assert report["error"] is None

    observations = report["observations"]
    assert len(observations) == 3

    assert [item["token_ids"] for item in observations] == [
        [101, 205, 309, 412],
        [101, 205, 999, 412],
        [101, 205, 309, 412],
    ]

    # Generated text is identical despite the token-ID difference.
    assert len({item["text"] for item in observations}) == 1
