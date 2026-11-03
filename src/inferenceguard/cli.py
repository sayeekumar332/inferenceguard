"""Command-line entry point for the offline verification MVP."""

from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import asdict
from pathlib import Path

from inferenceguard.core import Policy, ValidationError, canonical_digest, compare_runs, load_run
from inferenceguard.http_client import HttpInferenceClient
from inferenceguard.repeatability import run_repeatability


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inferenceguard")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("compare", help="Compare two offline inference observation files")
    repeat = sub.add_parser(
        "repeat",
        help="Check repeatability of HTTP inference completions",
    )
    repeat.add_argument("--endpoint", required=True)
    repeat.add_argument("--model", required=True)
    repeat.add_argument("--prompt", required=True)
    repeat.add_argument("--repeats", type=int, default=3)
    repeat.add_argument("--max-tokens", type=int, default=32)
    repeat.add_argument("--timeout-seconds", type=float, default=30.0)
    repeat.add_argument("--output", required=True, type=Path)

    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--html", type=Path, default=None, help="Optional HTML report")
    p.add_argument("--logprob-tolerance", type=float, default=1e-3)
    args = parser.parse_args(argv)
    if args.command == "repeat":
        try:
            client = HttpInferenceClient(
                endpoint=args.endpoint,
                timeout_seconds=args.timeout_seconds,
            )
            result = run_repeatability(
                client,
                model=args.model,
                prompt=args.prompt,
                repeats=args.repeats,
                max_tokens=args.max_tokens,
            )
            report = asdict(result)

            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        except (OSError, ValueError) as exc:
            print(f"InferenceGuard error: {exc}", file=sys.stderr)
            return 2

        print(
            f"{result.verdict}: {result.completed}/{result.requested} "
            f"requests completed; report={args.output}"
        )
        return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[result.verdict]
    try:
        policy = Policy(logprob_abs_tolerance=args.logprob_tolerance)
        reference, candidate = load_run(args.reference), load_run(args.candidate)
        report = compare_runs(reference, candidate, policy)
        report["provenance"] = {
            "reference_sha256": canonical_digest(reference),
            "candidate_sha256": canonical_digest(candidate),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if args.html:
            args.html.parent.mkdir(parents=True, exist_ok=True)
            args.html.write_text(render_html(report), encoding="utf-8")
    except (ValidationError, OSError, ValueError) as exc:
        print(f"InferenceGuard error: {exc}", file=sys.stderr)
        return 2
    print(
        f"{report['verdict']}: {report['summary']['checked']} case(s), "
        f"{report['summary']['failed']} failing; report={args.output}"
    )
    return {"PASS": 0, "FAIL": 1, "INCOMPATIBLE": 2}[report["verdict"]]


def render_html(report: dict) -> str:
    """Static report, escaping externally supplied identifiers and reasons."""
    rows = "\n".join(
        "<tr><td>"
        + html.escape(case["id"])
        + "</td><td>"
        + html.escape(case["status"])
        + "</td><td>"
        + html.escape("; ".join(case["reasons"]))
        + "</td></tr>"
        for case in report["cases"]
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>InferenceGuard Verification Report</title>"
        "<style>body{font:16px system-ui;max-width:850px;margin:3rem auto;padding:1rem}"
        "table{border-collapse:collapse;width:100%}td,th{padding:.6rem;border:1px solid #aaa}"
        "</style></head><body><h1>InferenceGuard Verification Report</h1><h2>"
        + html.escape(report["verdict"])
        + "</h2><table><thead><tr>"
        "<th>Case</th><th>Status</th><th>Evidence</th></tr></thead><tbody>"
        + rows
        + "</tbody></table></body></html>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
