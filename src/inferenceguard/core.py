"""Strict, intentionally small, offline differential verification core.

Not a proof of semantic correctness: checks supported response invariants only.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    """Input is malformed or two experiments cannot be compared."""


@dataclass(frozen=True)
class Policy:
    """Comparison policy; no hidden acceptance thresholds."""

    logprob_abs_tolerance: float = 1e-3
    require_token_identity: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.logprob_abs_tolerance) or self.logprob_abs_tolerance < 0:
            raise ValidationError("logprob_abs_tolerance must be finite and nonnegative")


def load_run(path: str | Path) -> dict[str, Any]:
    """Read an experiment file, rejecting invalid or oversized JSON."""
    p = Path(path)
    if p.stat().st_size > 8 * 1024 * 1024:
        raise ValidationError(f"Input too large (8 MiB maximum): {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Cannot load {p}: {exc}") from exc
    validate_run(data)
    return data


def _reject_constant(value: str) -> None:
    raise ValidationError(f"Non-finite JSON numeric constant: {value}")


def _require_str(obj: dict[str, Any], key: str, context: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{context}.{key} must be a nonempty string")
    return value


def validate_run(data: Any) -> None:
    if not isinstance(data, dict) or set(data) != {"schema_version", "metadata", "cases"}:
        raise ValidationError("Expected schema_version, metadata and cases only")
    if data["schema_version"] != "1.0":
        raise ValidationError("Unsupported schema_version (expected 1.0)")
    meta = data["metadata"]
    if not isinstance(meta, dict):
        raise ValidationError("metadata must be an object")
    for key in ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision", "sampling"):
        if key == "sampling":
            if not isinstance(meta.get(key), dict) or not meta[key]:
                raise ValidationError("metadata.sampling must be a nonempty object")
            for skey, value in meta[key].items():
                if not isinstance(skey, str) or not skey:
                    raise ValidationError("sampling keys must be strings")
                if isinstance(value, (dict, list)) or value is None:
                    raise ValidationError("sampling values must be JSON scalar values")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValidationError("sampling values must be finite")
        else:
            _require_str(meta, key, "metadata")
    cases = data["cases"]
    if not isinstance(cases, list) or not cases or len(cases) > 10_000:
        raise ValidationError("cases must contain 1..10000 items")
    seen = set()
    for idx, item in enumerate(cases):
        ctx = f"cases[{idx}]"
        if not isinstance(item, dict) or set(item) != {"id", "prompt_hash", "tokens", "logprobs"}:
            raise ValidationError(f"{ctx}: expected id, prompt_hash, tokens, logprobs")
        cid = _require_str(item, "id", ctx)
        _require_str(item, "prompt_hash", ctx)
        if cid in seen:
            raise ValidationError(f"Duplicate case ID: {cid}")
        seen.add(cid)
        tokens = item["tokens"]
        if (
            not isinstance(tokens, list)
            or len(tokens) > 8192
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in tokens)
        ):
            raise ValidationError(f"{ctx}.tokens must be up to 8192 nonnegative token IDs")
        probs = item["logprobs"]
        if probs is not None and (
            not isinstance(probs, list)
            or len(probs) != len(tokens)
            or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in probs
            )
        ):
            raise ValidationError(f"{ctx}.logprobs must be null or finite values per token")


def compare_runs(
    reference: dict[str, Any], candidate: dict[str, Any], policy: Policy
) -> dict[str, Any]:
    """Compare aligned observations, failing closed on incompatible metadata."""
    validate_run(reference)
    validate_run(candidate)
    keys = ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision", "sampling")
    mismatched = [
        key for key in keys if reference["metadata"].get(key) != candidate["metadata"].get(key)
    ]
    if mismatched:
        return _result("INCOMPATIBLE", [], [f"Metadata differs: {', '.join(mismatched)}"], policy)
    refs = {item["id"]: item for item in reference["cases"]}
    cands = {item["id"]: item for item in candidate["cases"]}
    if refs.keys() != cands.keys():
        return _result("INCOMPATIBLE", [], ["Case ID sets differ"], policy)
    outcomes = []
    for case_id in sorted(refs):
        a, b = refs[case_id], cands[case_id]
        failures = []
        if a["prompt_hash"] != b["prompt_hash"]:
            failures.append("Prompt fingerprint differs")
        if policy.require_token_identity and a["tokens"] != b["tokens"]:
            failures.append("Generated token IDs differ")
        if (a["logprobs"] is None) != (b["logprobs"] is None):
            failures.append("Log-probability availability differs")
        elif a["logprobs"] is not None:
            if len(a["logprobs"]) != len(b["logprobs"]):
                failures.append("Log-probability length differs")
            elif any(
                abs(float(x) - float(y)) > policy.logprob_abs_tolerance
                for x, y in zip(a["logprobs"], b["logprobs"], strict=True)
            ):
                failures.append("Log-probability tolerance exceeded")
        outcomes.append(
            {"id": case_id, "status": "FAIL" if failures else "PASS", "reasons": failures}
        )
    status = "FAIL" if any(x["status"] == "FAIL" for x in outcomes) else "PASS"
    return _result(status, outcomes, [], policy)


def _result(
    status: str, cases: list[dict[str, Any]], warnings: list[str], policy: Policy
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "verdict": status,
        "policy": {
            "require_token_identity": policy.require_token_identity,
            "logprob_abs_tolerance": policy.logprob_abs_tolerance,
        },
        "summary": {"checked": len(cases), "failed": sum(c["status"] == "FAIL" for c in cases)},
        "warnings": warnings,
        "cases": cases,
    }


def canonical_digest(data: Any) -> str:
    """Stable content fingerprint for provenance; not a cryptographic signature."""
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
