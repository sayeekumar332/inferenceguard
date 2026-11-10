# InferenceGuard — GitHub Governance and Development Evidence

**Repository:** [sayeekumar332/inferenceguard](https://github.com/sayeekumar332/inferenceguard) · **Current package:** `0.1.0a0`

## 1. Objectives

Maintain a traceable engineering history, verified quality checks, protected integration branches, and reviewable experimental evidence. Separate what is **configured** from what is **confirmed by an actual GitHub check or PR**.

## 2. Branch and pull-request policy

- `main` is the integrated development baseline; submit changes through `feat/*`, `fix/*`, `docs/*`, and `test/*` branches.
- Use PR descriptions to identify the problem, delivered changes, evidence, and boundaries.
- Require appropriate automated checks and review before merging, according to the repository's effective branch rules.
- Avoid direct or force pushes to protected `main`.
- Prefer squash merging when appropriate and retain reviewable PR discussion and original experiment evidence. Do not automatically delete merged topic branches solely to improve appearance.
- GitHub App and automation permissions should follow least privilege; connected-app write access does not imply permission to bypass branch protection.

## 3. Quality gate

From the repository root:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m build
git diff --check
```

The **merged P5.2 milestone** was verified locally with **170 passing tests**, Ruff checks, and a protected-branch PR workflow. Test counts are snapshots: always use the relevant branch's latest CI results for new changes, rather than assuming the same count indefinitely.

Confirm the actual workflow names, required check contexts, review requirements, secret scanning and dependency-audit settings in GitHub before making claims about their enforcement. Some controls depend on GitHub plan and repository settings; an owner's self-review is not an independent review.

## 4. PR evidence standard

Each PR should specify:

1. Problem statement and affected behavior.
2. Concrete files and functionality changed.
3. Tests/CI outcomes and links to reproduced evidence.
4. Compatibility, limitations and remaining work.

A documentation-only PR must not claim to implement features already delivered. PR #17 updates the top-level README with the integrated GPU and P5.2 evidence; the `docs/` refresh is a separate documentation change.

## 5. Experiment evidence retention

- Preserve original scripts, environment/configuration metadata, observations and report files for each distinct experiment.
- Do not edit archived evidence to make a newer result look like the original.
- Record model, vLLM version, GPU, precision, sampling, token-capture method, and relevant cache settings.
- Keep CPU-runnable evidence consistency tests in CI where practical, without requiring GPU access.
- Review prompts, tokens, paths, logs and environment files for secrets or sensitive data before publication.
- Distinguish the older T4 cache-ON/OFF study from the newer vLLM 0.33.0 managed-server study; neither reproduces the referenced H100 issue.

Evidence for the newer study is in [the Tesla T4 report](../experiments/vllm_cache_observability_t4/T4_EXPERIMENT_REPORT.md). Customer-facing simulated cases are documented in [the demonstration guide](../demos/customer_demo/README.md).

## 6. Release and external claims

Use the package version in `pyproject.toml`; it is currently an alpha release, `0.1.0a0`.

- Report **170 passing tests** specifically as the P5.2 recorded milestone, not a permanent invariant.
- Describe the real T4 experiment as three matching generated token-ID sequences and 1,120 recorded prefix-cache-hit tokens under its tested setup.
- Describe local customer demos as **simulated HTTP responses**, not real LLM runs.
- Do not equate cache-hit ratios with measured cost or latency reductions.
- No test suite or small hardware experiment independently certifies model correctness, end-to-end security, or cross-platform determinism.
