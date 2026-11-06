# InferenceGuard — GitHub Governance and Development Evidence

**Repository:** [sayeekumar332/inferenceguard](https://github.com/sayeekumar332/inferenceguard)

## 1. Governance objectives

Maintain a traceable engineering history, enforce automated quality checks where the GitHub repository configuration permits, and preserve the experimental evidence and pull-request discussion supporting each milestone.

## 2. Branching and integration policy

- `main` is the integrated, tested development baseline.
- Use descriptive topic branches such as `feat/*`, `fix/*`, `docs/*`, and `test/*`.
- Submit changes through pull requests with a clear description, test evidence, scope, and limitations specific to the change.
- Prefer squash merging after required checks pass; preserve the original topic branch and its commits as supplementary development history.
- **Do not automatically delete merged feature branches.** Original branch commits and pull requests are retained as reviewable process evidence; experimental JSON and scripts committed to `main` are the principal result evidence.
- Retain local branches unless intentionally removed; `git fetch --prune` only prunes remote-tracking references for branches already deleted on the remote.
- Protect `main` from direct pushes, force pushes, and deletion where repository rules and access rights permit.

## 3. Repository controls and verification checklist

The following are desired or previously configured practices; confirm the live repository settings and effective enforcement before describing a particular control as active.

- Verify default branch is `main` and the branch ruleset applies to it.
- Require pull requests and relevant CI status checks before merging.
- Check status names against actual workflow runs (for example, the Python-version test/lint matrix and dependency audit); do not assume names before seeing results.
- Require conversation resolution where supported.
- Use least-privilege GitHub Actions permissions, with read-only defaults unless a workflow needs scoped write access.
- Enable Dependabot updates, secret scanning, and push protection where available; verify actual availability and settings.
- Require an independent reviewer when available; self-approval is not independent review.
- Verify the effective rules using repository settings and actual PR behavior rather than relying on a checklist alone.

GitHub plan, repository visibility, and permissions can affect which branch protections or advanced security features are enforced. Record any exceptions accurately.

## 4. CI and local quality gates

Before proposing a PR, run:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
git diff --check
```

Where relevant, run package-build and dependency-audit workflows and check the exact PR CI outcomes in GitHub. The recorded baseline after token-ID comparison PR #7 was **55 passing local tests**; counts will grow as new tests are added.

## 5. PR documentation contract

A PR description should state:

1. **Problem:** the user-facing or engineering problem it addresses.
2. **Delivered:** behavior or documentation actually changed by this PR.
3. **Evidence:** tests, recorded observations, and links to supporting artifacts.
4. **Integration boundary:** the next specific requirement if broader hardware, protocol, or backend coverage is sought.

A documentation-only PR must not claim it added functionality already implemented in earlier PRs. Link motivating issues, including [vLLM issue #40896](https://github.com/vllm-project/vllm/issues/40896), while distinguishing detection/observability work from a fix to upstream vLLM internals.

## 6. Experimental evidence retention

- Commit reusable experiment scripts, clearly labeled original JSON reports, comparison logic, and interpretive documentation.
- Record hardware, model, inference engine, available version/revision information, sampling settings, and measurement method.
- Prefer small CPU-only verification tests of captured evidence so CI does not require GPUs.
- Keep original experimental observations unchanged; create new files for subsequent runs.
- Remove credentials, private prompts, sensitive metadata, and other secrets before public publication.
- Describe Tesla T4 results as T4 results; they do not substitute for a matched NVIDIA H100 reproduction.

## 7. Release and product communications

Use the actual package version from `pyproject.toml` (currently `0.1.0a0`). Keep product language capability-led and evidence-based: distinguish functionality demonstrated by tests, real hardware experiments, and future integrations. Passing CI and finite experiments support specific engineering claims; they do not serve as independent certification of model correctness.
