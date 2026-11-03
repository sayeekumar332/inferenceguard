# GitHub Governance Checklist

Target repository: `sayeekumar332/inferenceguard` (create separately; do not reuse unrelated repos).

## Repository settings

- Set default branch to `main`.
- Use a branch ruleset targeting `main`; enable enforcement.
- Require pull requests before merging; block direct pushes.
- Require the `test-and-lint` checks across three supported Python versions and `dependency-audit` after the first successful CI run establishes exact check names.
- Require conversation resolution; block force pushes and deletions.
- Use squash merging, automatically delete merged feature branches, and require linear history if supported.
- Enable Dependabot security updates, secret scanning, and push protection where available.
- Set GitHub Actions default workflow permissions to read-only, with write only where explicitly needed.
- Optionally require independent review when an eligible second reviewer exists. GitHub does not count self-approval as independent review.
- Test the protection by trying to push directly to `main` (expect rejection), then submit a PR and confirm required CI gates.

## Important plan and rollout limitations

Ruleset and branch-protection enforcement depends on repository visibility and GitHub plan. Personal public repositories generally support branch protections; private repositories may need an eligible paid plan. If the platform does not enforce the rules for your plan, document that limitation instead of falsely claiming protection. Exact check names are visible only after workflows have run.

## Intended branching discipline

`main` <- PR from `feature/*` or `fix/*`. Conventional commits, linked issue, passing CI, written test evidence. Tags `v0.1.0-alpha.1` onward after validation.
