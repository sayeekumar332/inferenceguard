# Threat model (initial)

## Assets

Release correctness verdicts; prompt/model metadata; test-result traces.

## Trust boundaries

Run JSON is untrusted. A corrupted or adversarial collector could fake input identity or results. SHA-256 content fingerprints detect local changes but do not authenticate producers. HTML reports must never interpret user-controlled data as markup.

## Controls implemented

Strict schema checks; bounded input size and case count; finite numeric checks; HTML escaping; minimal dependencies; no network calls; no arbitrary shell commands; bounded CI privileges.

## Controls pending

Authenticated artifacts, reproducibility manifests, prompt redaction, retention policy, secure HTTPS adapters and credentials, hardened sandboxing, rate limiting, version pinning of Actions SHAs, SBOM, dependency vulnerability review, performance/load tests, sign-off from independent reviewer.

## Prohibited claims

Do not claim numerical correctness from string similarity alone; do not claim absence of bugs from a passing finite test suite; do not call an alpha report an independent certification.
