# InferenceGuard

**Status: v0.1 foundation / alpha. NOT production-certified.**

InferenceGuard is an offline differential verification CLI for AI inference observations. It compares reproducible reference/candidate results and generates machine-readable and human-readable evidence. Initial scope is deliberately small: this foundation **does not yet call vLLM/llama.cpp or prove model semantic correctness**.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
inferenceguard compare --reference examples/reference.json --candidate examples/candidate_fault.json --output reports/result.json --html reports/result.html
# expected exit code: 1 (deliberately injected token mismatch)
```

Compare with the same reference file as candidate for exit code 0.

## Outcomes

* `PASS` (exit 0): all implemented invariants hold for the supplied observations.
* `FAIL` (exit 1): one or more measured invariants were violated.
* `INCOMPATIBLE` / malformed input (exit 2): cannot issue a defensible pass/fail result.

## Guarantees and exclusions

The checker validates schema, alignment of case IDs and prompt fingerprints, model/tokenizer revision, sampling metadata, token identity, and optional per-token log-probabilities within configurable absolute tolerance. It does not currently verify that model results are semantically correct. For stochastic or nondeterministic inference modes, token identity can yield false alarms and the current runner should not be used as a release gate without additional statistical design. The test data must be collected from the same test protocol; neither case IDs nor prompt hashes are cryptographic attestations.

## Documentation

- [Architecture and specification](docs/architecture.md)
- [GitHub repository protections](docs/github-governance.md)
- [Threat model and limitations](docs/threat-model.md)

## Roadmap

1. Input validation and offline comparison (initial foundation).
2. Real HTTP inference adapters and deterministic prompts.
3. Cache-on/cache-off and concurrency perturbation with correct run metadata.
4. Real historical regression reproduction and evaluated false-positive rates.
5. Signed/provenanced evidence and enterprise integrations after market validation.

MIT license. This is not an attestation or safety certification product.
