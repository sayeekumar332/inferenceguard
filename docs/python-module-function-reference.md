# InferenceGuard — Python Module, Function and Parameter Reference

**Project:** [InferenceGuard](../README.md) · **Package version:** `0.1.0a0` · **Baseline:** P5.2, 170 passing pytest cases.

## Purpose and scope

This developer reference explains the **Python implementation scripts** used by InferenceGuard, their classes and functions, accepted parameters, outputs and their roles in the project. It is based on the actual repository source code at the P5.2 documentation checkpoint, not on planned functionality.

Covered: the eight Python files in `src/inferenceguard/`, the three P5.2 scripts in `demos/customer_demo/`, and the three GPU experiment scripts under `experiments/`. **Automated test functions** are documented separately in [Test Inventory](test-inventory.md); this document links to those tests instead of repeating their test-by-test catalogue. JSON datasets, Markdown reports and empty package marker modules are not executable APIs.

### How to interpret signatures

- `*` means the following arguments must be passed by keyword.
- `Path` is a filesystem path; `dict` is a Python mapping; `None` can indicate absent evidence.
- `self` is the current class instance, **not** a value the caller passes.
- A default such as `repeats=3` is used unless the caller supplies another value.
- A private leading-underscore function or class is an implementation helper, **not** a stable public interface.
- Some experiment functions lack Python annotations; their documented argument roles come from how the code uses them.
- HTTP completion output requires a compatible server; token-ID support is **not universal** among OpenAI-compatible backends.

## 1. Code organization and responsibility

| Python script | Primary responsibility |
|---|---|
| [`src/inferenceguard/__init__.py`](../src/inferenceguard/__init__.py) | Package description and `__version__` declaration |
| [`src/inferenceguard/__main__.py`](../src/inferenceguard/__main__.py) | Allows `python -m inferenceguard` to invoke the CLI |
| [`src/inferenceguard/core.py`](../src/inferenceguard/core.py) | Offline differential comparison and evidence schema validation |
| [`src/inferenceguard/http_client.py`](../src/inferenceguard/http_client.py) | HTTP completion requests, response validation, token-ID acquisition |
| [`src/inferenceguard/repeatability.py`](../src/inferenceguard/repeatability.py) | Repeated request orchestration and PASS/FAIL/INCONCLUSIVE decisions |
| [`src/inferenceguard/prefix_cache_metrics.py`](../src/inferenceguard/prefix_cache_metrics.py) | Prometheus counter parsing, retrieval and delta calculations |
| [`src/inferenceguard/cache_observation.py`](../src/inferenceguard/cache_observation.py) | Combining inference calls with independent cache metric observations |
| [`src/inferenceguard/cli.py`](../src/inferenceguard/cli.py) | Command-line entry, report writing, exit codes and HTML rendering |
| [`demos/customer_demo/healthy_demo.py`](../demos/customer_demo/healthy_demo.py) | Reusable simulated local HTTP completion server and healthy demo |
| [`demos/customer_demo/regression_demo.py`](../demos/customer_demo/regression_demo.py) | Intentional second-response token mismatch |
| [`demos/customer_demo/metrics_unavailable_demo.py`](../demos/customer_demo/metrics_unavailable_demo.py) | Inference PASS with unavailable metrics |
| [`experiments/t4_qwen3_0_6b/run_prefix_cache_experiment.py`](../experiments/t4_qwen3_0_6b/run_prefix_cache_experiment.py) | Earlier direct-vLLM cache ON/OFF GPU experiment |
| [`experiments/vllm_http_t4/run_experiment.py`](../experiments/vllm_http_t4/run_experiment.py) | Managed vLLM HTTP token-ID experiment and report validator |
| [`experiments/vllm_cache_observability_t4/run_experiment.py`](../experiments/vllm_cache_observability_t4/run_experiment.py) | Managed server, prefix-cache metric preflight, combined evidence verification and cleanup |

## 2. `core.py` — offline differential verification

**Role:** load two versioned JSON runs, validate schemas and comparison compatibility, compare generated token IDs and optionally log-probabilities.

### Classes and data fields

**`ValidationError(ValueError)`** — explicit exception for malformed or incompatible offline evidence.

**`Policy` (frozen dataclass)** — defines the comparison policy:

| Field | Type / default | Meaning |
|---|---|---|
| `logprob_abs_tolerance` | `float = 1e-3` | Maximum permitted absolute difference between paired log-probabilities; finite and nonnegative |
| `require_token_identity` | `bool = True` | Whether unequal generated token sequences cause failure |

### Functions

| Function / signature | Parameters | Return and role |
|---|---|---|
| `Policy.__post_init__(self) -> None` | `self`: constructed policy | Validates a finite, nonnegative log-probability tolerance; raises `ValidationError` on invalid values |
| `load_run(path: str \| Path) -> dict[str, Any]` | `path`: JSON evidence filepath | Loads UTF-8 JSON (maximum 8 MiB), rejects invalid non-finite JSON constants, validates schema and returns the parsed mapping |
| `_reject_constant(value: str) -> None` | `value`: forbidden JSON numeric literal (for example `NaN`) | Private JSON-parser callback that raises `ValidationError` |
| `_require_str(obj: dict, key: str, context: str) -> str` | `obj`: mapping; `key`: required field; `context`: error location | Private helper validating a nonempty string, returning its value |
| `validate_run(data: Any) -> None` | `data`: parsed offline run | Validates `schema_version: "1.0"`, metadata, unique cases, token limits and aligned finite log-probabilities |
| `compare_runs(reference: dict, candidate: dict, policy: Policy) -> dict` | `reference`: baseline; `candidate`: proposed run; `policy`: thresholds/identity policy | Returns structured `PASS`, `FAIL`, or `INCOMPATIBLE` report with per-case reasons |
| `_result(status: str, cases: list[dict], warnings: list[str], policy: Policy) -> dict` | `status`: verdict; `cases`: comparison outcomes; `warnings`: incompatibility explanation; `policy`: active settings | Private standardized offline report constructor |
| `canonical_digest(data: Any) -> str` | `data`: JSON-serializable evidence | Returns SHA-256 hex digest of canonical JSON bytes; **not** a producer signature |

**Why this matters:** model/tokenizer incompatibility cannot be mistaken for a valid regression comparison; per-case differences and evidence provenance remain inspectable.

## 3. `http_client.py` — live inference acquisition

**Role:** send a non-streaming OpenAI-compatible `/v1/completions` request and parse completion text, finish reason and optional genuine generated token IDs.

### Classes and fields

**`InferenceTransportError(Exception)`** — network, timeout or HTTP-layer request failure.

**`InferenceResponseError(Exception)`** — malformed, oversized or incompatible server response.

**`InferenceObservation` (frozen dataclass)**

| Field | Type | Meaning |
|---|---|---|
| `text` | `str` | Returned completion text |
| `finish_reason` | `str \| None` | Server-supplied completion termination reason |
| `token_ids` | `tuple[int, ...] \| None = None` | Genuine server-supplied token IDs, or absent data |

### Methods

| Method / signature | Parameters | Result and behavior |
|---|---|---|
| `HttpInferenceClient.__init__(self, endpoint: str, timeout_seconds: float = 30.0, *, include_token_ids: bool = False) -> None` | `endpoint`: HTTP(S) URL ending exactly in `/v1/completions`; `timeout_seconds`: positive finite request timeout; `include_token_ids`: enable opt-in token observation | Validates URL/timeout and stores configuration; rejects unsupported schemes, userinfo, URL query or fragment |
| `HttpInferenceClient.generate(self, model: str, prompt: str, max_tokens: int = 32) -> InferenceObservation` | `model`: nonempty served model ID; `prompt`: input text; `max_tokens`: positive generation limit | Sends JSON with `temperature: 0` and `stream: false`; includes `return_token_ids: true` when enabled. Returns validated observation or raises a typed transport/response error |

**Constraint:** `MAX_RESPONSE_BYTES = 1 MiB`; exactly one response choice with index 0 is expected. In token mode, the adapter reads `choices[0].token_ids` only when present and validates nonnegative integer values. It does not retokenize the text to invent missing IDs.

## 4. `repeatability.py` — repeated-request verdict engine

### Protocol and result

**`CompletionClient(Protocol)`** defines `generate(self, model: str, prompt: str, max_tokens: int = 32) -> InferenceObservation`. It allows the repeatability engine to accept the real HTTP client, an observed client wrapper, or a test double.

**`RepeatabilityResult` (frozen dataclass)** holds:
`verdict` (`PASS`/`FAIL`/`INCONCLUSIVE`), `requested` (requested call count), `completed` (completed call count), `observations` (tuple of results), `mismatch_indices` (zero-based indices), and `error` (failure description or `None`).

| Function / method | Parameters | Return and role |
|---|---|---|
| `CompletionClient.generate(self, model: str, prompt: str, max_tokens: int = 32)` | `model`: model ID; `prompt`: input string; `max_tokens`: generation cap | Protocol method defining the expected inference client contract |
| `run_repeatability(client: CompletionClient, *, model: str, prompt: str, repeats: int = 3, max_tokens: int = 32, comparison_mode: Literal["text", "tokens"] = "text") -> RepeatabilityResult` | `client`: inference implementation; `model`: model ID; `prompt`: repeated prompt; `repeats`: integer 2–1000; `max_tokens`: per-call generation limit; `comparison_mode`: `text` or `tokens` | Makes sequential identical requests, compares each with the first response, records mismatches, and returns a structured verdict |

**Verdict detail:** a request exception or missing required token data makes the run `INCONCLUSIVE`, not `PASS`. A completed run with unequal selected observations becomes `FAIL`.

## 5. `prefix_cache_metrics.py` — Prometheus counter reader

**Role:** obtain the vLLM query/hit counters and calculate safe per-interval changes.

| Function / method | Parameters | Return and role |
|---|---|---|
| `parse_prefix_cache_metrics(metrics_text: str) -> dict[str, int]` | `metrics_text`: Prometheus exposition text | Returns `{"queries": int, "hits": int}`; accepts supported `_total` forms, rejects missing, malformed, duplicate or nonintegral counters |
| `calculate_cache_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int \| float \| None]` | `before`: previous counter snapshot; `after`: next snapshot | Returns `queries`, `hits` deltas and `hit_rate` fraction, or `None` for a zero-query interval. Rejects resets and impossible values |
| `_RejectRedirects.redirect_request(self, req, fp, code, msg, headers, newurl)` | Standard urllib redirect-hook arguments: request, response pointer, HTTP code/message, headers and destination URL | Always returns `None` to prevent following a redirect while retrieving metrics |
| `fetch_prefix_cache_metrics(base_url: str, timeout: float = 5.0) -> dict[str, int]` | `base_url`: HTTP(S) server origin/base URL, no supplied `/metrics` suffix; `timeout`: seconds, 0 < timeout ≤ 60 | Reads `<base_url>/metrics` with a 1 MiB cap, redirect rejection and strict validation; returns parsed counters |

**Important:** parser validity is separate from workload attribution. These counters may be server-wide; unrelated inference requests can contaminate interval deltas.

## 6. `cache_observation.py` — inference + cache monitoring wrapper

**`CacheObservedClient`** implements the same `generate` interface as an ordinary completion client, but also records cache measurements independently.

| Method / signature | Parameters | Return and role |
|---|---|---|
| `CacheObservedClient.__init__(self, client: CompletionClient, metrics_reader: Callable[[], dict[str, int]]) -> None` | `client`: actual underlying inference client; `metrics_reader`: zero-argument callback returning current `queries`/`hits` counters | Captures initial metrics snapshot, initializes `deltas`, `errors`, `initial_error` and previous-snapshot state |
| `CacheObservedClient.generate(self, model: str, prompt: str, max_tokens: int = 32) -> InferenceObservation` | `model`: model ID; `prompt`: text; `max_tokens`: generation cap | Delegates inference, then collects a metrics snapshot and interval delta. On metrics failure, appends `None` and an error without invalidating successful inference |

**State semantics:** `deltas` and `errors` each gain one entry per **completed** inference request. An unavailable previous snapshot may establish a new baseline without reporting a valid delta for that particular request. Inference failures propagate to the repeatability engine.

## 7. `cli.py` — command entry, output and exit codes

| Function / signature | Parameters | Return and role |
|---|---|---|
| `main(argv: list[str] \| None = None) -> int` | `argv`: command arguments excluding program name; `None` reads the process command line | Parses `compare` or `repeat`, orchestrates verification and report writing; returns exit code 0 (`PASS`), 1 (`FAIL`), or 2 (`INCOMPATIBLE`, `INCONCLUSIVE`, or execution error) |
| `render_html(report: dict) -> str` | `report`: offline `compare_runs` result with per-case IDs, status and reasons | Returns escaped, static HTML report suitable for local review; not a live server/UI |

### CLI argument reference — `compare`

| Argument | Required / default | Purpose |
|---|---|---|
| `--reference` | Required path | Baseline offline JSON run |
| `--candidate` | Required path | Candidate offline JSON run |
| `--output` | Required path | JSON verification report destination |
| `--html` | Optional path, default `None` | Additional HTML report destination |
| `--logprob-tolerance` | Optional float, default `0.001` | Absolute tolerance for aligned log-probability comparisons |

### CLI argument reference — `repeat`

| Argument | Required / default | Purpose |
|---|---|---|
| `--endpoint` | Required | OpenAI-compatible `/v1/completions` URL |
| `--model` | Required | Model served by the endpoint |
| `--prompt` | Required | Text sent identically with each call |
| `--repeats` | Default `3` | Requested call count (validated 2–1000) |
| `--comparison-mode` | `text` (default) or `tokens` | Invariant used to compare responses |
| `--max-tokens` | Default `32` | Per-call generation limit |
| `--timeout-seconds` | Default `30.0` | Network request timeout |
| `--output` | Required path | JSON report destination |
| `--observe-prefix-cache` | Flag, off by default | Also retrieve vLLM `/metrics` and include independent `cache_observability` status |

**Cache integration:** the CLI classifies cache intervals as `AVAILABLE`, `PARTIAL` or `UNAVAILABLE`. An inference `PASS` with unavailable metrics still exits with code 0 unless a separate consumer explicitly checks cache status.

## 8. `__init__.py` and `__main__.py` — package wiring

- `src/inferenceguard/__init__.py`: exposes `__version__ = "0.1.0a0"`; no user-defined functions.
- `src/inferenceguard/__main__.py`: imports `cli.main` and exits with its returned status; no additional functions or parameters.

## 9. `demos/customer_demo/healthy_demo.py` — reusable local demo server

**Important:** the local server produces **simulated** completion text and token IDs. It sends real HTTP responses but does not execute an LLM.

**`SIMULATED_TOKEN_IDS = [101, 205, 309, 412]`** is the fixed demonstration sequence.

| Function / method | Parameters | Return and role |
|---|---|---|
| `DemoHandler.response_token_ids(self) -> list[int]` | `self`: HTTP handler instance | Returns the standard simulated token sequence; overridden by the regression demo |
| `DemoHandler.do_POST(self)` | `self`: handler carrying path, headers, request/response streams | Validates a POST to `/v1/completions`, parses JSON and replies with one simulated completion |
| `DemoHandler.log_message(self, format, *args)` | `format`: HTTP log format string; `args`: formatting arguments | Suppresses BaseHTTPRequestHandler console logs |
| `run_demo(output: Path, *, handler_class: type[BaseHTTPRequestHandler] = DemoHandler, observe_prefix_cache: bool = False) -> int` | `output`: JSON report path; `handler_class`: endpoint implementation; `observe_prefix_cache`: whether to request metrics collection | Starts a threaded loopback server on an ephemeral port, runs the actual InferenceGuard CLI for three token-mode requests, then reliably shuts down the server; returns CLI exit code |
| `main() -> int` | No Python parameters; CLI requires `--output PATH` | Parses destination and calls `run_demo` |

**Dependency:** both the regression and metrics-unavailable demos reuse `run_demo`, so changes to this shared script affect all three scenarios.

## 10. `demos/customer_demo/regression_demo.py` — injected failure case

| Function / method | Parameters | Return and role |
|---|---|---|
| `run_regression_demo(output: Path) -> int` | `output`: report path | Constructs a request-counting handler, changes the third token ID in response 2, then invokes `run_demo`; expected inference `FAIL` |
| `RegressionHandler.response_token_ids(self) -> list[int]` *(nested class within `run_regression_demo`)* | `self`: handler instance | Uses an enclosed request counter protected by a `threading.Lock`; returns `[101,205,999,412]` for request 2, otherwise standard IDs |
| `main() -> int` | No Python parameters; CLI requires `--output PATH` | Executes the deliberate regression scenario and returns CLI exit code, expected **1** |

The decoded response text stays the same even when the token IDs differ. This is a controlled negative-path demonstration, not evidence of a real model defect.

## 11. `demos/customer_demo/metrics_unavailable_demo.py` — missing metric endpoint

| Function / method | Parameters | Return and role |
|---|---|---|
| `main() -> int` | No Python parameters; CLI requires `--output PATH` | Calls shared `run_demo(output, observe_prefix_cache=True)`. Since no working `/metrics` route is supplied, inference can `PASS` while cache status is `UNAVAILABLE`; expected CLI exit code **0** |

## 12. `experiments/t4_qwen3_0_6b/run_prefix_cache_experiment.py` — earlier direct GPU experiment

This standalone script directly constructs a vLLM `LLM` object. It is distinct from the newer managed-server HTTP experiment.

| Function | Parameters | Return and role |
|---|---|---|
| `main()` | No Python parameters; CLI `--cache {on,off}` is required; CLI `--output PATH` is required | Builds a long shared-prefix prompt, runs three direct vLLM inference calls, records generated token IDs and cached prompt tokens, writes JSON evidence and prints a token-identity summary; return is implicit `None` |

**Runtime configuration in source:** `Qwen/Qwen3-0.6B`, float16, `max_model_len=1024`, `gpu_memory_utilization=0.65`, eager execution, temperature 0, 24 maximum generated tokens. The `--cache` argument controls `enable_prefix_caching`. This script requires an appropriate NVIDIA GPU, PyTorch and vLLM environment.

## 13. `experiments/vllm_http_t4/run_experiment.py` — original managed HTTP experiment

**Module constants:** `MODEL="Qwen/Qwen3-0.6B"`; `BASE_URL="http://127.0.0.1:8000"`; `OUTPUT_DIR` points into its own experiment evidence folder.

| Function | Parameters | Return and role |
|---|---|---|
| `gpu_name()` | None | Runs `nvidia-smi` and returns the detected GPU name as text; raises if unavailable |
| `wait_for_server(process)` | `process`: launched vLLM subprocess | Polls `/health` until HTTP 200, early process exit, or timeout (up to 180 attempts with delays) |
| `validate_report(report, exit_code)` | `report`: parsed JSON inference evidence; `exit_code`: observed CLI return code | Checks three completed requests, genuine token-ID lists, correct mismatch indices, verdict and exit code; raises on disagreement; prints validation summary |
| `main()` | None; uses module constants | Saves environment metadata, launches vLLM, checks readiness, runs the InferenceGuard token-mode CLI, validates the report and terminates the server on exit |

The HTTP experiment uses a real server but does not include the later separate prefix-cache metrics validation. Its test suite can revalidate saved evidence on a CPU.

## 14. `experiments/vllm_cache_observability_t4/run_experiment.py` — combined managed-server GPU experiment

This is the **later integrated workflow** supporting actual T4 results recorded in [T4_EXPERIMENT_REPORT.md](../experiments/vllm_cache_observability_t4/T4_EXPERIMENT_REPORT.md). It reuses helper functions from the earlier HTTP script and runs cache-metrics preflight checks before verification.

**Module constants:** `OUTPUT_DIR` is the evidence folder; `CACHE_TEST_PROMPT` is a long repeated prefix designed to make cache reuse measurable.

| Function | Parameters | Return and role |
|---|---|---|
| `validate_cache_report(report, exit_code)` | `report`: combined JSON evidence; `exit_code`: InferenceGuard return code | Calls token-evidence validator; requires inference `PASS`, complete `AVAILABLE` metrics, valid per-request deltas, mathematically consistent hit rates and positive hits on repeated requests; returns aggregate summary mapping |
| `build_server_command(vllm_executable: str) -> list[str]` | `vllm_executable`: executable path/name | Produces the fixed vLLM serve command using Tesla T4-compatible FP16 settings and prefix caching enabled |
| `build_inferenceguard_command(python_executable: str, output_path: str) -> list[str]` | `python_executable`: Python executable; `output_path`: destination JSON path | Produces the CLI command for three token-mode requests with `--observe-prefix-cache` and 24 output tokens |
| `check_metrics_preflight(base_url: str) -> dict[str,int]` | `base_url`: server URL, typically localhost port 8000 | Reads one valid metrics snapshot and checks counter consistency before inference begins |
| `managed_vllm_server(command: list[str], log_path: Path)` | `command`: vLLM launch argument vector; `log_path`: stdout/stderr log destination | Context manager yielding a subprocess; cleans it up on normal completion or failure, killing after shutdown timeout |
| `prepare_server_for_inference(process)` | `process`: launched vLLM process | Requires healthy server and successful metrics preflight; returns initial metric snapshot |
| `execute_inferenceguard(python_executable: str, report_path: Path) -> dict` | `python_executable`: Python interpreter; `report_path`: JSON output location | Removes stale report, invokes InferenceGuard, loads fresh JSON, independently validates combined evidence and returns summary |
| `run_controlled_experiment(vllm_executable: str, python_executable: str, output_dir: Path) -> dict` | `vllm_executable`: server command; `python_executable`: interpreter; `output_dir`: artifact directory | Coordinates server creation, preflight, inference verification and guaranteed server shutdown; returns result summary |
| `collect_environment_metadata() -> dict` | None | Records GPU name, vLLM/Python version, model, precision, sampling, prefix caching and server configuration |
| `run_experiment_entrypoint(vllm_executable: str, python_executable: str, output_dir: Path) -> dict` | `vllm_executable`: server binary; `python_executable`: Python binary; `output_dir`: artifact destination | Writes `environment.json`, runs controlled experiment and writes `failure.json` on exception |
| `main() -> int` | None; discovers `vllm` and current Python | Top-level runner; returns 0 on validated success, 1 on detected execution failure |

**Implemented workflow:** this runner starts and manages a vLLM server, checks readiness and cache metrics, executes real HTTP verification, validates evidence, and cleans up. The source module header has been updated to match this functionality.

### Real GPU evidence from this workflow

| Measurement | Recorded result |
|---|---:|
| NVIDIA GPU | Tesla T4 |
| Real inference requests completed | 3/3 |
| Generated token-ID mismatch count | 0 |
| Prefix-cache query tokens | 1,689 |
| Prefix-cache-hit tokens | 1,120 |
| Aggregate cache-hit ratio | 66.31% |

The cache ratio is a counter ratio, **not** a measured latency speedup. See the experiment report for limitations and runtime diagnostics.

## 15. End-to-end call relationships

```text
CLI entry: python -m inferenceguard
    |
    +-- cli.main(argv)
          |
          +-- compare (offline)
          |     +-- core.load_run(reference/candidate)
          |     +-- core.compare_runs(..., Policy)
          |     +-- core.canonical_digest(...)
          |     +-- cli.render_html(...) [optional]
          |
          +-- repeat (HTTP)
                +-- HttpInferenceClient(endpoint, ...)
                |     +-- generate(model, prompt, max_tokens)
                |
                +-- CacheObservedClient(client, metrics_reader) [optional]
                |     +-- fetch_prefix_cache_metrics(base_url)
                |     +-- calculate_cache_delta(before, after)
                |
                +-- run_repeatability(client, ..., comparison_mode)
                +-- JSON verification report

P5.2 local customer demos
    +-- healthy_demo.run_demo(...)
    +-- regression_demo.run_regression_demo(...)
    +-- metrics_unavailable_demo.main()
          +-- reuse real cli.main() against simulated local HTTP

Real GPU: managed T4 experiment
    +-- run_experiment_entrypoint(...)
          +-- collect_environment_metadata()
          +-- run_controlled_experiment(...)
                +-- managed_vllm_server(...)
                +-- prepare_server_for_inference(...)
                +-- execute_inferenceguard(...)
                      +-- validate_cache_report(...)
```

## 16. Example usage

### Offline verification

```bash
inferenceguard compare \
  --reference examples/reference.json \
  --candidate examples/candidate_fault.json \
  --output reports/diff.json
```

### Live repeated inference (with a compatible model server)

```bash
inferenceguard repeat \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model Qwen/Qwen3-0.6B \
  --prompt "Explain inference repeatability" \
  --repeats 3 \
  --comparison-mode tokens \
  --observe-prefix-cache \
  --output reports/combined.json
```

### CPU-only customer demonstration

```bash
python -m demos.customer_demo.healthy_demo \
  --output /tmp/inferenceguard_healthy.json
```

## 17. Interface contracts and next extensions

- `compare` produces offline `PASS`, `FAIL`, `INCOMPATIBLE` outcomes; `repeat` produces `PASS`, `FAIL`, `INCONCLUSIVE`.
- Successful inference and successful cache observability are **not identical**. Check `cache_observability.status` independently when cache metrics are required.
- The HTTP client requires a server-specific token-ID response extension in token mode; a server that omits token IDs cannot produce genuine token-level PASS evidence.
- Prometheus prefix-cache counters are not inherently request-scoped; unrelated workloads can affect observed deltas.
- The demo responses and token IDs are **simulated**; the archived T4 experiments use real vLLM inference.
- These signatures describe the current `0.1.0a0` alpha implementation. Later releases may refine interfaces as backend coverage grows.

## Related references

- [Architecture](architecture.md)
- [Automated test inventory](test-inventory.md)
- [Prefix-cache observability](prefix_cache_observability.md)
- [Threat model](threat-model.md)
- [Customer demo guide](../demos/customer_demo/README.md)
- [Managed-server Tesla T4 report](../experiments/vllm_cache_observability_t4/T4_EXPERIMENT_REPORT.md)

**Maintenance:** revisit this guide when function signatures, class fields, CLI arguments or experiment workflows change. Recheck the source before adding new claims.
