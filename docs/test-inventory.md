# InferenceGuard — Automated Test Inventory and Verification Objectives

**Project:** [InferenceGuard](../README.md)  
**Baseline:** completed P5.2 milestone on `main` (170 passing pytest cases; Ruff lint and format checks passed).  
**Scope:** all `16` test modules under `tests/` at the documented baseline, with `133` named `test_*` functions.

## Purpose and how to read this document

This is an engineering-oriented map of **each named automated test**, its core assertion or failure case, and why that behavior matters to InferenceGuard. Names and expected behavior are based on the actual test source files, not hypothetical future tests.

**Count distinction:** pytest reported **170 passed** at the P5.2 checkpoint, although there are **133 named test functions**. Parameterized functions run multiple combinations and therefore contribute more than one collected pytest case. This document lists each named function once and identifies parameterized scenarios where relevant. It does not assert that each function represents exactly one collected case.

**Verification design:** Fast CPU tests cover success and failure conditions, while separate tests check the consistency of **recorded real NVIDIA Tesla T4 evidence**. The actual GPU experiments are documented independently. Together, these approaches provide rapid regression feedback and hardware-backed integration evidence.

## Coverage map

| Area | Test module | Named tests | What this area proves |
|---|---|---:|---|
| Offline differential comparison, schema and HTML | [`test_core.py`](../tests/test_core.py) | 11 | Offline differential comparison, schema and HTML behavior is checked against specific success and failure conditions |
| Core repeated-response comparator | [`test_repeatability.py`](../tests/test_repeatability.py) | 6 | Core repeated-response comparator behavior is checked against specific success and failure conditions |
| Token-ID comparison semantics | [`test_token_repeatability.py`](../tests/test_token_repeatability.py) | 7 | Token-ID comparison semantics behavior is checked against specific success and failure conditions |
| HTTP completion adapter and response validation | [`test_http_client.py`](../tests/test_http_client.py) | 9 | HTTP completion adapter and response validation behavior is checked against specific success and failure conditions |
| HTTP-level repeatability integration | [`test_http_repeatability.py`](../tests/test_http_repeatability.py) | 4 | HTTP-level repeatability integration behavior is checked against specific success and failure conditions |
| Repeat-command interface and backward compatibility | [`test_repeat_cli.py`](../tests/test_repeat_cli.py) | 7 | Repeat-command interface and backward compatibility behavior is checked against specific success and failure conditions |
| Prometheus parser and cache delta arithmetic | [`test_prefix_cache_metrics.py`](../tests/test_prefix_cache_metrics.py) | 17 | Prometheus parser and cache delta arithmetic behavior is checked against specific success and failure conditions |
| HTTP metrics endpoint hardening | [`test_prefix_cache_http.py`](../tests/test_prefix_cache_http.py) | 11 | HTTP metrics endpoint hardening behavior is checked against specific success and failure conditions |
| Inference and cache observation interactions | [`test_cache_observed_repeatability.py`](../tests/test_cache_observed_repeatability.py) | 7 | Inference and cache observation interactions behavior is checked against specific success and failure conditions |
| Cache-observability CLI report contract | [`test_cache_observability_cli.py`](../tests/test_cache_observability_cli.py) | 6 | Cache-observability CLI report contract behavior is checked against specific success and failure conditions |
| Earlier Tesla T4 cache-ON/OFF archived evidence | [`test_gpu_experiment_evidence.py`](../tests/test_gpu_experiment_evidence.py) | 5 | Earlier Tesla T4 cache-ON/OFF archived evidence behavior is checked against specific success and failure conditions |
| T4 HTTP experiment evidence contract | [`test_vllm_http_t4_experiment.py`](../tests/test_vllm_http_t4_experiment.py) | 8 | T4 HTTP experiment evidence contract behavior is checked against specific success and failure conditions |
| Managed vLLM/T4 runner, lifecycle and evidence validation | [`test_vllm_cache_observability_t4_experiment.py`](../tests/test_vllm_cache_observability_t4_experiment.py) | 32 | Managed vLLM/T4 runner, lifecycle and evidence validation behavior is checked against specific success and failure conditions |
| P5.2 healthy customer demonstration | [`test_customer_demo_healthy.py`](../tests/test_customer_demo_healthy.py) | 1 | P5.2 healthy customer demonstration behavior is checked against specific success and failure conditions |
| P5.2 deliberate token-regression demonstration | [`test_customer_demo_regression.py`](../tests/test_customer_demo_regression.py) | 1 | P5.2 deliberate token-regression demonstration behavior is checked against specific success and failure conditions |
| P5.2 metrics-unavailable demonstration | [`test_customer_demo_metrics_unavailable.py`](../tests/test_customer_demo_metrics_unavailable.py) | 1 | P5.2 metrics-unavailable demonstration behavior is checked against specific success and failure conditions |

## Full test-by-test catalogue

### 1. Offline differential comparison, schema and HTML

**Source:** [`tests/test_core.py`](../tests/test_core.py) · **11 named tests**

| Exact test name | What it checks and accomplishes |
|---|---|
| `test_identical_pass` | Accepts identical offline reference and candidate evidence with a PASS verdict. |
| `test_corrupted_tokens_fail` | Detects a changed generated token sequence and produces FAIL. |
| `test_metadata_mismatch_incompatible` | Treats a different model revision as incompatible evidence. |
| `test_prompt_fingerprint_fail` | Detects a prompt fingerprint mismatch rather than silently aligning unrelated prompts. |
| `test_logprobs_tolerance` | Enforces the configured tolerance for token log-probability differences. |
| `test_invalid_nonfinite_policy` | Rejects a non-finite comparison tolerance. |
| `test_duplicate_case_rejected` | Rejects duplicate case IDs in offline evidence. |
| `test_different_case_sets_are_incompatible` | Reports incompatible sets of case identifiers. |
| `test_html_escapes_case_identifiers` | Escapes case identifiers in HTML to prevent injected markup or scripts. |
| `test_cli_exit_codes` | Checks compare-command exit-code and report behavior. |
| `test_nan_json_rejected` | Rejects non-standard NaN values in input JSON. |

### 2. Core repeated-response comparator

**Source:** [`tests/test_repeatability.py`](../tests/test_repeatability.py) · **6 named tests**

| Exact test name | What it checks and accomplishes |
|---|---|
| `test_identical_responses_pass` | Marks identical completion text across repeated calls as PASS. |
| `test_first_request_differs` | Detects cold-first versus subsequent response divergence. |
| `test_last_request_differs` | Detects divergence confined to the final response. |
| `test_transport_failure_is_inconclusive` | Returns INCONCLUSIVE after an inference transport failure. |
| `test_failure_not_misclassified_as_pass` | Does not issue a misleading PASS when divergence is followed by a failed request. |
| `test_invalid_repeat_count_rejected` | Rejects invalid repetition counts. |

### 3. Token-ID comparison semantics

**Source:** [`tests/test_token_repeatability.py`](../tests/test_token_repeatability.py) · **7 named tests**

| Exact test name | What it checks and accomplishes |
|---|---|
| `test_identical_token_ids_pass` | Accepts identical generated token sequences. |
| `test_different_token_ids_fail_even_when_text_matches` | Detects differing token IDs even when decoded response text is identical. |
| `test_different_token_sequence_lengths_fail` | Detects token-sequence length differences. |
| `test_missing_token_ids_are_inconclusive` | Returns INCONCLUSIVE if any required token sequence is missing. |
| `test_identical_tokens_with_different_text_pass_in_token_mode` | Uses token identity—not text identity—as the verdict basis in token mode. |
| `test_default_text_mode_preserves_existing_behavior` | Preserves text-based comparison behavior when token mode is not selected. |
| `test_invalid_comparison_mode_rejected` | Rejects unsupported comparison-mode values. |

### 4. HTTP completion adapter and response validation

**Source:** [`tests/test_http_client.py`](../tests/test_http_client.py) · **9 named tests**

| Exact test name | What it checks and accomplishes |
|---|---|
| `test_valid_completion` | Parses a valid OpenAI-compatible completion response. |
| `test_vllm_token_ids_acquired` | Acquires genuine token IDs provided by the compatible completion response. |
| `test_missing_choices_rejected` | Rejects responses missing the required choices structure. |
| `test_invalid_json_rejected` | Rejects invalid JSON returned by the inference endpoint. |
| `test_http_server_error` | Classifies an HTTP server error as a transport failure. |
| `test_invalid_endpoint_rejected` | Rejects an invalid or unsupported completion endpoint URL. |
| `test_invalid_timeout_rejected` | Rejects an invalid request timeout. |
| `test_vllm_token_request_flag_sent` | Verifies that token mode sends the return_token_ids request extension. |
| `test_vllm_malformed_token_ids_rejected` | Rejects malformed token-ID response data (parameterized inputs). |

### 5. HTTP-level repeatability integration

**Source:** [`tests/test_http_repeatability.py`](../tests/test_http_repeatability.py) · **4 named tests**

| Exact test name | What it checks and accomplishes |
|---|---|
| `test_http_identical_responses_pass` | Checks that three identical HTTP responses lead to PASS. |
| `test_http_first_request_divergence_detected` | Detects first-request versus warm-request divergence over HTTP. |
| `test_http_server_failure_is_inconclusive` | Treats an HTTP server failure as INCONCLUSIVE and records completed calls. |
| `test_repeat_cli_end_to_end` | Exercises CLI, HTTP transport, repeatability verdict, JSON report and exit code together (parameterized scenarios). |

### 6. Repeat-command interface and backward compatibility

**Source:** [`tests/test_repeat_cli.py`](../tests/test_repeat_cli.py) · **7 named tests**

| Exact test name | What it checks and accomplishes |
|---|---|
| `test_repeat_cli_verdicts` | Checks CLI verdict-to-exit-code mapping over parameterized repeated-response scenarios. |
| `test_repeat_cli_rejects_invalid_repeats` | Rejects invalid --repeats input. |
| `test_repeat_cli_token_mode_without_token_ids_is_inconclusive` | Treats absent token IDs in token mode as INCONCLUSIVE. |
| `test_repeat_cli_detects_token_mismatch` | Reports a generated token mismatch through the CLI. |
| `test_repeat_cli_default_report_preserves_text_schema` | Preserves the original text-mode JSON report shape. |
| `test_repeat_cli_real_http_token_integration` | Checks real local HTTP token-ID integration, reports and exits (parameterized scenarios). |
| `test_repeat_cli_real_http_text_backward_compatibility` | Checks that real local HTTP text mode remains backward-compatible. |

### 7. Prometheus parser and cache delta arithmetic

**Source:** [`tests/test_prefix_cache_metrics.py`](../tests/test_prefix_cache_metrics.py) · **17 named tests**

| Exact test name | What it checks and accomplishes |
|---|---|
| `test_parse_unlabelled_metrics` | Parses counters without labels. |
| `test_parse_labelled_metrics` | Parses counters carrying Prometheus label sets. |
| `test_accept_metric_names_without_total_suffix` | Accepts required metric names without the _total suffix. |
| `test_ignore_unrelated_metrics` | Ignores unrelated Prometheus metric names. |
| `test_reject_missing_metrics` | Rejects missing required cache counters (parameterized inputs). |
| `test_reject_invalid_counter_values` | Rejects invalid cache-counter representations (parameterized inputs). |
| `test_reject_duplicate_time_series` | Rejects ambiguous duplicate matching time series. |
| `test_calculate_cache_delta` | Computes query/hit deltas and a valid hit-rate fraction. |
| `test_zero_query_interval` | Represents zero-query hit rates as undefined (None), not as zero. |
| `test_zero_cache_hits` | Represents a valid interval with zero cache hits. |
| `test_full_cache_hit_rate` | Computes an interval with 100% cache hits. |
| `test_reject_counter_reset` | Detects a reset of the query counter. |
| `test_reject_hit_counter_reset` | Detects a reset of the hit counter. |
| `test_reject_hits_exceeding_queries` | Rejects intervals or snapshots with impossible hit counts. |
| `test_reject_invalid_snapshot_values` | Rejects invalid snapshot counter types or values (parameterized inputs). |
| `test_parser_accepts_real_vllm_float_counters` | Parses vLLM-style integer-valued floating-point Prometheus counters. |
| `test_parser_rejects_fractional_prometheus_counters` | Rejects fractional counter values that are not valid integer token counts. |

### 8. HTTP metrics endpoint hardening

**Source:** [`tests/test_prefix_cache_http.py`](../tests/test_prefix_cache_http.py) · **11 named tests**

| Exact test name | What it checks and accomplishes |
|---|---|
| `test_fetch_valid_prefix_cache_metrics` | Fetches and parses valid prefix-cache metrics over HTTP. |
| `test_fetch_metrics_with_trailing_slash` | Accepts a metrics base URL with a trailing slash. |
| `test_reject_missing_prefix_cache_metric` | Rejects responses missing required metrics. |
| `test_reject_malformed_prefix_cache_metric` | Rejects malformed metrics supplied by the server. |
| `test_reject_http_server_error` | Handles a metrics HTTP 503 error. |
| `test_reject_connection_error` | Handles a metrics connection error. |
| `test_reject_http_timeout` | Checks metrics timeout failure handling. |
| `test_reject_redirect_response` | Rejects an HTTP redirect rather than following it. |
| `test_reject_oversized_metrics_response` | Rejects metrics responses exceeding the configured size limit. |
| `test_reject_non_http_url` | Rejects unsupported URL schemes. |
| `test_reject_redirect_with_valid_destination` | Rejects redirects even when the target would otherwise return valid metrics. |

### 9. Inference and cache observation interactions

**Source:** [`tests/test_cache_observed_repeatability.py`](../tests/test_cache_observed_repeatability.py) · **7 named tests**

| Exact test name | What it checks and accomplishes |
|---|---|
| `test_observe_cache_activity_for_three_requests` | Collects interval cache deltas for three repeated inference requests. |
| `test_cache_activity_does_not_hide_token_divergence` | Ensures token divergence is still detected when cache observations are present. |
| `test_zero_cache_queries_are_not_reported_as_zero_rate` | Preserves an undefined hit rate for a zero-query interval. |
| `test_metrics_unavailable_before_first_request` | Records metrics unavailability before the first request without invalidating inference. |
| `test_metrics_failure_after_inference` | Records a post-inference metrics collection error separately. |
| `test_counter_reset_does_not_invalidate_tokens` | Handles a cache counter reset without changing the inference token verdict. |
| `test_inference_failure_remains_inconclusive` | Preserves an INCONCLUSIVE result if the underlying inference request fails. |

### 10. Cache-observability CLI report contract

**Source:** [`tests/test_cache_observability_cli.py`](../tests/test_cache_observability_cli.py) · **6 named tests**

| Exact test name | What it checks and accomplishes |
|---|---|
| `test_default_cli_report_remains_unchanged` | Ensures the default CLI path never requests metrics and retains its prior report shape. |
| `test_cli_records_cache_observations` | Includes cache observations when explicitly requested. |
| `test_cli_preserves_verdict_when_metrics_fail` | Preserves the inference verdict despite metrics acquisition failures. |
| `test_cli_detects_token_divergence_with_cache_metrics` | Detects token divergence while metrics are also being collected. |
| `test_cli_reports_partial_cache_observability` | Reports PARTIAL when only some cache intervals have data. |
| `test_cli_incomplete_inference_remains_inconclusive` | Preserves INCONCLUSIVE for incomplete inference even with observation enabled. |

### 11. Earlier Tesla T4 cache-ON/OFF archived evidence

**Source:** [`tests/test_gpu_experiment_evidence.py`](../tests/test_gpu_experiment_evidence.py) · **5 named tests**

| Exact test name | What it checks and accomplishes |
|---|---|
| `test_experiment_configuration` | Validates the recorded configurations for cache-disabled and cache-enabled trials. |
| `test_cache_disabled_has_no_hits` | Confirms no cache hits were recorded in cache-disabled observations. |
| `test_cache_enabled_reuses_prefix` | Confirms actual prompt prefix reuse in the cache-enabled warm observations. |
| `test_generated_tokens_are_identical` | Checks that recorded generated token IDs agree across the experiment requests. |
| `test_comparison_matches_original_evidence` | Checks that the archived comparison report agrees with the original observations. |

### 12. T4 HTTP experiment evidence contract

**Source:** [`tests/test_vllm_http_t4_experiment.py`](../tests/test_vllm_http_t4_experiment.py) · **8 named tests**

| Exact test name | What it checks and accomplishes |
|---|---|
| `test_identical_token_sequences_accepted` | Accepts a consistent three-request token report with matching sequences. |
| `test_detected_token_divergence_accepted` | Accepts a correctly reported detected token divergence. |
| `test_invalid_token_evidence_rejected` | Rejects invalid or missing token evidence (parameterized invalid inputs). |
| `test_incorrect_verdict_rejected` | Rejects a PASS verdict contradicted by token differences. |
| `test_incorrect_exit_code_rejected` | Rejects an exit code inconsistent with the report. |
| `test_incorrect_mismatch_indices_rejected` | Rejects incorrect mismatch indices. |
| `test_incomplete_requests_rejected` | Rejects a report with fewer completed requests than required. |
| `test_preserved_real_t4_http_evidence` | Validates the preserved real-vLLM HTTP observation artifact without needing a GPU. |

### 13. Managed vLLM/T4 runner, lifecycle and evidence validation

**Source:** [`tests/test_vllm_cache_observability_t4_experiment.py`](../tests/test_vllm_cache_observability_t4_experiment.py) · **32 named tests**

| Exact test name | What it checks and accomplishes |
|---|---|
| `test_valid_combined_evidence_accepted` | Accepts a structurally valid combined inference and cache report. |
| `test_missing_cache_observability_rejected` | Rejects a report lacking cache-observability evidence. |
| `test_unavailable_cache_metrics_rejected` | Rejects unavailable cache metrics when the experiment requires valid metrics. |
| `test_token_divergence_rejected` | Rejects token divergence in a supposed successful baseline. |
| `test_no_measured_cache_hits_rejected` | Rejects a report without demonstrated cache hits. |
| `test_incorrect_cache_hit_rate_rejected` | Rejects a mathematically inconsistent cache-hit rate. |
| `test_invalid_cache_counter_rejected` | Rejects invalid cache-counter values (parameterized inputs). |
| `test_first_request_cache_hits_do_not_prove_reuse` | Ensures first-request hits alone do not prove repeated-prefix reuse. |
| `test_vllm_server_command_preserves_t4_configuration` | Checks the generated vLLM startup command retains required Tesla T4 configuration. |
| `test_inferenceguard_command_enables_cache_observability` | Checks the generated InferenceGuard command enables cache observation. |
| `test_inferenceguard_command_preserves_output_destination` | Checks the generated command writes to the intended report path. |
| `test_metrics_preflight_accepts_valid_counters` | Accepts legitimate metrics during preflight checks. |
| `test_metrics_preflight_rejects_invalid_counters` | Rejects malformed or incompatible counters during metrics preflight. |
| `test_metrics_preflight_rejects_unavailable_endpoint` | Rejects an unavailable metrics endpoint before starting the experiment. |
| `test_server_lifecycle_starts_and_stops` | Ensures a managed server is started and shut down. |
| `test_server_lifecycle_cleans_up_after_failure` | Ensures server resources are cleaned up after an exception. |
| `test_server_lifecycle_kills_after_shutdown_timeout` | Tests forced termination when graceful server shutdown times out. |
| `test_server_preflight_accepts_healthy_server_and_metrics` | Accepts a healthy server with valid metrics before running inference. |
| `test_server_preflight_rejects_early_server_exit` | Rejects a server that exits prematurely. |
| `test_server_preflight_rejects_unavailable_metrics` | Rejects an unavailable metrics endpoint at server readiness checks. |
| `test_execute_inferenceguard_accepts_valid_evidence` | Accepts successful InferenceGuard execution and a validated report. |
| `test_execute_inferenceguard_rejects_missing_report` | Rejects a missing inference verification report. |
| `test_execute_inferenceguard_rejects_invalid_evidence` | Rejects an inference report that violates evidence requirements. |
| `test_controlled_experiment_runs_all_stages` | Runs all controlled stages in the correct orchestrated workflow. |
| `test_controlled_experiment_stops_on_preflight_failure` | Stops the workflow when preflight validation fails. |
| `test_controlled_experiment_stops_on_inference_failure` | Stops the workflow on inference failure and cleans up. |
| `test_environment_metadata_records_gpu_and_vllm_version` | Records GPU and vLLM version in the experiment metadata. |
| `test_environment_metadata_records_experiment_configuration` | Records key experiment and serving configuration settings. |
| `test_experiment_entrypoint_saves_metadata_and_runs` | Checks entrypoint persistence of environment evidence and execution. |
| `test_experiment_entrypoint_preserves_failure_diagnostics` | Preserves diagnostics when experiment execution fails. |
| `test_experiment_main_invokes_entrypoint` | Checks that the command-line main function invokes the experiment entrypoint. |
| `test_experiment_prompt_has_substantial_shared_prefix` | Checks the prompt supplies sufficient shared-prefix material for the cache experiment. |

### 14. P5.2 healthy customer demonstration

**Source:** [`tests/test_customer_demo_healthy.py`](../tests/test_customer_demo_healthy.py) · **1 named test**

| Exact test name | What it checks and accomplishes |
|---|---|
| `test_healthy_customer_demo` | Runs the complete local HTTP demo via a subprocess and verifies a PASS report, matching simulated token IDs and successful exit. |

### 15. P5.2 deliberate token-regression demonstration

**Source:** [`tests/test_customer_demo_regression.py`](../tests/test_customer_demo_regression.py) · **1 named test**

| Exact test name | What it checks and accomplishes |
|---|---|
| `test_regression_customer_demo` | Runs the local regression demo and verifies the intended FAIL report, second-response token mismatch, unchanged text and exit 1. |

### 16. P5.2 metrics-unavailable demonstration

**Source:** [`tests/test_customer_demo_metrics_unavailable.py`](../tests/test_customer_demo_metrics_unavailable.py) · **1 named test**

| Exact test name | What it checks and accomplishes |
|---|---|
| `test_metrics_unavailable_customer_demo` | Runs the local no-metrics demo and verifies inference PASS, UNAVAILABLE cache status, null deltas and recorded errors. |

## Verification commands

From the repository root after installing the developer dependencies:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
git diff --check
```

To examine the collected parameterized cases and verify the case count in your checkout:

```bash
python -m pytest --collect-only -q
```

To run a specific test module:

```bash
python -m pytest tests/test_prefix_cache_metrics.py -v
```

To run an individual named test:

```bash
python -m pytest tests/test_core.py::test_corrupted_tokens_fail -v
```

## What the test suite establishes

- **Correctness of comparisons:** identical evidence passes, actual differences fail, incompatible or incomplete evidence is not falsely labeled successful.
- **Token-observation integrity:** differences in generated token IDs are detected even when decoded text is identical; absent or malformed token IDs are not silently reconstructed.
- **Cache observability integrity:** missing, ambiguous, reset or malformed counters cannot be mistaken for valid zero-hit measurements; cache status and inference verdict remain separate.
- **Client and pipeline resilience:** invalid HTTP, server failures, invalid input, timeouts, redirects, and oversized metrics receive explicit handling.
- **Reproducible evidence:** archived GPU reports and the managed experiment runner have independently testable consistency, lifecycle, and failure-handling contracts.
- **Customer demonstration reproducibility:** the three local simulated scenarios exercise documented PASS, FAIL and UNAVAILABLE outcomes through actual HTTP interactions.

## Coverage boundaries and future validation

The automated suite checks the **implemented verification behavior** on a CPU and validates archived GPU evidence, while actual NVIDIA Tesla T4 inference was exercised separately. Next extensions are cross-GPU and cross-backend studies, concurrent or distributed workloads, and stronger evidence provenance.

The checks do not establish universal determinism, measured latency or cost improvements, or security certification. Reproducing the original H100-based [vLLM issue #40896](https://github.com/vllm-project/vllm/issues/40896) would require an appropriately matched separate experiment.

For actual GPU evidence, see the [managed-server Tesla T4 experiment report](../experiments/vllm_cache_observability_t4/T4_EXPERIMENT_REPORT.md) and the [earlier T4 experiment](../experiments/t4_qwen3_0_6b/README.md). For customer-facing scenarios, see the [P5.2 demonstration guide](../demos/customer_demo/README.md).

## Maintenance

Update this inventory when new test functions or parameterized cases are added, renamed, or removed. Record new actual collection counts from `pytest --collect-only` or completed pytest runs. Treat the **170 passing cases** as the P5.2 historical checkpoint, not a perpetual count.
