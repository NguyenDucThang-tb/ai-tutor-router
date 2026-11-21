# Metrics definitions

`evaluation/metrics.py` reports by split, model tier, and fallback policy so pure-model and operational-fallback results are never merged. Accuracy is determinate correct answers divided by determinate successful outputs; evaluator coverage is determinate outputs divided by successful outputs and status counts expose unsupported, ambiguous, and unparseable cases. Success rate is successful executions divided by all expected execution records, including failures, uncertain outcomes, and known-unstarted tasks. Missing model-tier records are counted as incomplete pairs; present but unsuccessful executions are counted separately from evaluator-unknown pairs. Wilson 95% intervals are reported for accuracy.

Known calculated cost is the sum of known attempt subtotals. It is not actual billing. Complete cost is reported only when every attempt cost is known. Token totals are preserved per execution. Latency boundaries are:

- `feature_extraction_ms` and `router_decision_ms`: isolated synchronous stage durations.
- `scheduling_queue_ms`: time from job scheduling until its semaphore slot is acquired.
- `model_only_latency_ms`: primary-model attempt durations, including primary retries and excluding fallback attempts.
- `fallback_latency_ms` and `fallback_count`: fallback attempt duration and count.
- `retry_delay_ms` and `retry_count`: measured backoff time and primary retry count.
- `gateway_elapsed_ms`: wall time awaiting the gateway operation, including retries, backoff, and fallback.
- `evaluator_latency_ms`: answer evaluation duration.
- `pipeline_end_to_end_ms`: starts before feature extraction and ends after evaluation; includes routing, scheduling/semaphore wait, gateway work, and evaluation.
- `pipeline_elapsed_to_terminal_ms`: elapsed time until a completed failure response. It is not labeled end-to-end because evaluation did not finish. In-flight/uncertain crash records leave both terminal and end-to-end pipeline durations unset.

Use model-only latency for capability comparisons and pipeline end-to-end for the full user path. Mock timings are not provider latency estimates. Report warm-up, concurrency, host, network, and provider conditions for real runs.

Paired outcomes include both correct, small-only correct, strong-only correct, both incorrect, and evaluator unknown. A missing tier is not silently paired. This phase does not calculate bootstrap confidence intervals, cost-per-1000 projections, significance tests, or oracle router savings; implement and preregister those before scientific use. Never apply inferential tests to mock outputs.

Correctness-based small-model sufficiency is 1 for successful, determinate correct real small-model outputs; 0 for determinate incorrect real outputs; null for failures and evaluator-unknown cases. The frozen criterion name is `correctness-based-sufficiency-v1`. It measures correctness only, not pedagogical sufficiency. Test-split label generation raises an error. Mock and replay records yield null. No learned router is trained in Phase 2.
