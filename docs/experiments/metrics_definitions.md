# Metrics definitions

`evaluation/metrics.py` reports by split and model tier. Accuracy is determinate correct answers divided by determinate evaluated outputs; evaluator coverage is determinate outputs divided by successful outputs and status counts expose unsupported, ambiguous, and unparseable cases. Success rate is successful executions divided by all execution records; failures are retained in that denominator. Wilson 95% intervals are reported for accuracy.

Known calculated cost is the sum of known attempt subtotals. It is not actual billing. Complete cost is reported only when every attempt cost is known. Token totals are preserved per execution. Latency includes mean and nearest-rank P50/P95 of recorded end-to-end durations; mock timings are not provider latency estimates. Reporting should include warm-up, concurrency, host, network, and provider conditions for real runs.

Paired outcomes include both correct, small-only correct, strong-only correct, both incorrect, and evaluator unknown. A missing tier is not silently paired. This phase does not calculate bootstrap confidence intervals, cost-per-1000 projections, significance tests, or oracle router savings; implement and preregister those before scientific use. Never apply inferential tests to mock outputs.

Correctness-based small-model sufficiency is 1 for successful, determinate correct real small-model outputs; 0 for determinate incorrect real outputs; null for failures and evaluator-unknown cases. The frozen criterion name is `correctness-based-sufficiency-v1`. It measures correctness only, not pedagogical sufficiency. Test-split label generation raises an error. Mock and replay records yield null. No learned router is trained in Phase 2.
