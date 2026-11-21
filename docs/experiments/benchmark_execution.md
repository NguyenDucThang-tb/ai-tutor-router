# Benchmark execution

## Offline and replay

```bash
uv run python -m experiments.benchmark run --config configs/benchmark_mock.yaml \
  --dataset-dir data/processed/gsm8k --manifest data/manifests/gsm8k.json --split validation
uv run python -m experiments.benchmark report --run-dir artifacts/runs/<run-id>
uv run python -m experiments.benchmark replay --source-records artifacts/runs/<run-id>/records.jsonl \
  --dataset-dir data/processed/gsm8k --manifest data/manifests/gsm8k.json \
  --run-id replay-example --output-dir artifacts/runs/replay-example
```

Mock mode exercises paired scheduling and persistence only. It cannot be used as accuracy evidence or training labels. Replay requires matching sample IDs and source hashes, reruns the evaluator, and is marked non-scientific.

The paired runner uses identical questions and a frozen `math-direct-v1` prompt, temperature 0, configured token cap, deterministic sample order, and a stable run ID. Run manifests record source URL/organization/license, raw source checksums, dataset revision/content hash, assigned split, prompt hash, evaluator version, model/provider identifiers, provider region if known, decoding, seed, concurrency, timeout, retry/fallback, pricing metadata, software versions, timestamp, and Git SHA. Secrets are excluded.

SQLite uses per-task leases, unique claim tokens, and monotonically increasing fencing tokens. `pending` means known not started; an expired `claimed` lease can return to pending. `in_flight` means invocation may have crossed the provider boundary. Expired or cancelled in-flight work becomes `uncertain`, retains a possible-cost attempt, and is not reissued by default. Real-provider recovery requires both `allow_uncertain_reissue: true` and `uncertain_reissue_approval_id`; the recovery record keeps the earlier unknown attempt and may therefore have incomplete cost. Setting the flag does not provide exactly-once execution. No provider idempotency guarantee is assumed or used. A crash after provider processing but before SQLite save can still leave an outcome unknown. Live leases are heartbeated; concurrent runners skip live claims. Stale writers fail fencing checks. Cancellation before invocation releases a claim as pending; cancellation after invocation marks it uncertain. Completed executions are not repeated unless `rerun_completed` is explicitly enabled. The runner caps samples to honor `max_requests` against configured retries and one fallback attempt.

Paid gateway retries default to zero. Real mode with `max_retries > 0`, `retry_failed`, or `rerun_completed` also requires `allow_paid_retries: true` and `paid_retry_approval_id`. An observed timeout or other nondefinitive provider failure with attempt provenance is recorded as uncertain, not as a definite failure. Provider-specific idempotency is not enabled because no guarantee has been verified for this adapter/provider combination.

The runner writes explicit `not_started` rows for expected tasks that remain pending or held by another live runner. This preserves paired denominators; uncertain records are separately identified.

## Real-provider gate

Real mode requires `mode: real`, `provider: litellm`, explicit `allow_real_provider: true`, an approval reference, two distinct model IDs, configured provider credentials in environment, bounded sample/request/concurrency settings, and all existing Phase 1 safeguards. The example config intentionally has the opt-in disabled. Do not put secrets into YAML or manifests. Temperature zero is not a determinism guarantee. If Phase 1's per-request budget guard is active, a missing reliable estimate fails closed; no prices are fabricated. A pricing source and effective date are provenance fields, not a bundled price database. No real calls were made for this phase.

Run artifacts include JSONL records, SQLite checkpoint state, manifest, and summary. A pure-model protocol has fallback disabled; operational fallback runs are grouped and analyzed separately. `scientific_protocol_eligible` describes protocol configuration and the registered official GSM8K source/revision/license plus checksummed raw files and sample provenance. Each record's `scientific_eligible` additionally requires a successful LiteLLM attempt, configured primary/fallback model identity, frozen prompt and temperature, matching request/output provenance, and completed evaluation. This gate does not assert correctness or high evaluator coverage; those remain separate metrics. Other datasets are not scientifically eligible until explicitly registered and verified.
