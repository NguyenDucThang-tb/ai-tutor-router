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

The paired runner uses identical questions and a frozen `math-direct-v1` prompt, temperature 0, configured token cap, deterministic sample order, and a stable run ID. Run manifests record source hash/revision, assigned split, prompt hash, evaluator version in records, model/provider identifiers, provider region if known, decoding, seed, concurrency, timeout, retry/fallback, pricing metadata, software versions, timestamp, and Git SHA. Secrets are excluded.

SQLite transactions claim each `(run, sample, tier)` once. Existing completed executions are not repeated unless `rerun_completed` is explicitly enabled. Failed and interrupted rows require explicit retry flags. The runner caps samples to honor `max_requests` against configured retries and one fallback attempt. Provider timeout, concurrency, and retry remain bounded.

## Real-provider gate

Real mode requires `mode: real`, `provider: litellm`, explicit `allow_real_provider: true`, an approval reference, two distinct model IDs, configured provider credentials in environment, bounded sample/request/concurrency settings, and all existing Phase 1 safeguards. The example config intentionally has the opt-in disabled. Do not put secrets into YAML or manifests. Temperature zero is not a determinism guarantee. If Phase 1's per-request budget guard is active, a missing reliable estimate fails closed; no prices are fabricated. A pricing source and effective date are provenance fields, not a bundled price database. No real calls were made for this phase.

Run artifacts include JSONL records, SQLite checkpoint state, manifest, and summary. A pure-model protocol has fallback disabled; operational fallback runs must be treated and analyzed as a separate protocol.
