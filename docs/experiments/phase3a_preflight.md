# Phase 3A real-provider preflight and spend gate

Run offline preflight with:

```bash
python -m experiments.preflight --config configs/phase3a_pilot.example.yaml --dry-run
```

To preflight the exact stratified pilot, add `--collection-plan collection_plan.json`. The requested split defaults to `validation`; for a train plan, pass both `--split train` and `--collection-plan collection_plan.json`. Preflight rejects a plan whose split differs from `--split` and reports the selected split, sample IDs, their SHA-256, and plan hash. Official-test plans remain rejected. Without a plan, the report explicitly identifies the backward-compatible `legacy_sorted_prefix_v1` selection for the requested train or validation split.

It validates processed data checksums and pinned source provenance, configured model IDs against the dated price catalog, prompt/evaluator references, approval flags, request/attempt caps, presence of `TUTOR_LITELLM_API_KEY` by variable name, and optional checkpoint existence. It writes `preflight_report.json`; it never contacts a provider. Secret values are not included. `PASS` means validation passed, not that provider access is verified; `BLOCKED` means the configuration must not be invoked.

## Current hard blocker

Public list prices and word-count projections cannot establish a hard dollar limit. There is no exact verified tokenizer for the selected provider IDs and no provider-enforced spend cap. A provider can bill an uncertain request even when no response is persisted; retries/fallback multiply exposure. Therefore preflight reports `hard_spend_bound: BLOCKED`, sets `hard_upper_bound` to null, and the normal real runner refuses before invoking LiteLLM. Decimal arithmetic is used for scenario projections; these are not actual billing and cannot authorize a paid call. Configured run and per-request budgets are required but do not make the gate pass.

To enable real calls, implement and verify model-specific input token counting, maximum billable output accounting (including reasoning tokens), concurrency-safe SQLite reservation of worst-case attempts across retries/fallback/recovery, and provider billing limits or a demonstrated hard bound. Until then there is no approved real pilot command. The example config is a sample design only, with real execution disabled.

Preflight reports missing keys only as environment variable names; it does not test whether credentials are valid. No paid API calls are made by preflight or tests.
