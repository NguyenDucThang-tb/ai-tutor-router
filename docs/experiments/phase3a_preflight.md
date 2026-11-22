# Phase 3A real-provider preflight and spend gate

Run offline preflight with:

```bash
python -m experiments.preflight --config configs/phase3a_pilot.example.yaml --dry-run
```

To preflight the exact stratified pilot, add `--collection-plan collection_plan.json`. The requested split defaults to `validation`; for a train plan, pass both `--split train` and `--collection-plan collection_plan.json`. Preflight rejects a plan whose split differs from `--split` and reports the selected split, sample IDs, their SHA-256, and plan hash. Official-test plans remain rejected. Without a plan, the report explicitly identifies the backward-compatible `legacy_sorted_prefix_v1` selection for the requested train or validation split.

It validates processed data checksums and pinned source provenance, configured model IDs and capability records, prompt/evaluator references, approval flags, request/attempt caps, presence of `TUTOR_LITELLM_API_KEY` by variable name, and optional checkpoint existence. It writes `preflight_report.json`; it never contacts a provider. Secret values are not included. The report includes the centralized paid-gate result and its reason codes, plus existing local budget exposure and remaining local budget when a matching SQLite checkpoint exists. That ledger excludes shared-account and external process use. `PASS` on an individual check does not mean overall paid readiness; `BLOCKED` means the configuration must not be invoked.

## Current hard blocker

The candidate-pair recommendation is now Gemini 3.5 Flash-Lite / 3.8 Flash, while the sample YAML has been updated to those IDs. Gemini publishes an input count facility and model input/output limits, but that does not establish a complete bound for actual billable request framing, special categories, current account tier, or provider charges. Its project spend cap is experimental and may overrun for roughly ten minutes while billing data is processed. The provider's tier cap is shared across projects attached to a billing account. OpenAI similarly documents that enforced hard spend limits may overshoot slightly, and Anthropic account limits are not verified for this project. Therefore no candidate has a certified per-run hard bound; preflight must report the paid gate `BLOCKED`, with `hard_upper_bound` null.

The SQLite checkpoint now contains an integer micro-USD exposure ledger. It atomically enforces local per-request/run ceilings against the immutable manifest, ties each reservation to a live fencing claim, retains uncertain exposure after invocation, and does not release exposure based only on usage-derived cost. This local ledger is not provider billing and is not a bill maximum. The runner evaluates the centralized gate before processing the dataset; injected gateways cannot authorize paid execution.

The example config is a proposal, not approval, and there is no approved real pilot command. To reconsider the gate, verify account/provider access and account limits, validate exact LiteLLM request-to-provider identity, implement provider-specific exact or bounded accounting for every billable category, and establish an external cap whose documented scope and enforcement meet the maximum pilot exposure. Provider controls with known delayed enforcement are not sufficient to claim the run budget as a hard limit.

Preflight reports missing keys only as environment variable names; it does not test whether credentials are valid. No paid API calls are made by preflight or tests.
