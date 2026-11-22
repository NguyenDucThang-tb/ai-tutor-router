# Phase 3A real-provider preflight and spend gate

Run offline preflight with the frozen collection plan:

```bash
uv run python -m experiments.preflight --config configs/phase3a_pilot.example.yaml --collection-plan configs/phase3a_pilot.collection-plan.json --dry-run
```

To preflight the exact stratified pilot, add `--collection-plan collection_plan.json`. The requested split defaults to `validation`; for a train plan, pass both `--split train` and `--collection-plan collection_plan.json`. Preflight rejects a plan whose split differs from `--split` and reports the selected split, sample IDs, their SHA-256, and plan hash. Official-test plans remain rejected. Without a plan, the report explicitly identifies the backward-compatible `legacy_sorted_prefix_v1` selection for the requested train or validation split.

It validates processed data checksums and pinned source provenance, configured model IDs and capability records, prompt/evaluator references, the signed frozen-config approval artifact, request/attempt caps, presence of `TUTOR_LITELLM_API_KEY` by variable name, and optional checkpoint existence. It writes `preflight_report.json`; it never contacts a provider. Secret values are not included. The report includes the centralized paid-gate result and its reason codes, plus existing local budget exposure and remaining local budget when a matching SQLite checkpoint exists. That ledger excludes shared-account and external process use. `PASS` on an individual check does not mean overall paid readiness; `BLOCKED` means the configuration must not be invoked.

The offline provider verifier is separate and intentionally avoids account endpoints and LiteLLM imports:

```bash
uv run python -m experiments.provider_verify --config configs/phase3a_pilot.example.yaml --offline
```

It reports public model documentation separately from account access, model access, billing scope, and paid authorization. Omitting `--offline` returns an unsupported-mode error; no account verification network mode is implemented. The approval artifact at `configs/phase3a_pilot.authorization.json` is `NOT_APPROVED` and unsigned. Its digest changes whenever the approved model, data/plan/sample identity, prompt, generation policy, request envelope, or budget changes. An explicit trusted signing key is required to validate approval metadata; it is neither present nor included in this branch.

## Current hard blocker

The offline candidate recommendation remains Gemini 3.5 Flash-Lite / 3.8 Flash. The locked LiteLLM package contains matching Gemini chat-completion IDs and model limits, but offline mapping does not establish account access or actual billing. Gemini's input count facility does not establish an enforceable bill bound. Its project spend cap is experimental and may overrun during roughly ten minutes of billing-data delay. OpenAI likewise documents delayed hard-limit enforcement, and Anthropic account limits are unverified for this project. Therefore no candidate has a certified per-run hard bound; preflight must report the paid gate `BLOCKED`, with `hard_upper_bound` null.

The SQLite checkpoint contains an integer micro-USD exposure ledger. It atomically enforces local per-request/run ceilings against the immutable manifest, ties each reservation to a live fencing claim, retains uncertain exposure after invocation, and does not release exposure based only on usage-derived cost. This local ledger is not provider billing and is not a bill maximum. The runner validates plan/data and the signed approval digest before task creation, then evaluates the centralized gate; injected gateways cannot authorize paid execution. The separate ordinary tutoring app LiteLLM path does not use this benchmark ledger.

The example config is a proposal, not approval, and there is no approved real pilot command. To reconsider the gate, verify account/provider access and account limits, validate exact LiteLLM request-to-provider identity, implement provider-specific exact or bounded accounting for every billable category, and establish an external cap whose documented scope and enforcement meet the maximum pilot exposure. Provider controls with known delayed enforcement are not sufficient to claim the run budget as a hard limit.

Preflight reports missing keys only as environment variable names; it does not test whether credentials are valid. No paid API calls are made by preflight or tests.
