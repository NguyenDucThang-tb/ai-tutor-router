# Phase 3A real pilot protocol (prepared, not authorized)

## Frozen proposal

- Dataset: pinned GSM8K
- Split: validation
- Seed: 42
- Sample count: 10 (20 paired model requests)
- Candidate small model: `gemini/gemini-3.5-flash-lite`
- Candidate strong model: `gemini/gemini-3.8-flash`
- Retries: 0; fallback: disabled; concurrency: 1
- Temperature: 0; max output: 256 tokens
- Prompt: `math-direct-v1`; evaluator: deterministic `numeric-symbolic-v1`
- Collection plan: versioned, validation-only, exact dataset content checksum and IDs

`configs/phase3a_pilot.example.yaml` records this proposal. The plan is frozen at `configs/phase3a_pilot.collection-plan.json`, generated from the locally verified pinned dataset and validated against its complete source checksum. Plan SHA-256: `1d5065acc16b50c1521422ec2a912ed28a10bd483d905054fbdd24aefd642493`; selected-ID SHA-256: `077275a61e532506eab2d10c09d36960976253969380ca256af7b95cbcced5ed`. Do not substitute or sort-prefix samples if plan validation fails. Both model arms receive the same exact selected samples through the collection-plan-bound benchmark CLI.

## Authorization and acceptance

This proposal is not spend approval. `configs/phase3a_pilot.authorization.json` binds the expected immutable provider, model pair, dataset revision and checksum, split, exact sample IDs, plan hash, prompt, generation settings, request/attempt envelope, and local budget values. It has a SHA-256 digest, null approver fields, `NOT_APPROVED`, and no signature. File presence or an editable YAML boolean cannot authorize a call.

If a future review authorizes this exact run, the approval metadata must match `experiment_approval_id`; an HMAC signature must be generated outside this repository using the separately managed `TUTOR_PAID_APPROVAL_HMAC_KEY` with at least 32 bytes. The secret is never stored in source or printed. Preflight and the runner independently verify the signature and the exact current context. The run manifest stores both the signed artifact SHA-256 and immutable configuration SHA-256 for audit. The HMAC is integrity evidence for an authorization workflow, not proof of provider access, billing scope, or spend limits. The technical paid gate remains mandatory and currently fails closed. Do not set an approval status or key until the user separately authorizes a concrete experiment.

Preflight performs no inference. The runner rejects missing, mismatched, unsigned, or tampered approval artifacts before run task creation and provider invocation. Injected gateways cannot authorize production paid execution. Even a valid signature would not override failed token, cost, account, or external-spend checks.

After a separately approved and technically admitted run, retain per-model output, request/attempt IDs, input/output/reasoning usage, usage-derived cost, provider-account reconciliation, latencies, evaluations, failures, incomplete pairs, plan hash and source hashes. Report all 10 planned pairs, actual completed pairs, incomplete pairs and failures separately. Ten examples support only a pilot smoke of the protocol, not generalizable accuracy or savings claims.

Current pilot state: `NOT_RUN`; readiness: `REAL_PILOT_BLOCKED`.
