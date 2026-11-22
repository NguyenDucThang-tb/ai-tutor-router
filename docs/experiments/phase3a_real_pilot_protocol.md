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

This is not spend approval. No experiment approval reference is embedded. Preflight performs no inference and must report `BLOCKED` while token/cost/account bounds remain unverified. The runner independently invokes the paid admission gate and refuses the injected gateway shortcut. A separate written approval must identify provider, exact pair, count, attempts, maximum spend and approval reference; that approval still cannot override a failed technical gate.

After a separately approved and technically admitted run, retain per-model output, request/attempt IDs, input/output/reasoning usage, usage-derived cost, provider-account reconciliation, latencies, evaluations, failures, incomplete pairs, plan hash and source hashes. Report all 10 planned pairs, actual completed pairs, incomplete pairs and failures separately. Ten examples support only a pilot smoke of the protocol, not generalizable accuracy or savings claims.

Current pilot state: `NOT_RUN`; readiness: `REAL_PILOT_BLOCKED`.
