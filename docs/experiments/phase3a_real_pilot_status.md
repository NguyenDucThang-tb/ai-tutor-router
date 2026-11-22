# Phase 3A real pilot status

**Readiness: `REAL_PILOT_BLOCKED`**  
**Real pilot: `NOT_RUN`**  
**Real paired benchmark: `NOT_RUN`**  
**Training labels: `TRAINING_LABELS_NOT_AVAILABLE`**
**Phase 3B: `NOT_STARTED`**

## Evidence ledger (2026-11-22)

| Evidence level | State |
|---|---|
| Code implemented | Offline provider metadata verifier; immutable run-configuration digest; HMAC-signed approval validation; existing SQLite per-request reservations and uncertainty retention. |
| Offline tests | Fake gateway and ledger tests; provider verification is tested to make no network calls; installed LiteLLM 1.110.2 Gemini mapping checked from local package data. |
| Public provider documentation | Current model, price, token-counting, lifecycle, and billing controls reviewed; see `provider_verification.md`. |
| Account access / model access | `NOT_VERIFIED`; no account API or console was accessed. |
| Billing scope / external hard ceiling | `NOT_VERIFIED`; published caps have scope or delay limitations. |
| Paid authorization | `NOT_APPROVED`; artifact remains unsigned with null approval identity and timestamp. |
| Real model execution | `NOT_RUN`; no paid API calls. |
| Scientific comparison / training labels | `NOT_AVAILABLE`; no model output or correctness evidence. |

The committed plan validates against the local pinned GSM8K data. Its SHA-256 is `1d5065acc16b50c1521422ec2a912ed28a10bd483d905054fbdd24aefd642493`; sample-ID SHA-256 is `077275a61e532506eab2d10c09d36960976253969380ca256af7b95cbcced5ed`. It selects 10 validation samples and 20 paired primary requests. These values are preparation evidence, not execution evidence.

The selected candidate pair remains Gemini Flash-Lite / Flash for offline preparation only. The real run remains blocked because exact billable exposure and a synchronous, appropriately scoped external spend ceiling are not established, account/model access is unknown, and no user approval exists. Local SQLite enforcement cannot constrain shared-account use or the provider invoice. The ordinary app's configured LiteLLM path is not covered by the benchmark SQLite ledger.
