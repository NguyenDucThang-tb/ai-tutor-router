# Provider spend-safety design

## Threat model and measurement terms

The system can estimate token cost; reserve a local maximum exposure; record provider usage metadata; and later compare account reports. These are different measurements:

| Term | Meaning |
|---|---|
| Estimated spending | Price projection from token estimates and a versioned public price entry. |
| Reserved spending | Local SQLite exposure reserved before an invocation under the fenced claim. |
| Observed usage cost | Locally calculated/provider-reported usage cost; not an invoice. |
| Provider-reported billing | Amount returned in account usage reports, subject to provider processing and aggregation. |
| Actual account charges | Final provider/billing-account charge, only established through provider billing records and reconciliation. |

Risks included in review: unknown input tokenization and message framing; reasoning tokens; output generation and provider maximums; retries and fallback; concurrent tasks; process crash before or after invocation; network timeout after possible billing; duplicate execution on resume; delayed billing; price changes; cached/special token categories; missing usage; API keys shared by other clients; and other processes sharing the same provider account. Application-side estimates are not provider hard caps.

## Local ledger invariants

`RunStore` stores exposure as integer micro-USD. Each reservation is bound to a unique execution claim token, run/sample/tier, and JSON pricing provenance. A SQLite `BEGIN IMMEDIATE` transaction verifies live ownership/fencing, checks immutable per-request and run ceilings from the manifest, sums prior retained exposure, and inserts the reservation before it can be marked invoked. A held reservation is released only by the owning claim when provider invocation is known not to have started. Invocation changes it to uncertain; crashes and timeouts retain that exposure. An unknown result remains uncertain. A usage-derived cost is recorded as observed but does not reduce exposure: it is not an independent provider invoice. Stale claims cannot create or change a reservation.

This ledger enforces local arithmetic only. A correct local ledger cannot establish a maximum provider charge if tokenization, billable categories, prices, or provider-side enforcement are not bounded. The current capability registry therefore gives no provider `supports_provable_run_bound=true`; real execution stays blocked.

## Paid admission

`paid_execution_gate.assess_paid_execution()` returns structured PASS/FAIL/BLOCKED. It checks approvals, exact model capability records, account verification, token support, output cap, explicit budgets, retry envelope, and fallback. A supplied gateway is classified as an untrusted adapter; it cannot establish production identity or spending guarantees. No real execution currently passes. This gate is evaluated in the benchmark runner before data processing or provider invocation.

For any future provider admission, a supported model must first have a provider-specific verified token counter for the exact system/user request framing; output/reasoning/cached token semantics; current price categories; verified account controls; and a sufficiently scoped external limit. Required proof must include provider settings/export or operator-confirmed evidence saved with the run. A project cap with a documented billing delay is not a hard per-run ceiling.

## Recovery interaction

Original run budgets stay in the SQLite manifest and are part of strict manifest compatibility. Recovery authorization does not alter the scientific run identity or its original budget. A held reservation from a known-unstarted claim can be released; an invoked, crashed, timed-out, or otherwise uncertain request keeps its exposure and cannot be automatically retried. A paid retry would require explicit recovery approval and a fresh reservation within the original cap. This branch still has no paid path that meets those admission requirements.

## Residual limitations

- SQLite is single-host coordination. Separate database files or machines do not share caps.
- Other processes using the same provider account are invisible to the local ledger.
- Provider reports may lag and may not identify per-request actual charges.
- The ledger is not wired to a provider-specific reservation calculator because no calculator is presently verified as a billable upper bound.
- Provider admin state is not queried by preflight; account access and provider-side limits remain unverified.
- The model request path uses LiteLLM; verified aliases and mapping behavior must be checked against the exact locked LiteLLM version before any future paid calls.
