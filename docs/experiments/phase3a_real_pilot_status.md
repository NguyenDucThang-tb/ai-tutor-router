# Phase 3A real pilot status

**Readiness: `REAL_PILOT_BLOCKED`**  
**Real pilot: `NOT_RUN`**  
**Real paired benchmark: `NOT_RUN`**  
**Training labels: `NOT_AVAILABLE`**  
**Phase 3B: `NOT_STARTED`**

The current documented model pair is Gemini 3.5 Flash-Lite and Gemini 3.8 Flash. The provider exposes documented input counting and maximum input/output limits, but no verified exact upper bound for this exact LiteLLM request/billing path. Its project cap is experimental and can overrun during billing-data delay; billing tier caps are account-wide, and actual account/project configuration is unknown. No account credentials or settings were inspected. The 10-sample plan is proposal-only and has not been generated from or committed with the ignored local dataset. There is no experiment approval reference.

| Item | State |
|---|---|
| Official model identity and public standard prices | Verified on 2026-11-22 for Gemini candidate pair |
| Account/model access and effective price tier | Unverified |
| Provider-native input count facility | Documented; exactness for request path not established |
| Output token limits | Documented; not a complete bill bound by themselves |
| Local integer micro-USD ledger | Implemented and offline-tested |
| Atomic local reservation and uncertainty retention | Implemented and offline-tested |
| Provider-level pilot spend bound | Blocked |
| Paid execution gate | Implemented; fails closed |
| Pilot approval | Requires separate user authorization |
| Paid calls | Not run |

The frozen plan contains 10 validation IDs, with plan hash `1d5065acc16b50c1521422ec2a912ed28a10bd483d905054fbdd24aefd642493` and ID hash `077275a61e532506eab2d10c09d36960976253969380ca256af7b95cbcced5ed`. The remaining blocker is substantive: no verified maximum billed amount for each complete provider request plus a sufficiently scoped and instantaneous external limit. Local estimates/reservations cannot replace that guarantee.
