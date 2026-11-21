# Phase status and handoff

## Phase 0 — Research and architecture: COMPLETE (initial focused review)

Created the literature review, comparison, gap, research questions, methodology, architecture, routing policy, benchmark protocol, references, limitations, and milestone status. The review is focused rather than systematic. Candidate model pair selection is deferred until provider access, budget, and deployment constraints are known.

## Phase 1 — Baseline infrastructure: COMPLETE / MERGED

Implemented four deterministic routers, feature extraction, typed decisions, async mock and LiteLLM gateways, typed settings, FastAPI endpoints, request IDs, structured errors, bearer authentication, bounded process-local rate limiting, JSON logs, bounded retry/fallback, tests, GitHub Actions, and Docker configuration. Tutor request bodies are capped at 65,536 bytes while streamed. The mock is a transport stub, not an LLM, and does not answer math questions.

The LiteLLM adapter uses async `acompletion` and sets `num_retries=0` and `max_retries=0`, so its own bounded retry loop is the only retry loop. It retries rate-limit, timeout, and temporary failures; authentication and invalid responses stop immediately. At most one configured fallback model gets one attempt. Every attempt is returned in response metadata alongside aggregate and final-attempt costs. Aggregate totals remain unknown when any attempt's price is unknown, with known subtotals and missing counts retained. Adapter calls were tested using an injected async mock; no paid API call was made.

If a per-request budget is set, real calls fail closed because no validated token/pricing estimator exists. This is not dynamic budget optimization. LiteLLM-calculated cost is not provider billing; actual billed cost remains unknown unless supplied by a provider.

Authentication is explicitly disabled in development/test defaults. When enabled, a non-empty bearer token is required. Production rejects mock mode and requires auth and LiteLLM credentials. Rate limiting is in-memory and process-local, not shared across workers.

## Phase 1 verification

- `uv lock` — passed; `uv.lock` records exact resolved dependencies.
- `uv sync --extra dev` — passed in the project-local ignored `.venv` on Python 3.12.
- `uv run pytest -q` — 61 passed (mock gateway and mocked LiteLLM calls; no paid API).
- `uv run python -m compileall -q app tests` — passed.
- `uv run ruff check app tests` and `uv run ruff format --check app tests` — passed.
- `uv lock --check` and `docker compose config --quiet` — passed.
- `docker build -t ai-tutor-router:phase1-review .` — passed. Container health and mock tutor request passed, including aggregate and final-attempt cost fields.
- Phase 1 was merged to `main` through PR #1.
- No real model/API call, dataset run, benchmark, or scientific measurement was performed.

## Phase 2 — Dataset and benchmark foundation: IMPLEMENTED FOR REVIEW

The `feat/phase2-dataset-benchmark` branch adds the official pinned GSM8K adapter, manifests and checksums, deterministic train/validation split with untouched test, exact/near duplicate reporting, numeric and restricted symbolic evaluation, paired benchmark runner, explicit mock/replay/real modes, SQLite resume state, run manifests, reporting metrics, human review queue, and correctness-based sufficiency labels restricted to real determinate train/validation outputs. MATH is deferred because official source access was not verified. See `docs/experiments/` for protocols.

The actual pinned GSM8K preparation produced 8,792 samples (6,726 train, 747 validation, 1,319 official test), dataset hash `aa75c686332c09e12ce6412571b32b7efae779dfc8696d3448b29a45e5542e4b`. One near-duplicate pair (0.933333) was grouped within train; no cross-split overlap was reported. The raw and processed data remain ignored and untracked.

No model comparison, human quality assessment, paid API call, or scientific model-quality result is claimed. Real provider approval/model/pricing are not configured. CI and Docker checks for this phase are pending exact-head verification.

## Next gates

1. Verify tests, dependency lock, lint, formatting, compile, Docker, and GitHub Actions on the exact Phase 2 head.
2. Only use a real provider after explicit experiment approval, credentials, model IDs, and pricing provenance are reviewed.
3. Preserve test boundaries; Phase 3 can consume only determinate real correctness-based sufficiency labels and must not call them pedagogical quality labels.

Current readiness: Phase 1 is merged. Phase 2 provides an executable, data-backed evaluation foundation for review; no scientific conclusion is available.
