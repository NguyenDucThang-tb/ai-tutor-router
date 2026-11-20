# Phase status and handoff

## Phase 0 — Research and architecture: COMPLETE (initial focused review)

Created the literature review, comparison, gap, research questions, methodology, architecture, routing policy, benchmark protocol, references, limitations, and milestone status. The review is focused rather than systematic. Candidate model pair selection is deferred until provider access, budget, and deployment constraints are known.

## Phase 1 — Baseline infrastructure: IMPLEMENTED / AWAITING REMOTE CI

Implemented four deterministic routers, feature extraction, typed decisions, async mock and LiteLLM gateways, typed settings, FastAPI endpoints, request IDs, structured errors, bearer authentication, process-local rate limiting, JSON logs, bounded retry/fallback, tests, GitHub Actions, and Docker configuration. The mock is a transport stub, not an LLM, and does not answer math questions.

The LiteLLM adapter uses async `acompletion` and sets `num_retries=0` and `max_retries=0`, so its own bounded retry loop is the only retry loop. It retries rate-limit, timeout, and temporary failures; authentication and invalid responses stop immediately. At most one configured fallback model gets one attempt. Every attempt is returned in response metadata. Adapter calls were tested using an injected async mock; no paid API call was made.

If a per-request budget is set, real calls fail closed because no validated token/pricing estimator exists. This is not dynamic budget optimization. LiteLLM-calculated cost is not provider billing; actual billed cost remains unknown unless supplied by a provider.

Authentication is explicitly disabled in development/test defaults. When enabled, a non-empty bearer token is required. Production rejects mock mode and requires auth and LiteLLM credentials. Rate limiting is in-memory and process-local, not shared across workers.

## Verification performed

- `uv lock` — passed; `uv.lock` records exact resolved dependencies.
- `uv sync --extra dev` — passed in the project-local ignored `.venv` on Python 3.12.
- `uv run pytest -q` — 51 passed (mock gateway and mocked LiteLLM calls; no paid API).
- `uv run python -m compileall -q app tests` — passed.
- `uv run ruff check app tests` and `uv run ruff format --check app tests` — passed.
- `uv lock --check` and `docker compose config --quiet` — passed.
- `docker build -t ai-tutor-router:phase1 .` — passed. Container smoke check reached Docker `healthy`; `/health/ready` and `/v1/router/strategies` returned the expected mock responses.
- GitHub Actions is configured; remote CI on the final phase PR head is still pending.
- No real model/API call, dataset run, benchmark, or scientific measurement was performed.

## Next gates

1. Verify GitHub Actions on the exact PR head.
3. Only use a real provider after credentials, model IDs, and pricing provenance are reviewed.
4. Later phases may implement dataset ingestion, learned routing, and prompt chains; preserve train/validation/test boundaries.

Current readiness: local mock functionality, 51 tests, and Docker build/runtime are verified. Phase 1 is ready for review while remote CI completes. No scientific conclusion is available.
