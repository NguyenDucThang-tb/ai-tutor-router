# AI Tutor Quality-Aware Router

Research prototype for choosing a processing tier for text-based mathematics tutoring and measuring quality, cost, and latency. This is not a completed thesis system or production deployment.

## Current phase

**Phase 1: PARTIAL / NOT FULLY VERIFIED.** Phase 0 research and architecture documents are present. Deterministic baseline routers, mock and LiteLLM gateways, typed configuration, API reliability controls, tests, CI, and Docker configuration are implemented. The local test suite passes; remote CI and Docker image build/run still need verification. No learned router, dataset benchmark, or real-model measurements are available.

## Architecture

Client → request-ID/body/rate-limit middleware → FastAPI validation/auth → task features → standalone baseline router → async gateway → structured response. The research router is independent from FastAPI and provider APIs. See `docs/architecture/system_design.md`.

## Implemented in Phase 1

- Always-small, always-strong, deterministic random, and rule-based routers.
- Async deterministic mock gateway and a LiteLLM `acompletion` adapter.
- Typed settings, fail-closed production config, bearer auth, process-local rate limiting, request IDs, structured errors, and JSON logs.
- Bounded retries for transient errors and at most one configured fallback attempt.
- Mock-only unit and API tests, GitHub Actions workflow, and Docker/Compose files.

The mock is a plumbing stub; it does not produce or validate mathematical solutions. Phase 1 does not implement learned/calibrated routing, dataset experiments, or adaptive chains.

## Local setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen --extra dev
uv run uvicorn app.main:app --reload
uv run pytest
uv run ruff check app tests
uv run ruff format --check app tests
```

Copy `.env.example` to `.env` for local settings. The default provider is the credential-free mock. Set `TUTOR_AUTH_ENABLED=true` and `TUTOR_API_BEARER_TOKEN` to protect tutor requests. For LiteLLM, use `TUTOR_PROVIDER=litellm`, configure `TUTOR_LITELLM_API_KEY`, and set the model IDs. Keep credentials in a secrets store or ignored `.env`; never include keys in model identifiers.

Choose `TUTOR_ROUTER_STRATEGY` from `always_small`, `always_strong`, `random`, or `rule`. Random routing hashes the configured seed and request ID, so the same request ID deterministically replays the same tier without a shared RNG. Rule routing uses documented feature cues and emits no quality probability.

LiteLLM calls pass `num_retries=0` and `max_retries=0`; the adapter owns one retry loop, limited to rate limits, timeouts, and transient provider failures. Authentication and malformed-response errors are not retried. An optional fallback model is tried once after primary retries are exhausted. Each attempt is included in response metadata.

The rate limiter uses an in-memory sliding window keyed by client IP and is process-local; it is not shared among workers. Logs omit question text, bearer credentials, and provider exception bodies. Latency separates router processing, gateway, individual attempts, and total request time.

If `TUTOR_PER_REQUEST_BUDGET_USD` is configured, real provider requests fail closed because Phase 1 has no verified token/pricing estimator. This prevents an unknown estimate from being treated as zero; it is not dynamic budget optimization. LiteLLM-calculated cost is labeled separately from provider-reported usage and actual billed cost. Actual billing remains unknown unless supplied by a provider.

## Docker

Start the mock service:

```bash
docker compose up --build
curl http://localhost:8000/health/ready
curl -X POST http://localhost:8000/v1/tutor/ask \\
  -H 'Content-Type: application/json' \\
  -d '{"question":"Solve x^2 - 5x + 6 = 0","task_type":"algebra","mode":"direct","student_level":"high_school"}'
docker compose down
```

The container is non-root, read-only, drops capabilities, and uses locked dependencies. Production settings reject mock mode and require bearer auth and LiteLLM credentials. Docker build/runtime still need verification on a host with Docker daemon access.

## Research and references

See `docs/research/` and `docs/architecture/`. The benchmark plan requires held-out evaluation, paired queries, and separate correctness, instructional quality, cost, and latency. No model comparison has been run. Key references: [RouteLLM](https://arxiv.org/abs/2406.18665), [FrugalGPT](https://openreview.net/forum?id=TkXjqcwQ4s), [LiteLLM router concepts](https://docs.litellm.ai/docs/router/learn/concepts), and GSM8K ([Cobbe et al.](https://arxiv.org/abs/2110.14168)). Full citations are in `docs/research/references.bib`.
