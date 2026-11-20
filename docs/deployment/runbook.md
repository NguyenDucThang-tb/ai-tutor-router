# Local and container runbook

## Local setup

1. Install Python 3.12 and uv.
2. Run `uv sync --frozen --extra dev` to create `.venv` from `uv.lock`.
3. Copy `.env.example` to `.env`. The default mock provider needs no credentials.
4. Start with `uv run uvicorn app.main:app --host 127.0.0.1 --port 8000`.
5. Run `uv run pytest`, `uv run ruff check app tests`, and `uv run ruff format --check app tests`.

## API smoke request

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
curl http://127.0.0.1:8000/v1/router/strategies
curl -X POST http://127.0.0.1:8000/v1/tutor/ask \\
  -H 'Content-Type: application/json' \\
  -H 'X-Request-ID: demo-1' \\
  -d '{"question":"Solve x^2 - 5x + 6 = 0","task_type":"algebra","mode":"direct","student_level":"high_school"}'
```

The mock response is an explicit marker, not a mathematical answer. Select `TUTOR_ROUTER_STRATEGY` from `always_small`, `always_strong`, `random`, or `rule`. Set `TUTOR_PROVIDER=litellm`, configured model IDs, and `TUTOR_LITELLM_API_KEY` for the LiteLLM adapter. Enable auth with `TUTOR_AUTH_ENABLED=true` and a non-empty `TUTOR_API_BEARER_TOKEN`. Production mode requires authentication and LiteLLM credentials, rejects mock mode, and requires HTTPS for a configured API base.

## Retry, fallback, cost, and rate limits

Retries are capped at three and cover rate-limit, timeout, and transient errors only. Auth and invalid-response errors stop immediately. One configured fallback model gets one attempt after primary retries are exhausted. LiteLLM call retries are disabled to prevent retry multiplication. Per-attempt cost remains available in attempt metadata; aggregate cost and final-attempt cost are separate. Aggregate totals are null when an attempt's value is unknown, while known subtotals and unknown-attempt counts are retained. LiteLLM-calculated cost is identified as calculated, not actual billed cost.

Setting `TUTOR_PER_REQUEST_BUDGET_USD` currently refuses real calls because there is no verified pricing estimator. It never treats an unknown price as zero. Rate limiting is process-local and does not coordinate multiple workers; up to 4,096 client windows are retained, with excess unique clients sharing a bounded overflow window that continues enforcing the configured limit. The tutor request body is capped at 65,536 bytes while streamed, independent of Content-Length. Logs are JSON and omit student input, bearer credentials, and provider exception bodies. Latencies separately record router, individual gateway attempts, gateway total, and application processing time from middleware entry through response assembly. This does not include network transfer to the client.

## CI

GitHub Actions uses Python 3.12, the committed lockfile, mock mode, compileall, Ruff, and pytest. It runs on pushes to `main` and `feat/**`, and on pull requests targeting `main`. CI does not require paid API keys.

## Docker

Build and start in mock mode with `docker compose up --build`; check `http://localhost:8000/health/ready`. Use the curl request above against localhost, then stop with `docker compose down`. The image uses locked dependencies, non-root user, read-only root filesystem, dropped Linux capabilities, and no-new-privileges. The image build and a container health/route smoke check have been verified locally.

## Production caveats

This is a Phase 1 foundation, not a production deployment. Real provider calls, external TLS/reverse proxy, durable privacy/retention controls, distributed rate limiting, verified pricing-based budget enforcement, and persistent monitoring are not implemented or verified. Review credentials, model IDs, and cost provenance before enabling paid requests.
