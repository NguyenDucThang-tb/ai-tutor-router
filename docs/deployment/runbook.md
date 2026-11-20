# Local runbook (prototype)

1. Use Python 3.12 and create an isolated virtual environment.
2. Install with `pip install -e '.[dev]'`.
3. Copy `.env.example` to `.env` and retain `TUTOR_PROVIDER=mock` for the offline demo.
4. Start with `uvicorn app.main:app --host 127.0.0.1 --port 8000`.
5. Check `/health/live`, `/health/ready`, `/v1/router/strategies`, then POST a JSON request to `/v1/tutor/ask`.
6. Run `pytest` for the mock/unit suite.

This is not deployment-ready. Real provider adapter, bounded retry/fallback integration, persistent metrics, rate limiting, request privacy/retention, TLS/reverse proxy, secret management, and operational monitoring still need implementation and verification. Do not enable a paid provider until model IDs, price configuration, and budget controls are reviewed.
