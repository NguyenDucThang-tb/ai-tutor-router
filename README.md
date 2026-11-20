# AI Tutor Quality-Aware Router

Research prototype for choosing a processing tier for text-based mathematics tutoring. The research scope compares model routing and conditional solve–verify–explain workflows on quality, cost, and end-to-end latency. It is not a completed thesis system or a production deployment.

## Current phase

**Phase 1: PARTIAL / NOT FULLY VERIFIED.** Phase 0 research and architecture documents are present. The mock-backed service scaffold exists, but API integration and the pytest suite have not been verified in this environment. No learned router, dataset benchmark, or real-model measurements are available.

## Architecture

Client → FastAPI validation → feature extraction → independent routing policy → model gateway → structured response. The research router makes query-level quality/tier decisions; the planned LiteLLM adapter will own provider and deployment mechanics. See `docs/architecture/system_design.md`.

## Current status

- Implemented: FastAPI mock service, request schema, deterministic feature extraction, always-small/always-strong/seeded-random/rule/utility policy scaffolding, and gateway boundary.
- Not implemented: real provider adapters, trained ML/calibrated router, dataset pipeline, adaptive chaining, rate limiting, persistence, or real measurements.
- Mock output is plumbing evidence only and is not a mathematical answer.

## Run locally

Python 3.12 is required. Create an isolated virtual environment and install the project with `pip install -e '.[dev]'`. Start the API with `uvicorn app.main:app --reload`. Default provider is a credential-free mock. `TUTOR_AUTH_ENABLED=true` enables bearer-token auth; set `TUTOR_API_KEY` outside source control. Run `pytest` for the mock/unit suite. Never use the mock to infer quality, cost savings, or latency of real models.

Known setup blocker during baseline preparation: this environment could not reach the package index to install dependencies. The source passed Python syntax compilation and dependency-free smoke checks only; API integration and pytest remain unverified.

## Research documents

See `docs/research/` and `docs/architecture/`. The initial benchmark plan uses train/validation/test separation, paired question evaluation, per-task correctness, cost, and end-to-end latency. No model comparison has been run.

Key references: [RouteLLM](https://arxiv.org/abs/2406.18665), [FrugalGPT](https://openreview.net/forum?id=TkXjqcwQ4s), [LiteLLM router concepts](https://docs.litellm.ai/docs/router/learn/concepts), and GSM8K ([Cobbe et al.](https://arxiv.org/abs/2110.14168)). Bibliographic details and further sources are in `docs/research/references.bib`.
