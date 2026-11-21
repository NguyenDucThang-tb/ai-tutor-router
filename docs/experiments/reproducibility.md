# Reproducibility

Use `uv sync --frozen --extra dev`, record the branch commit, prepare a dataset from its pinned source revision, and retain the generated dataset manifest and file hashes with the run artifacts. Config files hold model identifiers and limits, never credentials. The run manifest records Git SHA, dataset revision/content and split hashes, prompt version/hash, model/provider identifiers, evaluator version, temperature/token cap/seed, concurrency, timeout, retry/fallback, pricing provenance, versions, timestamp, and mode.

Run IDs derive from benchmark protocol configuration and selected sample content; retry/recovery approval switches do not change the run ID. SQLite leases, unique claim tokens, and fencing protect concurrent ownership. Expired pre-invocation leases are safe to reclaim; expired in-flight leases remain uncertain and real reissue requires an explicit approval reference. Stable request IDs are for traceability only. No exactly-once guarantee or provider idempotency support is assumed. Provider-side availability, aliases, infrastructure, and nondeterministic outputs can still change; temperature zero and a seed do not guarantee identical completions. Preserve the original records and manifest when comparing evaluator versions. Replay does not call a provider and retains failed, interrupted, uncertain, and not-started records.

Verification commands:

```bash
uv lock --check
uv sync --frozen --extra dev
uv run ruff check app datasets evaluation experiments tests
uv run ruff format --check app datasets evaluation experiments tests
uv run python -m compileall -q app datasets evaluation experiments tests
uv run pytest
```
