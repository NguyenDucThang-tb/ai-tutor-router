# Reproducibility

Use `uv sync --frozen --extra dev`, record the branch commit, prepare a dataset from its pinned source revision, and retain the generated dataset manifest and file hashes with the run artifacts. Config files hold model identifiers and limits, never credentials. The run manifest records Git SHA, dataset revision/content and split hashes, prompt version/hash, model/provider identifiers, evaluator version, temperature/token cap/seed, concurrency, timeout, retry/fallback, pricing provenance, versions, timestamp, and mode.

Run IDs derive from canonical config and selected sample content. SQLite protects execution claims and checkpoint records. Provider-side availability, aliases, infrastructure, and nondeterministic outputs can still change; temperature zero and a seed do not guarantee identical completions. Preserve the original records and manifest when comparing evaluator versions. Replay does not call a provider.

Verification commands:

```bash
uv lock --check
uv sync --frozen --extra dev
uv run ruff check app datasets evaluation experiments tests
uv run ruff format --check app datasets evaluation experiments tests
uv run python -m compileall -q app datasets evaluation experiments tests
uv run pytest
```
