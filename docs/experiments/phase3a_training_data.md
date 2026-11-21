# Phase 3A future-router label export

Export only from provenance-verified, successful, real small-model outputs after a frozen evaluator run:

```bash
python -m experiments.training_data \
  --records artifacts/runs/<run-id>/records.jsonl \
  --dataset-dir data/processed/gsm8k \
  --manifest data/manifests/gsm8k.json \
  --output-dir artifacts/training_data/<run-id>
```

The exporter loads the checksummed processed dataset and checks sample IDs/splits/source hashes, non-synthetic provenance, real mode, provider/model catalog identity, output hash, attempt provenance, prompt hash, and evaluator version. It rejects duplicate source executions. Only the small tier is used. Mock, replay, failed, malformed-provenance, and test-split executions cannot create labels. Official test rows are counted as excluded and not emitted.

`small_model_sufficient` means only that the deterministic GSM8K correctness evaluator marked the answer correct (`1`) or incorrect (`0`). Unknown evaluator states go to `unknown.jsonl` with null labels. Correctness does not establish pedagogical sufficiency. `query_features` contains only question-derived values available before execution (`char_count`, word-based `token_count`, symbol and intent features, task type, requested step count). Outputs, references, usage, costs, latency, evaluator outcomes, and labels are not router input features.

The exporter writes train, validation, and unknown files plus a provenance/count manifest. Do not call the output a trained dataset or fit a model in Phase 3A. Future collection must remain split-aligned; test is for later frozen evaluation only.
