# Dataset protocol

## Supported source

The initial adapter supports the official [OpenAI GSM8K repository](https://github.com/openai/grade-school-math), pinned to commit `3101c7d5072418e28b9008a6636bde82a006892c`. Its README defines JSONL `question` and `answer` fields; the final answer follows `####`. The repository is archived. Its included license is MIT ([license](https://github.com/openai/grade-school-math/blob/master/LICENSE)). The adapter hashes downloaded bytes, stores source URLs, revision, access note, download time, and per-file SHA-256 in a manifest. No dataset is committed.

MATH is deferred. The official [Hendrycks MATH repository](https://github.com/hendrycks/math) currently contains no data files; its access issue remains open ([issue 29](https://github.com/hendrycks/math/issues/29)). No mirror or derivative is silently substituted. Revisit only after source access and applicable terms are verified.

## Preparation

```bash
uv run python -m experiments.dataset prepare --dataset gsm8k --seed 42
uv run python -m experiments.dataset validate --dataset gsm8k
```

Defaults write raw source under `data/raw/gsm8k`, normalized JSONL under `data/processed/gsm8k`, and manifests under `data/manifests`. Raw, processed, and generated artifacts are ignored by Git. The official test split remains assigned to test. Only official training rows are grouped and split into train/validation, default 90/10, using seed 42. The configured ratio is the train fraction of the original training rows.

Exact matching normalizes case and Unicode word tokens. Near matches use Jaccard similarity over three-token shingles (token sets for shorter questions) at configurable threshold 0.9. Connected duplicate groups stay in one training-derived split. Potential official-test overlaps are reported but not moved or removed. Validation checks unique IDs, source split preservation, cross-split duplicates, processed-file checksums, and dataset content hash.

`task_type=word_problem` and `language=en` are adapter metadata; difficulty is not inferred. Source fields and inferred metadata are separate in the schema.
