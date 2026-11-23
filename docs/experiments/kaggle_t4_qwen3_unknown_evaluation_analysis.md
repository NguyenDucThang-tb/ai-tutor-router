# Kaggle T4 Qwen3 Pilot: Complete Unknown-Evaluation Audit

- Analysis date: 2026-11-24
- Reference PR: #6 (not modified by this analysis)
- Run and source-code revision recorded by the archive: `a657738d76c6e56d470eef1bc45a1954a556a665`
- Inference rerun: no
- Original benchmark records and metrics: unchanged
- Scientific status: `KAGGLE_PILOT_INCOMPLETE`

## Evidence and integrity

The user-provided `/home/nguyenducthang/outputs.zip` exists (76,583 bytes); its
SHA-256 is `493cf8e85c549502a1bb050cdc5a38f18b22e72c2a69abac81eeea11ea455c7a`.
It was inspected for duplicate names, absolute/traversal paths and symlinks,
then extracted outside the repository to
`/tmp/ai-tutor-kaggle-audit-0kg3q8gh`. The six requested artifacts were present,
along with `environment.json`, `model_revisions.json`, and two figures. No raw
archive or result records are included in this Git change. Per-artifact hashes
are recorded in [`kaggle_t4_qwen3_posthoc_evaluation.json`](kaggle_t4_qwen3_posthoc_evaluation.json).

The records were audited against the frozen dataset, manifest, collection plan,
model revision file, and paired CSV:

- 10 unique planned sample IDs, in deterministic plan order, appear once in each
  model JSONL and once in the paired CSV. Both model result sets match the same
  plan exactly (20 model records, 10 pairs).
- All IDs belong to the frozen validation selection; references in both model
  files and the paired file match the frozen dataset rows. The checked-in
  processed validation file and collection plan pass the repository's frozen
  pilot loader.
- Dataset revision: `3101c7d5072418e28b9008a6636bde82a006892c`; dataset content
  SHA-256: `aa75c686332c09e12ce6412571b32b7efae779dfc8696d3448b29a45e5542e4b`.
- Processed validation file SHA-256:
  `456dae922cba6379c72348a9d76c6b21f3c3c23f3b2e0d9205ab33513754d540`.
- Collection plan SHA-256:
  `1d5065acc16b50c1521422ec2a912ed28a10bd483d905054fbdd24aefd642493`.
- Ordered sample-ID SHA-256:
  `077275a61e532506eab2d10c09d36960976253969380ca256af7b95cbcced5ed`.
- Small model: `Qwen/Qwen3-1.7B`, revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, FP16. Strong model:
  `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`, NF4
  4-bit with FP16 compute. These identities agree across the manifest, model
  revision artifact, and records.
- The 20 record outputs, evaluations and provenance align with the paired CSV.
  Recomputed original aggregate metrics match `metrics.json` exactly. Running
  the evaluator source from the archive's recorded source commit reproduces all
  20 stored v1 evaluation objects exactly.
- The archive manifest and environment both identify source revision
  `a657738d76c6e56d470eef1bc45a1954a556a665`. They record local Transformers
  execution and `scientific_eligible: false`.

The post-hoc JSON contains per-record sample IDs, references, model/revision,
source record location, original and post-hoc result, and SHA-256 of each raw
output. It deliberately excludes the raw model text. The original raw responses
remain available in the user-supplied archive; they are not reproduced here.

## Evaluator comparison

All 20 responses were re-evaluated offline with `numeric-symbolic-v2` and
compared by sample ID with the v1 result stored in each original record. The
semantic evaluation changed for exactly four records; the other 16 retain the
same status and correctness. Original v1 fields and aggregate benchmark metrics
were not edited.

| Sample ID | Model | Reference | V1 result | V2 result | Evidence-backed mechanism |
|---|---|---:|---|---|---|
| `gsm8k-train-003798` | Qwen3-1.7B | 5 | unparseable, `malformed_expression`; extracted currency-prefixed answer | correct, extracted `$5` | The answer is a numeric literal with one leading dollar sign. V1's strict numeric parser and symbolic fallback rejected that representation; v2 narrowly normalizes the prefix and numeric equality matches 5. |
| `gsm8k-train-000288` | Qwen3-8B | 18 | unparseable, `malformed_expression`; extracted currency-prefixed answer | correct, extracted `$18` | Same confirmed currency-prefix parsing limitation; normalized numeric value equals 18. |
| `gsm8k-train-003336` | Qwen3-8B | 86 | unparseable, `missing_final_answer_delimiter`; no extracted answer | correct, extracted `86` | The complete output ends with a bold Markdown `Final Answer:` label, which v1 did not recognize. V2 accepts this label form and the resulting numeric value equals 86. |
| `gsm8k-train-006450` | Qwen3-8B | 7 | unparseable, `missing_final_answer_delimiter`; no extracted answer | correct, extracted `7` | The complete output uses a bold Markdown `Final Answer` label with punctuation. V1 did not recognize it; v2 extracts 7, equal to the reference. |

These are confirmed evaluator/parser rejections, not confirmed reasoning failures.
The determination uses the full archived outputs, their output hashes, frozen
references, and deterministic evaluator results. V2 does not mark answers
correct merely because they were previously unknown: it extracts a numeric
answer and compares it with the reference. A regression test with a mismatching
reference verifies that currency normalization still produces `incorrect`.
V2 also prefers the strict legacy delimiter if both strict and Markdown labels
occur, avoiding ambiguity introduced by explanatory bold text.

## Original and post-hoc measurements

Original archived v1 metrics remain unchanged:

| Model | V1 correct / determinate | V1 unknown | Determinate-only accuracy | Conservative bound | Optimistic bound |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B FP16 | 9 / 9 | 1 | 9/9 = 100% | 9/10 = 90% | 10/10 = 100% |
| Qwen3-8B NF4 | 7 / 7 | 3 | 7/7 = 100% | 7/10 = 70% | 10/10 = 100% |

After v2 evaluation of all records, both models have 10/10 correct and
determinate evaluations. Paired outcomes change from 6 `both_correct` plus 4
`unknown_evaluation` to 10 `both_correct`; there are zero missing pairs, zero
unknown pairs, and zero determinate discordant pairs. This post-hoc result is
separate from the original record and is labeled with evaluator version and
archive provenance in the JSON artifact.

This is a descriptive ten-sample pilot only. It does not establish statistical
significance or generalization. The post-hoc 100% accuracy is evaluator-version
dependent and should not replace the v1 metrics in the original benchmark.

## Notebook snapshot discrepancy

The saved notebook's preflight cell reports execution commit
`25b56b88ff43728f94e66015ad03374cedf8a973`, while the ZIP manifest and
environment both report `a657738d76c6e56d470eef1bc45a1954a556a665`. The notebook
cell containing the error analysis is byte-for-byte equivalent in content to
the ZIP `error_analysis.md`, but cached notebook aggregate metrics differ from
the ZIP metrics:

| Metric | Notebook small | ZIP small | Notebook strong | ZIP strong |
|---|---:|---:|---:|---:|
| Mean inference latency (s) | 7.240362 | 7.096973 | 13.876868 | 14.215404 |
| Peak allocated GPU bytes | 9,922,677,248 | 3,506,724,352 | 10,486,584,320 | 6,559,726,592 |
| Peak reserved GPU bytes | 10,896,801,792 | 4,460,642,304 | 10,628,366,336 | 6,698,303,488 |
| Tokens/second | 19.626090 | 20.022621 | 11.097605 | 10.833318 |

The archive's run manifest and environment agree with each other and its
per-record metrics reproduce exactly, so this audit uses the ZIP as the
internally consistent artifact set. The exact cause of the notebook-versus-ZIP
snapshot mismatch cannot be established from these artifacts. The two metric
sets must not be combined; the discrepancy remains an explicit provenance
limitation.

## Reproduction and checks

The audit is reproducible without inference using the ZIP SHA-256 above, the
artifact hashes in the post-hoc JSON, the repository commit recorded in the
manifest, the checked-in frozen dataset/plan, and evaluator version
`numeric-symbolic-v2`. The raw ZIP was extracted only under `/tmp`; it is not a
repository artifact. The report and machine-readable comparison preserve no
raw output text.

Full test, lint, format, compile, lock, JSON, and diff-check results are reported
with the analysis commit. No GPU inference or paid-provider call was made.
