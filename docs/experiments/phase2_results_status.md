# Phase 2 results and scientific status

## Completed infrastructure evidence

- GSM8K was prepared from official revision `3101c7d5072418e28b9008a6636bde82a006892c`; downloaded bytes and processed artifacts are locally checksummed.
- Prepared 8,792 records: 6,726 train, 747 validation, and 1,319 untouched official test. Content SHA-256: `aa75c686332c09e12ce6412571b32b7efae779dfc8696d3448b29a45e5542e4b`.
- Deduplication found one near-duplicate pair (similarity 0.933333) grouped within train; validation reported zero cross-split overlaps.
- MATH is deferred because official source data was not available in the inspected upstream repository. No derivative mirror was substituted.
- Evaluator, dataset, replay, label guard, paired mock runner, persistence, and metrics are covered by automated tests. Mock runs are plumbing checks only.

## Not measured

No model-quality benchmark, human instructional-quality rating, real provider call, real token usage, real provider cost, latency comparison, confidence interval from experimental results, significance test, oracle routing bound, or learned router result is claimed. The correctness evaluator has limited numeric and symbolic coverage, and the symbolic subset is not a proof system. Pricing provenance must be supplied and verified before a cost-limited real run.

## Readiness

The infrastructure is ready for code review after CI on the exact branch head. Scientific comparison remains blocked on approved model IDs/provider access and pricing provenance; MATH source access remains deferred. Phase 3 may consume only successful real small-model outputs with determinate correctness labels on train/validation, using the registered criterion. Test examples remain held out. Do not treat mock/replay outputs or the correctness label as pedagogical quality.
