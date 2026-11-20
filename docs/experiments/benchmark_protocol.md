# Benchmark protocol (planned; no benchmark run yet)

- Dataset: provenance-tagged GSM8K initially; optional MATH and separately curated tutoring examples after license/access review. Preserve official test data. Seed 42. Check duplicates before split.
- Paired comparison: run each eligible question through each policy with frozen model IDs, prompt version, decoding parameters, provider region, and service settings. Random assignment must use a recorded seed.
- Cost/usage: store provider-reported token counts and billed costs when available. Keep catalog-price estimates as a separate field with pricing date. Include router, verification, retries, and fallback calls.
- Latency: record feature extraction, policy inference, each model call, verification, total wall-clock end-to-end; document warm-up, concurrency, ordering, rate limits, retry waits, caching, and failures. Report P50 and P95.
- Quality: normalized exact answer where valid, separate deterministic symbolic checks only on supported classes, and separate instructional rubric. Do not treat an LLM judge as ground truth without validation.
- Analysis: validation-only policy tuning; test once after freeze. Paired bootstrap confidence intervals for key differences; breakdown by task type/difficulty; calibration with Brier score, reliability curve and ECE. Report failures and unsupported evaluator cases.
- Reproducibility record: commit, config, dataset revision/hash, prompt version, model/provider identifiers, timestamp, seed, concurrency, and raw response metadata subject to privacy constraints.

No benchmark results are available in this repository yet.
