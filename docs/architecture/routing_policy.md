# Routing policy

Phase 1 implements four routing baselines. Each returns a configured model identifier, selected tier, reason code, decision metadata, and measured policy-processing latency. No Phase 1 router emits a quality probability.

## Always-small and always-strong

These route every validated request to the configured small or strong model, respectively.

## Random

For reproducible independent routing, compute a BLAKE2b digest of `seed + NUL + request_id`, map its 64-bit value to [0,1), and compare against `random_small_probability`. The same seed and request ID replay the same tier, independent of call order, concurrency, or global RNG state. The probability is configurable from zero to one.

## Rule baseline

Choose strong if any complexity cue is true: query character/token threshold exceeded, proof request, solution-check intent, math-symbol count above threshold, requested steps, or `word_problem`/`guided` task. Otherwise choose small. Threshold boundaries and cues have unit tests. The rules are a transparent baseline, not a learned quality estimate.

## Future learned policy

A future model may estimate small-model sufficiency and select thresholds on validation data. It must report calibration and task-segment risk, keep test data untouched during tuning, and attach provenance to model, prompt, labels, cost, and latency. Cost and latency estimates are intentionally absent from Phase 1 decisions because no supported measurements/prices exist.
