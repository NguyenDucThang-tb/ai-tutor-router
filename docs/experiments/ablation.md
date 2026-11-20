# Ablation plan (not run)

| Ablation | Hypothesis | Variable changed | Controls | Metrics / interpretation |
|---|---|---|---|---|
| A1 no quality estimate | Quality signal contributes to cost-quality frontier | Replace q estimate with fixed prior | Same tiers, cost/latency values, validation/test | Quality, cost, calibration; indicates value of signal |
| A2 no latency term | Latency term changes selected policy under delay-sensitive traffic | Set latency weight to zero | Same quality floor and cost inputs | Quality, cost, P50/P95; interpret only when latency estimates exist |
| A3 no cost term | Cost term changes economical selection | Set cost weight to zero | Same quality and latency inputs | Quality, cost, routing shares |
| A4 no adaptive chaining | Conditional chain adds measurable value | Direct vs adaptive chain | Same questions, prompts, candidate model IDs | Rubric/correctness, call count, cost, latency |
| A5 thresholds | Threshold controls risk/coverage trade-off | Sweep thresholds selected on validation | Fixed trained/calibrated estimator | Validation risk/coverage, frozen test operating points |
| A6 representations | Text representations differ in sufficiency prediction | TF-IDF vs sentence embeddings | Same split/classifier/evaluation | AUC, Brier, ECE, policy cost/quality |
| A7 calibration | Calibration changes policy constraint reliability | Raw vs Platt/isotonic | Same model and data | Brier/ECE, threshold risk, small share |

All studies remain planned. Record observations, interpretation, and validity threats only after actual runs.
