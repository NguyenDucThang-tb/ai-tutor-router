# Routing policy

For each request, estimate small-model sufficiency, then choose a tier under a validation-selected quality constraint. Let `q_s(x)` be the calibrated probability that the small model meets a task-specific correctness criterion for query `x`. A threshold policy selects small iff `q_s(x) >= tau`; choose `tau` on validation data to minimize average measured cost subject to a minimum correctness constraint. Keep a test set untouched until the policy is frozen.

The utility variant is `U_j(x) = alpha_Q * q_j(x) - alpha_C * C_j(x)/C_ref - alpha_L * L_j(x)/L_ref`, subject to `q_j(x) >= q_min`. Quality is a probability, cost and latency are normalized against documented reference scales. This avoids adding raw USD and milliseconds as if they shared units. Any weights, thresholds, and references are configuration artifacts selected on validation only.

The initial code has interpretable rule features and a deliberately labeled heuristic prior. `quality_aware` and `ml` names are scaffolding, not trained estimators; no quality probability is currently calibrated. Production decisions must include observed usage/latency and model/version/prompt identifiers, with estimated prices kept separate from provider-reported usage.
