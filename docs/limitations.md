# Limitations

- Current router is a small feature heuristic, not a trained/calibrated ML quality model; its `quality_estimate` is a heuristic prior only.
- Gateway is mock-only. Mock results do not represent model behavior or measured inference.
- No datasets, response labels, tutoring rubric judgments, or real cost/latency measurements are included.
- Chaining, symbolic verification workflow, learned router training, calibration, dataset leakage audit, ablations, and deployment safeguards are planned.
- Optional bearer authentication is minimal and disabled by default. Rate limiting, persistence, PII controls, and provider resilience are absent.
- GSM8K/MATH coverage and answer matching do not alone establish classroom effectiveness; benchmark contamination and subjective scoring remain concerns.
- The literature review is an initial focused review rather than systematic coverage.
