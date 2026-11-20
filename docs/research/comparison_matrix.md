# Scientific comparison matrix

| System / method | Decision point and signal | Objective | Strength | Limit for this thesis |
|---|---|---|---|---|
| RouteLLM (Ong et al., 2024) | Before generation; preference-trained router score | Strong/weak response preference vs cost | Several learned router families and threshold sweeps | Preference is not equivalent to mathematical correctness or teaching quality; score thresholds are workload-specific |
| FrugalGPT (Chen et al., TMLR 2024) | Sequentially after a cheap answer; answer quality/acceptability condition | Minimize cascade cost while maintaining/improving task quality | Can spend extra calls only on uncertain cases | Additional calls increase end-to-end latency; checker quality and workload-specific thresholds matter |
| LiteLLM router | Provider/deployment selection and failure handling | Availability, load distribution, retry/fallback | Mature provider abstraction and operations | Not a query-level tutoring-quality estimator |
| Selective prediction (Geifman & El-Yaniv, 2017) | Accept/reject based on confidence | Risk/coverage control | Clarifies quality constraints and coverage | Generic classifiers; must define meaningful, calibrated tutoring labels |
| Proposed threshold router | Before inference; features predict small-tier sufficiency | Minimize measured cost under validation quality floor | One-call decisions, easy to audit and measure | Predictor can be miscalibrated or drift; no evidence yet that it works on tutoring |
| Proposed adaptive chain | Before workflow; expected value of verification/explanation | Improve task-specific outcome against added cost/latency | Adds verification only where justified | Verification may be inconclusive; must measure chain overhead and avoid false assurance |

Sources: [RouteLLM](https://arxiv.org/abs/2406.18665), [FrugalGPT](https://openreview.net/forum?id=TkXjqcwQ4s), [LiteLLM router concepts](https://docs.litellm.ai/docs/router/learn/concepts), [selective classification](https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks.pdf).
