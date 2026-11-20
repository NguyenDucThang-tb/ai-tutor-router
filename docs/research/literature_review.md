# Literature review (Phase 0)

## RouteLLM

Ong et al., “RouteLLM: Learning to Route LLMs with Preference Data,” arXiv:2406.18665 (2024). RouteLLM studies pre-inference selection between a stronger and weaker model from preference comparisons. Its released framework includes matrix-factorization, similarity-weighted, BERT, and causal-LM router variants, with threshold-controlled trade-offs. The threshold is tied to each router's score and training workload; it cannot be copied as a universal probability or threshold. Its preference target and general-chat settings do not directly establish mathematical correctness or instructional quality. [Paper](https://arxiv.org/abs/2406.18665), [code](https://github.com/lm-sys/RouteLLM).

## FrugalGPT

Chen, Zaharia, and Zou, “FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance,” TMLR (2024). It proposes prompt adaptation, model approximation, and a learned LLM cascade; the cascade can invoke low-cost models and escalate based on an answer-quality condition. Unlike a pre-inference router, a cascade sees an earlier model response before deciding whether to spend for a later call, adding latency and invocation cost. The reported savings are workload/model/evaluator specific; the paper's tasks do not establish tutoring benefits. [TMLR paper](https://openreview.net/forum?id=TkXjqcwQ4s), [PDF](https://openreview.net/pdf?id=TkXjqcwQ4s).

## LiteLLM

LiteLLM documentation describes a common provider interface and deployment/model-group routing, retries within a group, fallbacks across groups, cooldowns, and cost/usage hooks. These are operational routing and reliability features. This project should delegate provider transport and deployment availability to LiteLLM where appropriate, while keeping its query-conditioned quality decision as the research component. Its deployment selection does not itself provide a tutoring-quality predictor. APIs evolve, so pin and verify the selected release before integration. [Router concepts](https://docs.litellm.ai/docs/router/learn/concepts), [reliability](https://docs.litellm.ai/docs/router/learn/reliability), [docs](https://docs.litellm.ai/docs/).

## Selective prediction and calibration

Geifman and El-Yaniv, “Selective Classification for Deep Neural Networks,” NeurIPS 2017, formalize prediction with a reject option and risk-coverage trade-offs. Geifman et al., “SelectiveNet: A Selective Prediction Network with Guaranteed Risk,” NeurIPS 2019, jointly learns prediction and selection. These motivate stating a quality constraint alongside coverage (small-tier share); a router's confidence is not correctness itself. This project will report calibration and selective risk by task category, not only aggregate quality. [NeurIPS 2017](https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks.pdf).

Kadavath et al., “Language Models (Mostly) Know What They Know,” arXiv:2207.05221 (2022), study elicited confidence and calibration for question answering. Kuhn, Gal, and Farquhar, “Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation,” ICLR 2023, show why semantically equivalent generations matter when estimating uncertainty. A tutoring router therefore needs outcome labels and calibration on its own question/model/prompt distribution; verbalized confidence alone is not treated as a probability of correctness. [Kadavath et al.](https://arxiv.org/abs/2207.05221), [Kuhn et al.](https://arxiv.org/abs/2302.09664).

## Mathematical evaluation

Cobbe et al., “Training Verifiers to Solve Math Word Problems,” arXiv:2110.14168 (2021), introduce GSM8K and study generated-solution verification for grade-school word problems. Exact final-answer matching can score mathematical outcome, but cannot alone score pedagogy, hints, or explanation quality. Hendrycks et al., “Measuring Mathematical Problem Solving With the MATH Dataset,” NeurIPS 2021, provide a broader competition-math benchmark; it should be an optional harder distribution rather than assumed representative of tutoring traffic. [GSM8K](https://arxiv.org/abs/2110.14168), [MATH](https://arxiv.org/abs/2103.03874).

Wang et al., “Self-Consistency Improves Chain of Thought Reasoning in Language Models,” ICLR 2023, sample multiple reasoning paths and aggregate answers. This is an inference-time compute strategy, not free verification: it adds model calls/tokens, and agreement is not proof. A deterministic symbolic checker has a narrower supported domain and must return inconclusive outside it. Zheng et al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena,” NeurIPS 2023, document judge biases including position and verbosity; any model-judged tutoring rubric must be grounded and audited against human ratings. [Self-consistency](https://arxiv.org/abs/2203.11171), [MT-Bench paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf).

## Synthesis and limits

Prior work establishes model routing, cascades, operational provider routing, and selective prediction foundations. It does not establish that a particular small/strong model pair, teacher rubric, or chaining rule improves tutoring at a given cost and latency. These remain empirical questions. Dataset licensing, contamination, grade-level coverage, and subjective instructional scoring are threats to validity. The literature list is deliberately compact and is not a systematic review.
