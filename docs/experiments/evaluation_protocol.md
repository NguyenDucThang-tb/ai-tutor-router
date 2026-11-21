# Mathematical and tutoring evaluation protocol

The deterministic evaluator is versioned `numeric-symbolic-v1`. It first requires exactly one `FINAL ANSWER:` line from the model. It does not search arbitrary prose for the last number. Competing answers are `ambiguous`; a missing or empty answer is `unparseable`.

Numeric comparison supports integers, finite decimals, signed values, well-formed comma grouping, rational values, simple numeric LaTeX fractions, and `\boxed{}` wrappers using exact rational arithmetic. GSM8K references require exactly one non-empty `####` line; preceding worked solution is preserved separately.

The symbolic fallback accepts a bounded arithmetic AST with numeric constants, `x`, `y`, `z`, unary signs, addition, subtraction, multiplication, division, and integer powers from -4 through 4. It rejects equations and unsupported LaTeX; it never uses `eval` or SymPy `parse_expr`. SymPy simplification is a bounded heuristic equality check, not a general proof. Resource exhaustion and unsupported structures must remain unsupported/unparseable, not be counted as incorrect.

Correctness and instructional quality are separate. `evaluation/rubrics.py` exposes a human review queue and a label schema that cannot mark generated labels reviewed. No LLM judge is treated as ground truth. Future human annotation should add clarity, organization, educational usefulness, age appropriateness, student-mistake handling, reviewer IDs, adjudication, and inter-rater agreement. This phase contains no human ratings.

Evaluator coverage is the fraction of successful outputs with a determinate correct/incorrect result. Unsupported, ambiguous, and unparseable remain separately reported. Mock and replay outputs are not model-quality evidence.
