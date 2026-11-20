# Research gap

## What existing work solves

RouteLLM supplies preference-based strong/weak model routers; FrugalGPT explores learned model cascades; LiteLLM provides provider/deployment abstraction, retries, and fallbacks; selective prediction supplies risk/coverage concepts. Mathematical benchmarks such as GSM8K and MATH evaluate mathematical outcomes.

## What remains open for this use case

These tools do not establish a calibrated prediction of whether a cheaper model satisfies both mathematical correctness and a tutoring rubric for a particular student request. A generic preference score is not a correctness guarantee. A cascade may improve quality but costs additional calls and latency. Final-answer correctness alone does not measure whether explanations are useful, appropriately scaffolded, or preserve a learner's own work.

## Proposed investigation

Compare pre-inference policy baselines and a learned predictor of small-model sufficiency on identical math-tutoring questions. Evaluate direct answer, guided tutoring, and selectively activated solve–verify–explain workflows. Calibrate thresholds on validation data and report quality, cost, end-to-end latency, invocation count, calibration, and task-segment outcomes together.

## Testability and novelty boundary

The claim is testable through paired query-level comparisons, held-out evaluation, bootstrap intervals, and fixed prompts/model versions. Applying known routing methods to tutoring is not itself claimed as novel. The proposed contribution is the measured evaluation design and task-specific policy analysis, if results support it. Claims that routing saves cost without unacceptable instructional harm, that calibration transfers across task types, or that chaining pays for its overhead require experiments and are not accepted in advance.
