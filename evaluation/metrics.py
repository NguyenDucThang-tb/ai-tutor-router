"""Aggregate benchmark records without treating unsupported answers as errors."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from math import ceil, sqrt

from experiments.records import ExperimentRecord


def summarize_records(records: Iterable[ExperimentRecord]) -> dict:
    records = list(records)
    groups: dict[tuple[str, str, str], list[ExperimentRecord]] = defaultdict(list)
    for record in records:
        groups[(record.split, record.model_tier, record.fallback_policy)].append(record)
    summary = {}
    for (split, tier, fallback_policy), group in sorted(groups.items()):
        evaluated = [
            r for r in group if r.execution_status == "success" and r.evaluation is not None
        ]
        determinate = [r for r in evaluated if r.evaluation.is_correct is not None]
        correct = sum(r.evaluation.is_correct is True for r in determinate)
        known = sum(r.metrics.known_calculated_cost_usd or 0 for r in group)
        unknown = sum(r.metrics.unknown_cost_attempts for r in group)
        successes = sum(r.execution_status == "success" for r in group)
        denominator = len(determinate)
        accuracy = correct / denominator if denominator else None
        scientific_determinate = [record for record in determinate if record.scientific_eligible]
        scientific_correct = sum(
            record.evaluation.is_correct is True for record in scientific_determinate
        )
        successful_costs = [r.cost.calculated_usd for r in group if r.execution_status == "success"]
        attempted_costs = [r.cost.calculated_usd for r in group]
        retry_count = sum(r.metrics.retry_count for r in group)
        fallback_count = sum(r.metrics.fallback_count for r in group)
        failure_categories = Counter(
            str(r.provenance["error_category"])
            for r in group
            if r.execution_status in {"provider_failure", "outcome_unknown"}
            and r.provenance.get("error_category")
        )
        latencies = sorted(
            r.metrics.pipeline_end_to_end_ms
            for r in group
            if r.metrics.pipeline_end_to_end_ms is not None
        )
        summary[f"{split}:{tier}:{fallback_policy}"] = {
            "records": len(group),
            "fallback_policy": fallback_policy,
            "execution_status_counts": dict(
                sorted(Counter(record.execution_status for record in group).items())
            ),
            "successful": successes,
            "success_rate": successes / len(group) if group else None,
            "evaluated": denominator,
            "successful_evaluated": len(evaluated),
            "evaluator_coverage": denominator / successes if successes else None,
            "evaluation_missing": successes - len(evaluated),
            "correct": correct,
            "incorrect": sum(r.evaluation.is_correct is False for r in determinate),
            "unparseable": sum(r.evaluation.status == "unparseable" for r in evaluated),
            "unsupported": sum(r.evaluation.status == "unsupported" for r in evaluated),
            "ambiguous": sum(r.evaluation.status == "ambiguous" for r in evaluated),
            "accuracy": accuracy,
            "accuracy_wilson_95": _wilson(correct, denominator),
            "known_calculated_cost_usd": known
            if any(r.metrics.known_calculated_cost_usd is not None for r in group)
            else None,
            "unknown_cost_attempts": unknown,
            "cost_is_complete": unknown == 0
            and all(r.cost.calculated_usd is not None for r in group),
            "average_cost_per_successful_request_usd": _mean(successful_costs)
            if successful_costs and all(value is not None for value in successful_costs)
            else None,
            "average_cost_per_attempted_request_usd": _mean(attempted_costs)
            if attempted_costs and all(value is not None for value in attempted_costs)
            else None,
            "provider_failures": sum(r.execution_status == "provider_failure" for r in group),
            "uncertain_outcomes": sum(r.execution_status == "outcome_unknown" for r in group),
            "interrupted_records": sum(r.execution_status == "interrupted" for r in group),
            "not_started_records": sum(r.execution_status == "not_started" for r in group),
            "failure_categories": dict(sorted(failure_categories.items())),
            "retry_count": retry_count,
            "fallback_count": fallback_count,
            "mean_pipeline_end_to_end_ms": _mean([r.metrics.pipeline_end_to_end_ms for r in group]),
            "mean_pipeline_elapsed_to_terminal_ms": _mean(
                [r.metrics.pipeline_elapsed_to_terminal_ms for r in group]
            ),
            "p50_pipeline_end_to_end_ms": _percentile(latencies, 0.50),
            "p95_pipeline_end_to_end_ms": _percentile(latencies, 0.95),
            "mean_model_only_latency_ms": _mean([r.metrics.model_only_latency_ms for r in group]),
            "mean_gateway_elapsed_ms": _mean([r.metrics.gateway_elapsed_ms for r in group]),
            "failed_records_retained_in_denominator": len(group) - successes,
            "scientifically_eligible_records": sum(r.scientific_eligible for r in group),
            "evaluation_complete_records": sum(r.evaluation_complete for r in group),
            "scientific_eligible_evaluated": len(scientific_determinate),
            "scientific_eligible_accuracy": scientific_correct / len(scientific_determinate)
            if scientific_determinate
            else None,
        }
    return {
        "groups": summary,
        "paired_outcomes": _paired_outcomes(records),
        "paired_records": _paired_table(records),
        "scientific_note": "Mock and replay records are not model-quality evidence.",
    }


def _mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _wilson(successes: int, trials: int) -> dict | None:
    if trials == 0:
        return None
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = z * sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denominator
    return {"lower": max(0.0, center - margin), "upper": min(1.0, center + margin)}


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return values[max(0, ceil(percentile * len(values)) - 1)]


def _paired_outcomes(records: list[ExperimentRecord]) -> dict[str, int]:
    by_sample: dict[tuple[str, str], dict[str, ExperimentRecord]] = defaultdict(dict)
    for record in records:
        by_sample[(record.split, record.sample_id)][record.model_tier] = record
    groups = {
        key: 0
        for key in (
            "both_correct",
            "small_correct_strong_incorrect",
            "small_incorrect_strong_correct",
            "both_incorrect",
            "evaluator_unknown",
            "execution_failure",
            "incomplete_pair",
        )
    }
    for pair in by_sample.values():
        if "small" not in pair or "strong" not in pair:
            groups["incomplete_pair"] += 1
            continue
        small_record, strong_record = pair["small"], pair["strong"]
        if (
            small_record.execution_status != "success"
            or strong_record.execution_status != "success"
        ):
            groups["execution_failure"] += 1
            continue
        small, strong = small_record.evaluation, strong_record.evaluation
        if small is None or strong is None or small.is_correct is None or strong.is_correct is None:
            groups["evaluator_unknown"] += 1
        elif small.is_correct and strong.is_correct:
            groups["both_correct"] += 1
        elif small.is_correct:
            groups["small_correct_strong_incorrect"] += 1
        elif strong.is_correct:
            groups["small_incorrect_strong_correct"] += 1
        else:
            groups["both_incorrect"] += 1
    return groups


def _paired_table(records: list[ExperimentRecord]) -> list[dict]:
    by_sample: dict[tuple[str, str], dict[str, ExperimentRecord]] = defaultdict(dict)
    for record in records:
        by_sample[(record.split, record.sample_id)][record.model_tier] = record
    result = []
    for (split, sample_id), pair in sorted(by_sample.items()):
        small, strong = pair.get("small"), pair.get("strong")
        if small is None or strong is None:
            outcome = "pair_incomplete"
        elif small.execution_status != "success" or strong.execution_status != "success":
            outcome = "execution_failure"
        elif (
            small.evaluation is None
            or strong.evaluation is None
            or small.evaluation.is_correct is None
            or strong.evaluation.is_correct is None
        ):
            outcome = "evaluator_unknown"
        elif small.evaluation.is_correct and strong.evaluation.is_correct:
            outcome = "both_correct"
        elif small.evaluation.is_correct:
            outcome = "small_correct_strong_incorrect"
        elif strong.evaluation.is_correct:
            outcome = "small_incorrect_strong_correct"
        else:
            outcome = "both_incorrect"
        exemplar = small or strong
        result.append(
            {
                "sample_id": sample_id,
                "split": split,
                "small_correct": small.evaluation.is_correct
                if small and small.evaluation
                else None,
                "strong_correct": strong.evaluation.is_correct
                if strong and strong.evaluation
                else None,
                "outcome": outcome,
                "small_cost_usd": small.cost.calculated_usd if small else None,
                "strong_cost_usd": strong.cost.calculated_usd if strong else None,
                "small_model_only_latency_ms": small.metrics.model_only_latency_ms
                if small
                else None,
                "strong_model_only_latency_ms": strong.metrics.model_only_latency_ms
                if strong
                else None,
                "small_pipeline_end_to_end_ms": small.metrics.pipeline_end_to_end_ms
                if small
                else None,
                "strong_pipeline_end_to_end_ms": strong.metrics.pipeline_end_to_end_ms
                if strong
                else None,
                "small_execution_status": small.execution_status if small else None,
                "strong_execution_status": strong.execution_status if strong else None,
                "task_type": exemplar.task_type,
                "difficulty": exemplar.difficulty,
            }
        )
    return result
