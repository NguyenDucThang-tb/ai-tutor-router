"""Offline-safe support for the Kaggle open-weight GSM8K pilot.

This module deliberately does not import a model runtime or invoke a model. The
notebook supplies an inference callback only after its explicit run flag is on.
Local records are a separate schema and are never Phase 3A provider records.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from datasets.schemas import DatasetManifest
from datasets.validation import load_processed_dataset
from experiments.collection_plan import plan_hash, validate_collection_plan

PILOT_PLAN_SHA256 = "1d5065acc16b50c1521422ec2a912ed28a10bd483d905054fbdd24aefd642493"
PILOT_SAMPLE_IDS_SHA256 = "077275a61e532506eab2d10c09d36960976253969380ca256af7b95cbcced5ed"
PILOT_DATASET_SHA256 = "aa75c686332c09e12ce6412571b32b7efae779dfc8696d3448b29a45e5542e4b"
PILOT_SAMPLE_COUNT = 10
LOCAL_RECORD_SCHEMA = "kaggle-local-model-record-v1"


def sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def load_frozen_pilot(
    dataset_dir: Path, manifest_path: Path, plan_path: Path
) -> tuple[list[Any], DatasetManifest, dict[str, Any]]:
    """Load and verify all frozen source artifacts; reject any plan drift."""
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text("utf-8"))
    records = load_processed_dataset(dataset_dir, manifest)
    plan = json.loads(plan_path.read_text("utf-8"))
    selected = validate_collection_plan(
        plan,
        records,
        manifest,
        expected_split="validation",
        expected_count=PILOT_SAMPLE_COUNT,
    )
    ids = [row.sample_id for row in selected]
    if plan_hash(plan) != PILOT_PLAN_SHA256:
        raise ValueError("frozen collection plan hash differs from the approved pilot")
    if manifest.dataset_content_sha256 != PILOT_DATASET_SHA256:
        raise ValueError("frozen dataset checksum differs from the approved pilot")
    if sample_ids_sha256(ids) != PILOT_SAMPLE_IDS_SHA256:
        raise ValueError("frozen sample ID hash differs from the approved pilot")
    return selected, manifest, plan


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records and reject malformed or repeated sample entries."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid result JSON at {path}:{line_number}") from exc
        if record.get("schema_version") != LOCAL_RECORD_SCHEMA:
            raise ValueError(f"unsupported local record schema at {path}:{line_number}")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"missing or duplicate sample_id at {path}:{line_number}")
        if (
            record.get("provider") != "local_transformers"
            or record.get("scientific_eligible") is not False
        ):
            raise ValueError("local records must remain separate from provider scientific records")
        seen.add(sample_id)
        records.append(record)
    return records


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    """Durably append one local result before proceeding to the next sample."""
    _validate_local_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def write_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Atomically export the current evaluated view without altering checkpoints."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            _validate_local_record(record)
            handle.write(
                json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def execute_samples(
    samples: Sequence[Any],
    output_path: Path,
    *,
    model_id: str,
    model_revision: str,
    run_enabled: bool,
    infer_one: Callable[[Any], Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Execute each absent sample at most once, preserving failures on restart.

    When disabled, the callback is never invoked. Existing terminal records,
    including errors and OOMs, are preserved and are never replayed implicitly.
    To re-run a failed item, the researcher must explicitly archive/remove its
    result line and record that intervention in the run manifest.
    """
    records = read_records(output_path)
    expected_provenance = dict(provenance)
    expected_ids = {sample.sample_id for sample in samples}
    for record in records:
        if record["sample_id"] not in expected_ids:
            raise ValueError("checkpoint contains a sample outside the frozen execution set")
        identity = {
            "model_id": model_id,
            "model_revision": model_revision,
            **expected_provenance,
        }
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("checkpoint provenance does not match the requested frozen run")
        expected_sample = next(row for row in samples if row.sample_id == record["sample_id"])
        if record.get("reference_answer") != expected_sample.reference_answer:
            raise ValueError("checkpoint reference answer does not match the frozen dataset")
    by_id = {record["sample_id"]: record for record in records}
    if not run_enabled:
        return [by_id[sample.sample_id] for sample in samples if sample.sample_id in by_id]
    for sample in samples:
        if sample.sample_id in by_id:
            continue
        base: dict[str, Any] = {
            **dict(provenance),
            "schema_version": LOCAL_RECORD_SCHEMA,
            "provider": "local_transformers",
            "scientific_eligible": False,
            "model_id": model_id,
            "model_revision": model_revision,
            "sample_id": sample.sample_id,
            "reference_answer": sample.reference_answer,
            "status": "failed",
            "model_output": None,
            "error_type": None,
            "error_message": None,
        }
        try:
            outcome = dict(infer_one(sample))
            protected = {
                "schema_version": LOCAL_RECORD_SCHEMA,
                "provider": "local_transformers",
                "scientific_eligible": False,
                "model_id": model_id,
                "model_revision": model_revision,
                "sample_id": sample.sample_id,
                "reference_answer": sample.reference_answer,
                **expected_provenance,
            }
            if any(key in outcome and outcome[key] != value for key, value in protected.items()):
                raise ValueError("inference outcome attempted to overwrite frozen local provenance")
            base.update(outcome)
            base["status"] = outcome.get("status", "completed")
        except Exception as exc:  # Persist runtime failures; never fabricate output.
            if getattr(exc, "fatal_run_error", False):
                raise
            base["status"] = "failed"
            base["error_type"] = type(exc).__name__
            base["error_message"] = str(exc)[:2000]
        append_record(output_path, base)
        by_id[sample.sample_id] = base
    return [by_id[sample.sample_id] for sample in samples if sample.sample_id in by_id]


def _validate_local_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != LOCAL_RECORD_SCHEMA:
        raise ValueError("unsupported local record schema")
    if (
        record.get("provider") != "local_transformers"
        or record.get("scientific_eligible") is not False
    ):
        raise ValueError("local records cannot claim provider execution or scientific eligibility")
    if not isinstance(record.get("sample_id"), str) or not record["sample_id"]:
        raise ValueError("local record requires a sample ID")


def evaluate_records(
    records: Iterable[Mapping[str, Any]], evaluator: Callable
) -> list[dict[str, Any]]:
    """Evaluate only successful outputs, leaving failures explicit and intact."""
    evaluated: list[dict[str, Any]] = []
    for source in records:
        record = dict(source)
        if record.get("status") == "completed" and isinstance(record.get("model_output"), str):
            result = evaluator(record["model_output"], record["reference_answer"])
            record["evaluation"] = result.model_dump(mode="json")
        else:
            record["evaluation"] = None
        evaluated.append(record)
    return evaluated


def _percentile95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def model_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sample_ids = [record.get("sample_id") for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("model metrics records contain duplicate sample IDs")
    attempts = [record for record in records if record.get("status") != "not_run_input_too_long"]
    successes = [record for record in records if _generation_completed(record)]
    latencies_by_id = {
        record.get("sample_id"): _metric_value(record, "inference_latency_s") for record in records
    }
    tokens_by_id = {
        record.get("sample_id"): _metric_value(record, "output_tokens", integer=True)
        for record in records
    }
    allocated_by_id = {
        record.get("sample_id"): _metric_value(record, "peak_gpu_allocated_bytes", integer=True)
        for record in records
    }
    reserved_by_id = {
        record.get("sample_id"): _metric_value(record, "peak_gpu_reserved_bytes", integer=True)
        for record in records
    }
    determinate = [record for record in successes if _evaluation_determinate(record)]
    correct = sum(
        (record.get("evaluation") or {}).get("is_correct") is True for record in determinate
    )
    latencies = [
        latencies_by_id.get(record.get("sample_id"))
        for record in successes
        if latencies_by_id.get(record.get("sample_id")) is not None
    ]
    tokens = [
        tokens_by_id.get(record.get("sample_id"))
        for record in successes
        if tokens_by_id.get(record.get("sample_id")) is not None
    ]
    allocated = [
        allocated_by_id.get(record.get("sample_id"))
        for record in successes
        if allocated_by_id.get(record.get("sample_id")) is not None
    ]
    reserved = [
        reserved_by_id.get(record.get("sample_id"))
        for record in successes
        if reserved_by_id.get(record.get("sample_id")) is not None
    ]
    throughput_records = [
        (tokens_by_id.get(record.get("sample_id")), latencies_by_id.get(record.get("sample_id")))
        for record in successes
        if tokens_by_id.get(record.get("sample_id")) is not None
        and latencies_by_id.get(record.get("sample_id")) is not None
        and latencies_by_id[record.get("sample_id")] > 0
    ]
    throughput_latency = sum(latency for _, latency in throughput_records)
    throughput_tokens = sum(token_count for token_count, _ in throughput_records)
    missing_latency_count = sum(
        latencies_by_id.get(record.get("sample_id")) is None
        or latencies_by_id[record.get("sample_id")] <= 0
        for record in successes
    )
    missing_token_count = sum(
        tokens_by_id.get(record.get("sample_id")) is None for record in successes
    )
    return {
        "attempted_samples": len(attempts),
        "successful_generations": len(successes),
        "determinately_evaluated_samples": len(determinate),
        "correct_samples": correct,
        "accuracy": (correct / len(determinate)) if determinate else None,
        "accuracy_denominator": len(determinate),
        "mean_inference_latency_s": statistics.mean(latencies) if latencies else None,
        "median_inference_latency_s": statistics.median(latencies) if latencies else None,
        "p95_inference_latency_s": _percentile95(latencies),
        "mean_output_tokens": statistics.mean(tokens) if tokens else None,
        "total_output_tokens": sum(tokens),
        "peak_gpu_allocated_bytes": max(allocated, default=None),
        "peak_gpu_reserved_bytes": max(reserved, default=None),
        "tokens_per_second": throughput_tokens / throughput_latency if throughput_latency else None,
        "throughput_sample_count": len(throughput_records),
        "missing_latency_count": missing_latency_count,
        "missing_token_count": missing_token_count,
        "unknown_evaluation_samples": sum(
            _generation_completed(record) and not _evaluation_determinate(record)
            for record in records
        ),
        "failed_samples": sum(
            record.get("status") == "failed"
            or (record.get("status") == "completed" and not _generation_completed(record))
            for record in records
        ),
        "not_executed_samples": sum(
            str(record.get("status", "")).startswith("not_run") for record in records
        ),
    }


def paired_rows(
    samples: Sequence[Any], small: Sequence[Mapping[str, Any]], strong: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    planned_ids = {sample.sample_id for sample in samples}

    def index_records(records):
        indexed = {}
        for record in records:
            sample_id = record.get("sample_id")
            if sample_id not in planned_ids or sample_id in indexed:
                raise ValueError("paired results contain an unknown or duplicate sample ID")
            indexed[sample_id] = record
        return indexed

    small_by_id = index_records(small)
    strong_by_id = index_records(strong)
    rows = []
    for sample in samples:
        left, right = small_by_id.get(sample.sample_id), strong_by_id.get(sample.sample_id)
        left_correct = (left.get("evaluation") or {}).get("is_correct") if left else None
        right_correct = (right.get("evaluation") or {}).get("is_correct") if right else None
        if not _generation_completed(left) or not _generation_completed(right):
            outcome = "missing_generation"
        elif not _evaluation_determinate(left) or not _evaluation_determinate(right):
            outcome = "unknown_evaluation"
        elif left_correct and right_correct:
            outcome = "both_correct"
        elif left_correct:
            outcome = "small_only_correct"
        elif right_correct:
            outcome = "strong_only_correct"
        else:
            outcome = "both_incorrect"
        rows.append(
            {
                "sample_id": sample.sample_id,
                "reference_answer": sample.reference_answer,
                "small_status": left.get("status") if left else "missing",
                "small_output": left.get("model_output") if left else None,
                "small_evaluation_status": (left.get("evaluation") or {}).get("status")
                if left
                else None,
                "small_correct": left_correct,
                "strong_status": right.get("status") if right else "missing",
                "strong_output": right.get("model_output") if right else None,
                "strong_evaluation_status": (right.get("evaluation") or {}).get("status")
                if right
                else None,
                "strong_correct": right_correct,
                "paired_outcome": outcome,
            }
        )
    return rows


def _metric_value(
    record: Mapping[str, Any], field: str, *, integer: bool = False
) -> int | float | None:
    value = record.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be numeric")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite non-negative number") from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    if integer and not numeric.is_integer():
        raise ValueError(f"{field} must be a finite non-negative integer")
    return int(value) if integer else numeric


def _generation_completed(record: Mapping[str, Any] | None) -> bool:
    return bool(
        record
        and record.get("status") == "completed"
        and isinstance(record.get("model_output"), str)
        and record["model_output"].strip()
    )


def _evaluation_determinate(record: Mapping[str, Any] | None) -> bool:
    if not record:
        return False
    evaluation = record.get("evaluation") or {}
    return evaluation.get("status") in {"correct", "incorrect"} and isinstance(
        evaluation.get("is_correct"), bool
    )


def paired_metrics(
    samples: Sequence[Any],
    small: Sequence[Mapping[str, Any]],
    strong: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Partition planned pairs by generation and evaluation completeness.

    A completed generation pair has a successful output string from both models.
    A determinate pair additionally has correct/incorrect evaluator results for
    both outputs. Unknown pairs generated both outputs but at least one result
    was not determinately evaluated. Missing pairs lack a successful output from
    either model. These last three counts partition planned pairs.
    """
    rows = paired_rows(samples, small, strong)
    outcome_counts = dict(sorted(Counter(row["paired_outcome"] for row in rows).items()))
    planned = len(samples)
    completed = sum(row["paired_outcome"] != "missing_generation" for row in rows)
    determinate = sum(
        outcome_counts.get(name, 0)
        for name in (
            "both_correct",
            "small_only_correct",
            "strong_only_correct",
            "both_incorrect",
        )
    )
    unknown = outcome_counts.get("unknown_evaluation", 0)
    missing = outcome_counts.get("missing_generation", 0)
    if sum(outcome_counts.values()) != planned or determinate + unknown + missing != planned:
        raise ValueError("paired outcome accounting does not partition the planned sample set")
    return {
        "planned_pair_count": planned,
        "completed_generation_pair_count": completed,
        "determinate_pair_count": determinate,
        "unknown_pair_count": unknown,
        "missing_pair_count": missing,
        "paired_outcome_counts": outcome_counts,
    }


def notebook_scientific_status(
    samples: Sequence[Any],
    small_full: Sequence[Mapping[str, Any]],
    strong_full: Sequence[Mapping[str, Any]],
    small_smoke: Sequence[Mapping[str, Any]],
    strong_smoke: Sequence[Mapping[str, Any]],
    *,
    small_model_id: str,
    small_model_revision: str,
    strong_model_id: str,
    strong_model_revision: str,
    expected_provenance: Mapping[str, Any],
    required_evidence: bool,
    full_run_marker: Mapping[str, Any] | None = None,
) -> str:
    """Derive display status from successful local outputs and verified provenance."""
    planned_ids = [sample.sample_id for sample in samples]

    def records_match(records, model_id, revision, run_mode, expected_ids, *, require_eval):
        if len(records) != len(expected_ids):
            return False
        by_id = {}
        for record in records:
            sample_id = record.get("sample_id")
            if sample_id not in expected_ids or sample_id in by_id:
                return False
            by_id[sample_id] = record
        if set(by_id) != set(expected_ids):
            return False
        for record in by_id.values():
            if (
                record.get("schema_version") != LOCAL_RECORD_SCHEMA
                or record.get("provider") != "local_transformers"
                or record.get("scientific_eligible") is not False
                or record.get("model_id") != model_id
                or record.get("model_revision") != revision
                or record.get("status") != "completed"
                or not isinstance(record.get("model_output"), str)
                or not record["model_output"].strip()
                or record.get("run_mode") != run_mode
                or any(record.get(key) != value for key, value in expected_provenance.items())
            ):
                return False
            if require_eval and not _evaluation_determinate(record):
                return False
        return True

    marker_fields = {
        "schema_version": "kaggle-full-run-start-v1",
        "status": "started",
        "sample_ids": planned_ids,
        "dataset_revision": expected_provenance.get("dataset_revision"),
        "dataset_content_sha256": expected_provenance.get("dataset_content_sha256"),
        "collection_plan_sha256": expected_provenance.get("collection_plan_sha256"),
        "prompt_version": expected_provenance.get("prompt_version"),
        "original_prompt_sha256": expected_provenance.get("original_prompt_sha256"),
        "small_model_id": small_model_id,
        "small_model_revision": small_model_revision,
        "strong_model_id": strong_model_id,
        "strong_model_revision": strong_model_revision,
    }
    marker_valid = bool(full_run_marker) and all(
        full_run_marker.get(key) == value for key, value in marker_fields.items()
    )

    complete = (
        required_evidence
        and marker_valid
        and records_match(
            small_full,
            small_model_id,
            small_model_revision,
            "full",
            planned_ids,
            require_eval=True,
        )
        and records_match(
            strong_full,
            strong_model_id,
            strong_model_revision,
            "full",
            planned_ids,
            require_eval=True,
        )
    )
    if complete:
        return "KAGGLE_PILOT_COMPLETED"

    full_execution_started = full_run_marker is not None or bool(small_full or strong_full)
    if full_execution_started:
        return "KAGGLE_PILOT_INCOMPLETE"

    smoke_id = planned_ids[:1]
    smoke_passed = (
        required_evidence
        and bool(smoke_id)
        and records_match(
            small_smoke,
            small_model_id,
            small_model_revision,
            "smoke",
            smoke_id,
            require_eval=False,
        )
        and records_match(
            strong_smoke,
            strong_model_id,
            strong_model_revision,
            "smoke",
            smoke_id,
            require_eval=False,
        )
    )
    if smoke_passed:
        return "KAGGLE_SMOKE_PASSED"
    return "NOTEBOOK_READY_FOR_KAGGLE"


def write_paired_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        list(rows[0])
        if rows
        else [
            "sample_id",
            "reference_answer",
            "small_status",
            "small_output",
            "small_evaluation_status",
            "small_correct",
            "strong_status",
            "strong_output",
            "strong_evaluation_status",
            "strong_correct",
            "paired_outcome",
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
