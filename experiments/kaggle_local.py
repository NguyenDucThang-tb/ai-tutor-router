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
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    attempts = [record for record in records if record.get("status") != "not_run_input_too_long"]
    successes = [record for record in records if record.get("status") == "completed"]
    determinate = [
        record
        for record in successes
        if (record.get("evaluation") or {}).get("status") in {"correct", "incorrect"}
    ]
    correct = sum(
        (record.get("evaluation") or {}).get("is_correct") is True for record in determinate
    )
    latencies = [
        float(record["inference_latency_s"])
        for record in successes
        if record.get("inference_latency_s") is not None
    ]
    tokens = [int(record.get("output_tokens") or 0) for record in successes]
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
        "peak_gpu_allocated_bytes": max(
            (r.get("peak_gpu_allocated_bytes", 0) or 0 for r in successes), default=None
        ),
        "peak_gpu_reserved_bytes": max(
            (r.get("peak_gpu_reserved_bytes", 0) or 0 for r in successes), default=None
        ),
        "tokens_per_second": (sum(tokens) / sum(latencies))
        if latencies and sum(latencies)
        else None,
        "unknown_evaluation_samples": sum(
            record.get("status") == "completed"
            and (record.get("evaluation") or {}).get("status") not in {"correct", "incorrect"}
            for record in records
        ),
        "failed_samples": sum(record.get("status") == "failed" for record in records),
        "not_executed_samples": sum(
            str(record.get("status", "")).startswith("not_run") for record in records
        ),
    }


def paired_rows(
    samples: Sequence[Any], small: Sequence[Mapping[str, Any]], strong: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    small_by_id = {record["sample_id"]: record for record in small}
    strong_by_id = {record["sample_id"]: record for record in strong}
    rows = []
    for sample in samples:
        left, right = small_by_id.get(sample.sample_id), strong_by_id.get(sample.sample_id)
        left_correct = (left.get("evaluation") or {}).get("is_correct") if left else None
        right_correct = (right.get("evaluation") or {}).get("is_correct") if right else None
        if left_correct is None or right_correct is None:
            outcome = "at_least_one_unknown"
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
