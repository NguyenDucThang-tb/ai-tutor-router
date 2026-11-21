"""Leakage-safe export of real, provenance-verified small-model labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from app.features import extract_features
from datasets import gsm8k
from datasets.provenance import sha256_records
from datasets.schemas import DatasetManifest
from datasets.validation import load_processed_dataset
from evaluation.evaluator import EVALUATOR_VERSION, evaluate_answer
from experiments.analysis import load_records
from experiments.collection_plan import validate_collection_plan
from experiments.model_catalog import resolve_model
from experiments.prompts import PROMPT_SHA256

LABEL_DEFINITION = "correctness-based-sufficiency-v1"


def _valid_real_output(record, sample, manifest) -> bool:
    model = resolve_model(record.requested_model_identifier)
    output_hash = hashlib.sha256((record.model_output or "").encode()).hexdigest()
    successful_attempts = [item for item in record.attempts if item.status == "success"]
    executed_name_matches = bool(
        model
        and record.executed_model_identifier
        and record.executed_model_identifier in {model.model_id, model.model_id.split("/", 1)[-1]}
    )
    return bool(
        record.execution_mode == "real"
        and record.provider == "litellm"
        and model is not None
        and _verified_source(manifest)
        and record.scientific_eligible
        and record.execution_status == "success"
        and record.model_output
        and record.evaluation is not None
        and record.evaluation_complete
        and record.evaluation == evaluate_answer(record.model_output or "", sample.reference_answer)
        and record.prompt_sha256 == PROMPT_SHA256
        and record.evaluator_version == EVALUATOR_VERSION
        and record.provenance.get("dataset") == manifest.dataset
        and record.provenance.get("dataset_revision") == manifest.revision
        and record.provenance.get("source_hash") == sample.source_hash
        and record.provenance.get("sample_is_synthetic") is False
        and record.provenance.get("output_sha256") == output_hash
        and record.attempts
        and executed_name_matches
        and any(
            item.model_identifier == record.executed_model_identifier and item.provider == "litellm"
            for item in successful_attempts
        )
    )


def _verified_source(manifest) -> bool:
    expected_splits = {"train", "test"}
    return bool(
        manifest.dataset == gsm8k.DATASET_NAME
        and manifest.source_organization == gsm8k.SOURCE_ORGANIZATION
        and manifest.source_url == gsm8k.SOURCE_REPOSITORY
        and manifest.revision == gsm8k.SOURCE_REVISION
        and manifest.license == gsm8k.LICENSE
        and manifest.dataset_content_sha256 == gsm8k.VERIFIED_PREPARED_CONTENT_SHA256
        and manifest.assigned_split_counts == gsm8k.VERIFIED_SPLIT_COUNTS
        and manifest.seed == 42
        and manifest.train_fraction == 0.9
        and manifest.near_duplicate_threshold == 0.9
        and set(manifest.raw_files) == expected_splits
        and all(
            manifest.raw_files[split].source_url == gsm8k.raw_file_url(split)
            and manifest.raw_files[split].sha256 == gsm8k.VERIFIED_RAW_FILE_SHA256[split]
            and manifest.raw_files[split].size_bytes == gsm8k.VERIFIED_RAW_FILE_SIZES[split]
            for split in expected_splits
        )
    )


def export_training_data(
    records, dataset_records, dataset_manifest, output_dir: Path, collection_plan=None
) -> dict:
    if sha256_records(dataset_records) != dataset_manifest.dataset_content_sha256:
        raise ValueError("loaded dataset records do not match the source manifest checksum")
    run_ids = {record.run_id for record in records}
    if len(run_ids) > 1:
        raise ValueError("training export accepts one frozen source run at a time")
    plan_sha256s = {
        record.provenance.get("collection_plan_sha256")
        for record in records
        if record.provenance.get("collection_plan_sha256") is not None
    }
    planned_records = any(
        record.provenance.get("selection_policy") == "versioned_collection_plan_v1"
        for record in records
    )
    if planned_records and collection_plan is None:
        raise ValueError("planned run records require the source collection plan for export")
    plan_data = None
    if collection_plan is not None:
        if isinstance(collection_plan, Path):
            plan_data = json.loads(collection_plan.read_text("utf-8"))
        else:
            plan_data = collection_plan
        planned_rows = validate_collection_plan(plan_data, dataset_records, dataset_manifest)
        if not planned_records or plan_sha256s != {plan_data.get("plan_sha256")}:
            raise ValueError("source collection plan does not match run record provenance")
        allowed_ids = {row.sample_id for row in planned_rows}
        if any(record.sample_id not in allowed_ids for record in records):
            raise ValueError("run record sample ID is outside its source collection plan")
    by_id = {row.sample_id: row for row in dataset_records}
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, list[dict]] = {"train": [], "validation": [], "unknown": []}
    reason_counts: Counter = Counter()
    excluded_test = 0
    seen = set()
    for record in records:
        if record.model_tier != "small":
            continue
        if record.split == "test":
            excluded_test += 1
            continue
        key = (record.run_id, record.sample_id, record.model_tier)
        if key in seen:
            raise ValueError(f"duplicate source execution: {key}")
        seen.add(key)
        sample = by_id.get(record.sample_id)
        if sample is None or sample.assigned_split != record.split:
            reason_counts["dataset_record_missing_or_split_mismatch"] += 1
            continue
        if not _valid_real_output(record, sample, dataset_manifest):
            reason_counts["unverified_or_ineligible_execution"] += 1
            continue
        target = "train" if record.split == "train" else "validation"
        correct = record.evaluation.is_correct
        status = "determinate" if correct is not None else "unknown"
        row = {
            "sample_id": record.sample_id,
            "assigned_split": target,
            "question": sample.question,
            "query_features": extract_features(sample.question, sample.task_type).__dict__,
            "small_model_sufficient": int(correct) if correct is not None else None,
            "label_status": status,
            "label_definition": LABEL_DEFINITION,
            "small_model_identifier": record.requested_model_identifier,
            "source_run_id": record.run_id,
            "source_evaluator_version": record.evaluator_version,
            "source_collection_plan_sha256": record.provenance.get("collection_plan_sha256"),
        }
        # No outputs, reference answers, usage, cost, or outcome metrics enter query_features.
        files[target if correct is not None else "unknown"].append(row)
    for filename, rows in files.items():
        (output_dir / f"{filename}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
    manifest = {
        "schema_version": "phase3a-router-labels-v1",
        "label_definition": LABEL_DEFINITION,
        "dataset": dataset_manifest.dataset,
        "dataset_revision": dataset_manifest.revision,
        "dataset_content_sha256": dataset_manifest.dataset_content_sha256,
        "collection_plan_schema_version": plan_data.get("schema_version") if plan_data else None,
        "collection_plan_sha256": plan_data.get("plan_sha256") if plan_data else None,
        "collection_plan_sample_ids_sha256": (
            plan_data.get("sample_ids_sha256") if plan_data else None
        ),
        "source_run_ids": sorted({record.run_id for record in records}),
        "evaluator_version": EVALUATOR_VERSION,
        "excluded_official_test_records": excluded_test,
        "counts": {name: len(rows) for name, rows in files.items()},
        "excluded_records": dict(reason_counts),
        "router_inputs_exclude_model_outputs_and_labels": True,
        "pedagogical_sufficiency_established": False,
        "source_manifest_verified": _verified_source(dataset_manifest),
        "scientific_training_data_available": bool(files["train"])
        and _verified_source(dataset_manifest),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/gsm8k"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/gsm8k.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training_data"))
    parser.add_argument("--collection-plan", type=Path)
    args = parser.parse_args(argv)
    manifest = DatasetManifest.model_validate_json(args.manifest.read_text("utf-8"))
    dataset = load_processed_dataset(args.dataset_dir, manifest)
    result = export_training_data(
        load_records(args.records), dataset, manifest, args.output_dir, args.collection_plan
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
