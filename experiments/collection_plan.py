"""Reproducible, pre-execution stratified sample selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from app.features import extract_features
from datasets.provenance import sha256_records
from datasets.schemas import DatasetManifest
from datasets.validation import load_processed_dataset

PLAN_SCHEMA_VERSION = "phase3a-collection-plan-v1"


def plan_hash(plan: Mapping) -> str:
    material = {key: value for key, value in plan.items() if key != "plan_sha256"}
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def seal_plan(plan: dict) -> dict:
    """Return a plan with its canonical integrity hash populated."""
    sealed = dict(plan)
    sealed["plan_sha256"] = plan_hash(sealed)
    return sealed


def validate_collection_plan(
    plan: Mapping,
    records,
    manifest: DatasetManifest,
    *,
    expected_split: str | None = None,
    expected_count: int | None = None,
):
    """Validate a frozen plan against the complete loaded dataset and return rows in plan order."""
    if not isinstance(plan, Mapping) or plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported collection plan schema version")
    if plan.get("plan_sha256") != plan_hash(plan):
        raise ValueError("collection plan hash mismatch")
    if manifest.dataset_content_sha256 != sha256_records(records):
        raise ValueError("loaded dataset records do not match the source manifest checksum")
    if plan.get("dataset") != manifest.dataset or plan.get("dataset_revision") != manifest.revision:
        raise ValueError("collection plan dataset revision does not match the source manifest")
    if plan.get("dataset_content_sha256") != manifest.dataset_content_sha256:
        raise ValueError("collection plan dataset checksum does not match the source manifest")
    split = plan.get("split")
    if split not in {"train", "validation"} or (expected_split and split != expected_split):
        raise ValueError("collection plan split is invalid or does not match requested split")
    sample_ids = plan.get("sample_ids")
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or any(not isinstance(value, str) or not value for value in sample_ids)
    ):
        raise ValueError("collection plan sample_ids must be a non-empty string list")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("collection plan contains duplicate sample IDs")
    if sample_ids != sorted(sample_ids):
        raise ValueError("collection plan sample IDs are not in deterministic sorted order")
    if plan.get("selected_count") != len(sample_ids):
        raise ValueError("collection plan sample count does not match selected IDs")
    if expected_count is not None and expected_count != len(sample_ids):
        raise ValueError("configured sample count must equal the complete collection plan")
    ids_hash = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
    if plan.get("sample_ids_sha256") != ids_hash:
        raise ValueError("collection plan sample ID hash mismatch")
    by_id = {}
    for row in records:
        if row.sample_id in by_id:
            raise ValueError(f"dataset contains duplicate sample ID: {row.sample_id}")
        by_id[row.sample_id] = row
        if row.dataset != manifest.dataset or row.dataset_revision != manifest.revision:
            raise ValueError("dataset record provenance does not match the source manifest")
    rows = []
    for sample_id in sample_ids:
        row = by_id.get(sample_id)
        if row is None:
            raise ValueError(f"collection plan sample ID is missing: {sample_id}")
        if row.assigned_split == "test":
            raise ValueError("collection plan cannot include official test sample IDs")
        if row.assigned_split != split:
            raise ValueError(f"collection plan sample is outside planned split: {sample_id}")
        rows.append(row)
    if plan.get("candidate_count") != sum(row.assigned_split == split for row in records):
        raise ValueError("collection plan candidate count does not match the selected split")
    return rows


def create_plan(records, split: str, samples: int, seed: int = 42) -> dict:
    if split not in {"train", "validation"}:
        raise ValueError("Phase 3A collection plan only permits train or validation")
    candidates = [row for row in records if row.assigned_split == split]
    if samples < 1 or samples > len(candidates):
        raise ValueError("requested sample count must fit the selected split")
    enriched = [(row, extract_features(row.question, row.task_type)) for row in candidates]
    lengths = sorted(feature.char_count for _, feature in enriched)
    boundaries = [lengths[min(len(lengths) - 1, (len(lengths) * q) // 4)] for q in (1, 2, 3)]
    symbol_counts = sorted(feature.math_symbol_count for _, feature in enriched)
    symbol_boundaries = [
        symbol_counts[min(len(symbol_counts) - 1, (len(symbol_counts) * q) // 4)] for q in (1, 2, 3)
    ]
    buckets: dict[str, list] = {}
    for row, feature in enriched:
        quartile = sum(feature.char_count > boundary for boundary in boundaries)
        symbol_quartile = sum(
            feature.math_symbol_count > boundary for boundary in symbol_boundaries
        )
        key = f"{row.task_type}:length_q{quartile + 1}:symbols_q{symbol_quartile + 1}"
        buckets.setdefault(key, []).append(row)
    for key, bucket in buckets.items():
        bucket.sort(key=lambda row: hashlib.sha256(f"{seed}:{row.sample_id}".encode()).hexdigest())
    selected = []
    # Round robin over sorted strata prevents dominant strata from taking the sample.
    strata = sorted(buckets)
    while len(selected) < samples:
        progressed = False
        for key in strata:
            if buckets[key]:
                selected.append((key, buckets[key].pop(0)))
                progressed = True
                if len(selected) == samples:
                    break
        if not progressed:
            break
    chosen = sorted((row for _, row in selected), key=lambda row: row.sample_id)
    distribution = Counter(key for key, _ in selected)
    return seal_plan(
        {
            "schema_version": PLAN_SCHEMA_VERSION,
            "dataset": records[0].dataset,
            "dataset_revision": records[0].dataset_revision,
            "dataset_content_sha256": sha256_records(records),
            "split": split,
            "seed": seed,
            "stratification": "task_type x char_count quartile x math_symbol_count quartile; symbols are a query-derived complexity proxy, not a difficulty label; SHA-256 seeded within-stratum order; round-robin allocation",
            "candidate_count": len(candidates),
            "selected_count": len(chosen),
            "stratum_counts": dict(sorted(distribution.items())),
            "sample_ids": [row.sample_id for row in chosen],
            "sample_ids_sha256": hashlib.sha256(
                "\n".join(row.sample_id for row in chosen).encode()
            ).hexdigest(),
            "features_are_pre_execution": True,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/gsm8k"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/gsm8k.json"))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("collection_plan.json"))
    args = parser.parse_args(argv)
    manifest = DatasetManifest.model_validate_json(args.manifest.read_text("utf-8"))
    records = load_processed_dataset(args.dataset_dir, manifest)
    plan = create_plan(records, args.split, args.samples, args.seed)
    # The complete source checksum is part of the hashed plan identity.
    plan["dataset_content_sha256"] = manifest.dataset_content_sha256
    plan = seal_plan(plan)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected_count": plan["selected_count"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
