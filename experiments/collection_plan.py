"""Reproducible, pre-execution stratified sample selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from app.features import extract_features
from datasets.schemas import DatasetManifest
from datasets.validation import load_processed_dataset


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
    return {
        "schema_version": "phase3a-collection-plan-v1",
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
    plan["dataset_content_sha256"] = manifest.dataset_content_sha256
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected_count": plan["selected_count"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
