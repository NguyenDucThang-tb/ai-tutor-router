from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from datasets.gsm8k import download_gsm8k, load_gsm8k
from datasets.schemas import DatasetManifest
from datasets.splits import split_training_records
from datasets.validation import load_processed_dataset, validate_records, write_prepared_dataset


def prepare(
    dataset: str,
    raw_directory: Path,
    output_directory: Path,
    manifest_path: Path,
    seed: int = 42,
    train_fraction: float = 0.9,
    near_duplicate_threshold: float = 0.9,
) -> dict:
    if dataset != "gsm8k":
        raise ValueError(
            f"Dataset '{dataset}' is not available; MATH is deferred pending source access review"
        )
    source_manifest = download_gsm8k(raw_directory)
    records = load_gsm8k(raw_directory, source_manifest)
    assigned, training_pairs = split_training_records(
        records,
        seed=seed,
        train_fraction=train_fraction,
        threshold=near_duplicate_threshold,
    )
    manifest, report, duplicate_pairs = write_prepared_dataset(
        assigned,
        source_manifest,
        output_directory,
        manifest_path,
        seed,
        train_fraction,
        near_duplicate_threshold,
        training_pairs,
    )
    return {
        "manifest": str(manifest_path),
        "dataset": manifest.dataset,
        "revision": manifest.revision,
        "sample_count": report.sample_count,
        "split_counts": report.split_counts,
        "duplicate_pair_count": len(duplicate_pairs),
        "cross_split_overlap_count": report.cross_split_overlap_count,
        "dataset_content_sha256": manifest.dataset_content_sha256,
    }


def validate(dataset: str, output_directory: Path, manifest_path: Path) -> dict:
    if dataset != "gsm8k":
        raise ValueError(
            f"Dataset '{dataset}' is not available; MATH is deferred pending source access review"
        )
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text("utf-8"))
    records = load_processed_dataset(output_directory, manifest)
    report, pairs = validate_records(records, manifest.near_duplicate_threshold)
    return {
        **report.model_dump(mode="json"),
        "duplicate_pairs": [pair.model_dump(mode="json") for pair in pairs],
        "manifest_valid": not report.errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and validate research datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "validate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--dataset", required=True, choices=("gsm8k",))
        sub.add_argument("--raw-dir", type=Path, default=Path("data/raw/gsm8k"))
        sub.add_argument("--processed-dir", type=Path, default=Path("data/processed/gsm8k"))
        sub.add_argument("--manifest", type=Path, default=Path("data/manifests/gsm8k.json"))
        if command == "prepare":
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--train-fraction", type=float, default=0.9)
            sub.add_argument("--near-duplicate-threshold", type=float, default=0.9)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(
                args.dataset,
                args.raw_dir,
                args.processed_dir,
                args.manifest,
                args.seed,
                args.train_fraction,
                args.near_duplicate_threshold,
            )
        else:
            result = validate(args.dataset, args.processed_dir, args.manifest)
    except (OSError, ValueError, TypeError, ValidationError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("manifest_valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
