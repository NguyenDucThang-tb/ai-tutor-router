from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from datasets.provenance import sha256_bytes, sha256_records
from datasets.schemas import DatasetManifest, DatasetRecord, DatasetValidationReport, DuplicatePair
from datasets.splits import find_duplicate_pairs


def validate_records(
    records: list[DatasetRecord], near_duplicate_threshold: float = 0.9
) -> tuple[DatasetValidationReport, list[DuplicatePair]]:
    errors: list[str] = []
    ids = [record.sample_id for record in records]
    invalid_ids = sorted(sample_id for sample_id, count in Counter(ids).items() if count > 1)
    if invalid_ids:
        errors.append("sample_id values are not unique")
    for record in records:
        if record.original_split == "test" and record.assigned_split != "test":
            errors.append(f"official test record moved: {record.sample_id}")
        if record.original_split == "train" and record.assigned_split == "test":
            errors.append(f"training record assigned to official test: {record.sample_id}")
    pairs = find_duplicate_pairs(records, threshold=near_duplicate_threshold)
    train_validation_overlaps = [
        pair for pair in pairs if {pair.left_split, pair.right_split} == {"train", "validation"}
    ]
    if train_validation_overlaps:
        errors.append(f"{len(train_validation_overlaps)} near/exact train-validation overlaps")
    counts = Counter(record.assigned_split for record in records)
    report = DatasetValidationReport(
        dataset=records[0].dataset if records else "unknown",
        sample_count=len(records),
        split_counts={split: counts.get(split, 0) for split in ("train", "validation", "test")},
        duplicate_pair_count=len(pairs),
        cross_split_overlap_count=sum(pair.cross_split for pair in pairs),
        invalid_sample_ids=invalid_ids,
        errors=errors,
    )
    return report, pairs


def write_prepared_dataset(
    records: list[DatasetRecord],
    source_manifest,
    output_directory: Path,
    manifest_path: Path,
    seed: int,
    train_fraction: float,
    near_duplicate_threshold: float,
    training_duplicate_pairs: list[DuplicatePair],
) -> tuple[DatasetManifest, DatasetValidationReport, list[DuplicatePair]]:
    report, duplicate_pairs = validate_records(records, near_duplicate_threshold)
    if report.errors:
        raise ValueError("Prepared dataset validation failed: " + "; ".join(report.errors))
    output_directory.mkdir(parents=True, exist_ok=True)
    file_hashes: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        selected = sorted(
            (record for record in records if record.assigned_split == split),
            key=lambda record: record.sample_id,
        )
        content = b"".join((record.model_dump_json() + "\n").encode("utf-8") for record in selected)
        target = output_directory / f"{split}.jsonl"
        atomic_write(target, content)
        file_hashes[split] = sha256_bytes(content)

    transformation_history = [
        "Read official train.jsonl/test.jsonl from the pinned upstream revision and hashed each source line.",
        "Extracted the unique final answer on the line marked ####; retained preceding worked solution separately.",
        f"Assigned whole duplicate-connected components from official train only using seed {seed}; validation target is {1 - train_fraction:.6f} of training records.",
        f"Exact duplicate normalization: Unicode word tokens, case-folded, whitespace joined. Near duplicate: Jaccard similarity of sets of 3-token word shingles >= {near_duplicate_threshold:.6f}; short questions use token sets.",
        "Official test records remain in the official test split; overlaps are reported and not removed or moved.",
    ]
    manifest = DatasetManifest(
        dataset=source_manifest.dataset,
        source_organization=source_manifest.source_organization,
        source_url=source_manifest.source_url,
        revision=source_manifest.revision,
        license=source_manifest.license,
        access_note=source_manifest.access_note,
        downloaded_at_utc=source_manifest.downloaded_at_utc,
        raw_files=source_manifest.raw_files,
        transformation_history=transformation_history,
        seed=seed,
        train_fraction=train_fraction,
        near_duplicate_threshold=near_duplicate_threshold,
        assigned_split_counts=report.split_counts,
        dataset_content_sha256=sha256_records(records),
        processed_file_sha256=file_hashes,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(manifest_path, manifest.model_dump_json(indent=2).encode("utf-8"))
    atomic_write(
        output_directory / "duplicates.json",
        json.dumps(
            {
                "threshold": near_duplicate_threshold,
                "pairs": [pair.model_dump(mode="json") for pair in duplicate_pairs],
                "training_pairs_grouped_before_split": [
                    pair.model_dump(mode="json") for pair in training_duplicate_pairs
                ],
                "official_test_overlaps_preserved": True,
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8"),
    )
    return manifest, report, duplicate_pairs


def load_processed_dataset(
    output_directory: Path, manifest: DatasetManifest
) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for split in ("train", "validation", "test"):
        path = output_directory / f"{split}.jsonl"
        content = path.read_bytes()
        if sha256_bytes(content) != manifest.processed_file_sha256.get(split):
            raise ValueError(f"Processed {split} file does not match manifest checksum")
        for line_number, line in enumerate(content.splitlines(), start=1):
            try:
                record = DatasetRecord.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"Invalid record in {path.name}:{line_number}") from exc
            if record.assigned_split != split:
                raise ValueError(f"Record in {path.name} claims split {record.assigned_split}")
            records.append(record)
    if sha256_records(records) != manifest.dataset_content_sha256:
        raise ValueError("Processed records do not match dataset content checksum")
    return records


def atomic_write(path: Path, content: bytes) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
