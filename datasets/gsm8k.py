from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from datasets.provenance import sha256_bytes
from datasets.schemas import DatasetRecord, DatasetSourceManifest, SourceFileManifest
from evaluation.answer_extraction import extract_gsm8k_reference

DATASET_NAME = "gsm8k"
SOURCE_ORGANIZATION = "OpenAI"
SOURCE_REPOSITORY = "https://github.com/openai/grade-school-math"
SOURCE_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
LICENSE = "MIT (upstream repository license; copyright OpenAI 2021)"
VERIFIED_RAW_FILE_SHA256 = {
    "train": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
    "test": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
}
VERIFIED_RAW_FILE_SIZES = {"train": 4166206, "test": 749738}
VERIFIED_PREPARED_CONTENT_SHA256 = (
    "aa75c686332c09e12ce6412571b32b7efae779dfc8696d3448b29a45e5542e4b"
)
VERIFIED_SPLIT_COUNTS = {"train": 6726, "validation": 747, "test": 1319}
SPLITS = ("train", "test")


def raw_file_url(split: str, revision: str = SOURCE_REVISION) -> str:
    if split not in SPLITS:
        raise ValueError(f"Unsupported GSM8K split: {split}")
    return (
        f"https://raw.githubusercontent.com/openai/grade-school-math/{revision}/"
        f"grade_school_math/data/{split}.jsonl"
    )


def download_gsm8k(raw_directory: Path, timeout_seconds: float = 60) -> DatasetSourceManifest:
    """Download the two official JSONL splits at an immutable upstream commit."""
    raw_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_directory / "source_manifest.json"
    if manifest_path.is_file():
        manifest = DatasetSourceManifest.model_validate_json(manifest_path.read_text("utf-8"))
        if manifest.revision != SOURCE_REVISION:
            raise ValueError("Existing raw GSM8K manifest uses a different revision")
        _verify_raw_files(raw_directory, manifest)
        return manifest

    files: dict[str, SourceFileManifest] = {}
    downloaded_at = datetime.now(UTC)
    for split in SPLITS:
        url = raw_file_url(split)
        request = Request(url, headers={"User-Agent": "ai-tutor-router-phase2/1.0"})
        with urlopen(request, timeout=timeout_seconds) as response:
            content = response.read()
        path = raw_directory / f"{split}.jsonl"
        _atomic_write(path, content)
        files[split] = SourceFileManifest(
            original_split=split,
            source_url=url,
            sha256=sha256_bytes(content),
            size_bytes=len(content),
            downloaded_at_utc=downloaded_at,
        )

    manifest = DatasetSourceManifest(
        dataset=DATASET_NAME,
        source_organization=SOURCE_ORGANIZATION,
        source_url=SOURCE_REPOSITORY,
        revision=SOURCE_REVISION,
        license=LICENSE,
        access_note="Official archived repository; raw train/test JSONL served at the pinned Git commit.",
        downloaded_at_utc=downloaded_at,
        raw_files=files,
    )
    _atomic_write(manifest_path, manifest.model_dump_json(indent=2).encode("utf-8"))
    return manifest


def load_gsm8k(raw_directory: Path, manifest: DatasetSourceManifest) -> list[DatasetRecord]:
    _verify_raw_files(raw_directory, manifest)
    records: list[DatasetRecord] = []
    for split in SPLITS:
        path = raw_directory / f"{split}.jsonl"
        with path.open("rb") as source:
            for index, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path.name}:{index}") from exc
                if not isinstance(row, dict) or not isinstance(row.get("question"), str):
                    raise TypeError(f"Missing GSM8K question in {path.name}:{index}")
                raw_answer = row.get("answer")
                if not isinstance(raw_answer, str):
                    raise TypeError(f"Missing GSM8K answer in {path.name}:{index}")
                reference_answer, reference_solution = extract_gsm8k_reference(raw_answer)
                records.append(
                    DatasetRecord(
                        sample_id=f"gsm8k-{split}-{index:06d}",
                        dataset=DATASET_NAME,
                        dataset_revision=manifest.revision,
                        original_split=split,
                        assigned_split="test" if split == "test" else "train",
                        question=row["question"].strip(),
                        reference_answer=reference_answer,
                        reference_solution=reference_solution,
                        task_type="word_problem",
                        language="en",
                        source_id=f"{split}:{index}",
                        source_hash=sha256_bytes(raw_line.rstrip(b"\r\n")),
                        source_metadata={"answer_format": "#### final answer"},
                        inferred_metadata={
                            "task_type": "word_problem",
                            "language": "en",
                            "difficulty": None,
                        },
                    )
                )
    return records


def _verify_raw_files(raw_directory: Path, manifest: DatasetSourceManifest) -> None:
    if manifest.dataset != DATASET_NAME or manifest.revision != SOURCE_REVISION:
        raise ValueError("GSM8K source manifest does not match the pinned dataset revision")
    for split in SPLITS:
        entry = manifest.raw_files.get(split)
        path = raw_directory / f"{split}.jsonl"
        if entry is None or not path.is_file():
            raise FileNotFoundError(f"Missing GSM8K {split} file or provenance record")
        content = path.read_bytes()
        if len(content) != entry.size_bytes or sha256_bytes(content) != entry.sha256:
            raise ValueError(f"GSM8K {split} file does not match its recorded checksum")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
