from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

OriginalSplit = Literal["train", "test"]
AssignedSplit = Literal["train", "validation", "test"]


class DatasetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    original_split: OriginalSplit
    assigned_split: AssignedSplit
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    reference_solution: str = ""
    task_type: str = "math"
    difficulty: str | None = None
    language: str = "en"
    source_id: str = Field(min_length=1)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    inferred_metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class SourceFileManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    original_split: OriginalSplit
    source_url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    downloaded_at_utc: datetime


class DatasetSourceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    source_organization: str
    source_url: str
    revision: str
    license: str
    access_note: str
    downloaded_at_utc: datetime
    raw_files: dict[str, SourceFileManifest]


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    source_organization: str
    source_url: str
    revision: str
    license: str
    access_note: str
    downloaded_at_utc: datetime
    raw_files: dict[str, SourceFileManifest]
    transformation_history: list[str]
    seed: int
    train_fraction: float = Field(gt=0, lt=1)
    near_duplicate_threshold: float = Field(gt=0, le=1)
    assigned_split_counts: dict[str, int]
    dataset_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    processed_file_sha256: dict[str, str]


class DuplicatePair(BaseModel):
    left_sample_id: str
    right_sample_id: str
    similarity: float = Field(ge=0, le=1)
    exact_question: bool
    cross_split: bool
    left_split: AssignedSplit
    right_split: AssignedSplit


class DatasetValidationReport(BaseModel):
    dataset: str
    sample_count: int
    split_counts: dict[str, int]
    duplicate_pair_count: int
    cross_split_overlap_count: int
    invalid_sample_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
