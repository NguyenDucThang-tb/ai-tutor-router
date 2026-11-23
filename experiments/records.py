from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.gateway import CostInfo, ExecutionAttempt, UsageInfo
from evaluation.schemas import EvaluationResult


class ExecutionMetrics(BaseModel):
    feature_extraction_ms: float | None = None
    router_decision_ms: float | None = None
    scheduling_queue_ms: float | None = None
    model_only_latency_ms: float | None = None
    fallback_latency_ms: float | None = None
    gateway_elapsed_ms: float | None = None
    evaluator_latency_ms: float | None = None
    pipeline_end_to_end_ms: float | None = None
    pipeline_elapsed_to_terminal_ms: float | None = None
    retry_delay_ms: float | None = None
    retry_count: int = 0
    fallback_count: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    calculated_cost_usd: float | None = None
    known_calculated_cost_usd: float | None = None
    unknown_cost_attempts: int = 0


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    sample_id: str
    split: Literal["train", "validation", "test"]
    model_tier: Literal["small", "strong"]
    requested_model_identifier: str
    executed_model_identifier: str | None = None
    selected_tier: str | None = None
    router_selected_tier: str | None = None
    prompt_version: str
    prompt_sha256: str
    evaluator_version: str = "numeric-symbolic-v2"
    provider: str
    fallback_policy: str = "disabled"
    execution_mode: Literal["mock", "real", "replay"]
    execution_status: Literal[
        "not_started", "success", "provider_failure", "interrupted", "outcome_unknown"
    ]
    model_output: str | None = None
    evaluation: EvaluationResult | None = None
    evaluation_complete: bool = False
    scientific_eligible: bool = False
    metrics: ExecutionMetrics = Field(default_factory=ExecutionMetrics)
    usage: UsageInfo | None = None
    cost: CostInfo = Field(default_factory=CostInfo)
    final_attempt_cost: CostInfo | None = None
    attempts: list[ExecutionAttempt] = Field(default_factory=list)
    task_type: str
    difficulty: str | None = None
    provenance: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at_utc: str
    git_commit_sha: str
    execution_mode: Literal["mock", "real", "replay"]
    dataset: str
    dataset_source_url: str
    dataset_source_organization: str
    dataset_license: str
    dataset_revision: str
    dataset_source_file_sha256: dict[str, str]
    dataset_content_sha256: str
    source_dataset_content_sha256: str | None = None
    sample_source_hashes: dict[str, str]
    assigned_split: str
    sample_count: int
    sample_ids_sha256: str
    collection_plan_schema_version: str | None = None
    collection_plan_sha256: str | None = None
    collection_plan_sample_ids: list[str] = Field(default_factory=list)
    collection_plan_sample_ids_sha256: str | None = None
    collection_plan_split: str | None = None
    selection_policy: str = "legacy_sorted_prefix_v1"
    small_model_identifier: str
    strong_model_identifier: str
    provider: str
    provider_region: str | None
    prompt_version: str
    prompt_sha256: str
    evaluator_version: str
    temperature: float
    max_tokens: int
    random_seed: int
    concurrency: int
    timeout_seconds: float
    max_requests: int
    max_retries: int
    retry_delay_seconds: float
    fallback_policy: str
    small_fallback_identifier: str | None
    strong_fallback_identifier: str | None
    budget_limit_usd: float | None
    max_run_budget_usd: float | None = None
    pricing_source: str | None
    pricing_effective_date: str | None
    experiment_approval_id: str | None
    approval_artifact_sha256: str | None = None
    approval_config_sha256: str | None = None
    # Read-only compatibility for manifests produced before recovery controls
    # moved into the SQLite recovery authorization audit.
    allow_paid_retries: bool = Field(default=False, exclude=True)
    paid_retry_approval_id: str | None = Field(default=None, exclude=True)
    software_versions: dict[str, str]
    scientific_protocol_eligible: bool
    scientific_eligibility_reasons: list[str]
