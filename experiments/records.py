from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.gateway import CostInfo, ExecutionAttempt, UsageInfo
from evaluation.schemas import EvaluationResult


class ExecutionMetrics(BaseModel):
    feature_extraction_ms: float | None = None
    router_decision_ms: float | None = None
    gateway_latency_ms: float | None = None
    evaluator_latency_ms: float | None = None
    end_to_end_latency_ms: float | None = None
    retry_delay_ms: float | None = None
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
    evaluator_version: str = "numeric-symbolic-v1"
    provider: str
    execution_mode: Literal["mock", "real", "replay"]
    execution_status: Literal["success", "provider_failure", "interrupted"]
    model_output: str | None = None
    evaluation: EvaluationResult | None = None
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
    dataset_revision: str
    dataset_content_sha256: str
    assigned_split: str
    sample_count: int
    sample_ids_sha256: str
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
    pricing_source: str | None
    pricing_effective_date: str | None
    experiment_approval_id: str | None
    software_versions: dict[str, str]
    scientific_eligible: bool
