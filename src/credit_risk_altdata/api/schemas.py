"""Pydantic request and response schemas for Phase 8 API endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class APIErrorDetail(BaseModel):
    """Structured API error details."""

    code: str
    message: str
    details: dict[str, Any] | list[Any] | None = None


class APIErrorResponse(BaseModel):
    """Structured API error response envelope."""

    error: APIErrorDetail


class HealthResponse(BaseModel):
    """Service liveness payload."""

    status: Literal["ok"]
    environment: Literal["dev", "test", "prod"]
    service: str


class ReadinessComponentStatus(BaseModel):
    """Per-component readiness status payload."""

    name: str
    ready: bool
    optional: bool = False
    path: str | None = None
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """Structured readiness payload."""

    status: Literal["ready", "degraded", "not_ready"]
    ready: bool
    components: list[ReadinessComponentStatus]


class ScoreRequest(BaseModel):
    """Request payload for single-row scoring."""

    model_config = ConfigDict(extra="forbid")

    engineered_features: dict[str, float | int | bool | None] = Field(min_length=1)
    applicant_id: int | str | None = None
    threshold: float | None = Field(default=None, gt=0.0, lt=1.0)
    include_prediction_label: bool = True


class ScoreResponse(BaseModel):
    """Scoring response payload."""

    applicant_id: int | str | None
    predicted_probability: float
    predicted_label: int | None
    threshold: float
    final_model_family: str
    final_candidate_name: str
    model_artifact_path: str
    warnings: list[str] = []


class ExplainRequest(BaseModel):
    """Request payload for explanation retrieval/generation."""

    model_config = ConfigDict(extra="forbid")

    applicant_id: int | str | None = None
    engineered_features: dict[str, float | int | bool | None] | None = None
    explanation_method: Literal["shap", "lime", "auto"] = "auto"
    top_k: int | None = Field(default=None, ge=1, le=100)
    allow_generate_if_missing: bool = False


class ExplainResponse(BaseModel):
    """Explanation response payload."""

    applicant_id: int | str | None
    explanation_method_requested: Literal["shap", "lime", "auto"]
    explanation_method_resolved: Literal["shap", "lime"] | None
    explanation_available: bool
    explanation_generated: bool
    payload: dict[str, Any] | None = None
    payload_path: str | None = None
    cohort_name: str | None = None
    warnings: list[str] = []
    errors: list[str] = []


class RiskReportRequest(BaseModel):
    """Request payload for narrative report retrieval/generation."""

    model_config = ConfigDict(extra="forbid")

    applicant_id: int | str | None = None
    explanation_method_source: Literal["shap", "lime", "auto"] = "auto"
    report_type: Literal["plain", "underwriter", "adverse-action", "all"] = "all"
    allow_generate_if_missing: bool = False
    allow_fallback: bool = True
    top_k: int | None = Field(default=None, ge=1, le=100)


class RiskReportRecord(BaseModel):
    """Single narrative report record for an applicant."""

    report_id: str
    report_type: Literal["plain", "underwriter", "adverse-action"]
    explanation_method_source: Literal["shap", "lime"] | str
    generated_text: str
    fallback_generated: bool
    failure_reason: str | None = None
    generation_timestamp: str


class RiskReportResponse(BaseModel):
    """Narrative report response payload."""

    applicant_id: int | str | None
    requested_report_type: Literal["plain", "underwriter", "adverse-action", "all"]
    report_available: bool
    report_generated: bool
    fallback_generated: bool
    payload_path: str | None = None
    disclaimer: str
    reports: list[RiskReportRecord] = []
    warnings: list[str] = []
    errors: list[str] = []


class ArtifactPathStatus(BaseModel):
    """Path existence summary entry."""

    path: str
    exists: bool


class ArtifactSummaryResponse(BaseModel):
    """Artifact availability summary payload."""

    final_candidate_name: str | None
    final_model_family: str | None
    threshold: float | None
    paths: dict[str, ArtifactPathStatus]
    generated_at: str
