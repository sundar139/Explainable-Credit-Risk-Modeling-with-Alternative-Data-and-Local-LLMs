"""Artifact-summary routes for demo visibility."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from credit_risk_altdata.api.dependencies import get_runtime_service
from credit_risk_altdata.api.schemas import ArtifactSummaryResponse
from credit_risk_altdata.api.services import APIRuntimeService

router = APIRouter(tags=["artifacts"])


@router.get("/artifacts/summary", response_model=ArtifactSummaryResponse)
def get_artifacts_summary(
    service: Annotated[APIRuntimeService, Depends(get_runtime_service)],
) -> ArtifactSummaryResponse:
    """Return machine-readable summary of key model/explainability/report artifacts."""
    return ArtifactSummaryResponse(**service.artifacts_summary_payload())
