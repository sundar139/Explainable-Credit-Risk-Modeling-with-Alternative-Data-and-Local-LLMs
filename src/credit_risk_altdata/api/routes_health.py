"""Health and readiness routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from credit_risk_altdata.api.dependencies import get_runtime_service
from credit_risk_altdata.api.schemas import HealthResponse, ReadinessResponse
from credit_risk_altdata.api.services import APIRuntimeService

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse)
def get_health(
    service: Annotated[APIRuntimeService, Depends(get_runtime_service)],
) -> HealthResponse:
    """Return lightweight API liveness status."""
    return HealthResponse(**service.health_payload())


@router.get("/readiness", response_model=ReadinessResponse)
def get_readiness(
    service: Annotated[APIRuntimeService, Depends(get_runtime_service)],
) -> ReadinessResponse:
    """Return structured readiness status for critical components."""
    return ReadinessResponse(**service.readiness_payload())
