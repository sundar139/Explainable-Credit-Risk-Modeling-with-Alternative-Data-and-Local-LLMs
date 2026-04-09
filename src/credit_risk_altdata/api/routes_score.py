"""Scoring routes for engineered feature payloads."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from credit_risk_altdata.api.dependencies import get_runtime_service
from credit_risk_altdata.api.schemas import ScoreRequest, ScoreResponse
from credit_risk_altdata.api.services import APIRuntimeService, APIServiceError

router = APIRouter(tags=["scoring"])


@router.post("/score", response_model=ScoreResponse)
def post_score(
    request: ScoreRequest,
    service: Annotated[APIRuntimeService, Depends(get_runtime_service)],
) -> ScoreResponse:
    """Score one engineered feature payload with the production model."""
    try:
        payload = service.score_payload(
            engineered_features=request.engineered_features,
            applicant_id=request.applicant_id,
            threshold=request.threshold,
            include_prediction_label=request.include_prediction_label,
        )
    except APIServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc

    return ScoreResponse(**payload)
