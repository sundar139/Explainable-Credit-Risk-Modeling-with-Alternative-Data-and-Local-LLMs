"""Explainability retrieval and generation routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from credit_risk_altdata.api.dependencies import get_runtime_service
from credit_risk_altdata.api.schemas import ExplainRequest, ExplainResponse
from credit_risk_altdata.api.services import APIRuntimeService, APIServiceError

router = APIRouter(tags=["explainability"])


@router.post("/explain", response_model=ExplainResponse)
def post_explain(
    request: ExplainRequest,
    service: Annotated[APIRuntimeService, Depends(get_runtime_service)],
) -> ExplainResponse:
    """Retrieve or generate explainability payloads within artifact-backed constraints."""
    try:
        payload = service.explain_payload(
            applicant_id=request.applicant_id,
            explanation_method=request.explanation_method,
            allow_generate_if_missing=request.allow_generate_if_missing,
            top_k=request.top_k,
            engineered_features=request.engineered_features,
        )
    except APIServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc

    return ExplainResponse(**payload)
