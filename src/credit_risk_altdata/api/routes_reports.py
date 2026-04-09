"""Narrative risk-report retrieval and generation routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from credit_risk_altdata.api.dependencies import get_runtime_service
from credit_risk_altdata.api.schemas import RiskReportRequest, RiskReportResponse
from credit_risk_altdata.api.services import APIRuntimeService, APIServiceError

router = APIRouter(tags=["risk-reports"])


@router.post("/risk-report", response_model=RiskReportResponse)
def post_risk_report(
    request: RiskReportRequest,
    service: Annotated[APIRuntimeService, Depends(get_runtime_service)],
) -> RiskReportResponse:
    """Retrieve or generate local narrative risk reports from explanation artifacts."""
    try:
        payload = service.risk_report_payload(
            applicant_id=request.applicant_id,
            explanation_method_source=request.explanation_method_source,
            report_type=request.report_type,
            allow_generate_if_missing=request.allow_generate_if_missing,
            allow_fallback=request.allow_fallback,
            top_k=request.top_k,
        )
    except APIServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc

    return RiskReportResponse(**payload)
