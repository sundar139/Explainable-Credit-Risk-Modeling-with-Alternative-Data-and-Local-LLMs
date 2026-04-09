"""FastAPI application entry point."""

from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from credit_risk_altdata.api.routes_artifacts import router as artifacts_router
from credit_risk_altdata.api.routes_explain import router as explain_router
from credit_risk_altdata.api.routes_health import router as health_router
from credit_risk_altdata.api.routes_reports import router as reports_router
from credit_risk_altdata.api.routes_score import router as score_router
from credit_risk_altdata.config import Settings, get_settings
from credit_risk_altdata.logging import configure_logging


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: Any | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details,
            }
        },
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create configured FastAPI application."""
    configure_logging()
    resolved_settings = settings if settings is not None else get_settings()

    api_app = FastAPI(
        title=resolved_settings.app_name,
        version="0.1.0",
        description=(
            "Phase 8 API for health/readiness, scoring, explainability retrieval/generation, "
            "and local narrative report retrieval/generation"
        ),
    )

    api_app.include_router(health_router)
    api_app.include_router(score_router)
    api_app.include_router(explain_router)
    api_app.include_router(reports_router)
    api_app.include_router(artifacts_router)

    @api_app.exception_handler(HTTPException)
    async def http_exception_handler(_: Any, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and {"code", "message"}.issubset(detail.keys()):
            return _error_response(
                status_code=exc.status_code,
                code=str(detail["code"]),
                message=str(detail["message"]),
                details=detail.get("details"),
            )
        return _error_response(
            status_code=exc.status_code,
            code="http_error",
            message=str(detail),
            details=None,
        )

    @api_app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Any, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            status_code=422,
            code="request_validation_error",
            message="Request payload validation failed",
            details=exc.errors(),
        )

    @api_app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Any, exc: Exception) -> JSONResponse:
        return _error_response(
            status_code=500,
            code="internal_server_error",
            message="Unhandled server error",
            details={"reason": str(exc)},
        )

    return api_app


app = create_app()


def run() -> None:
    """Run API server with production-friendly defaults."""
    uvicorn.run(
        "credit_risk_altdata.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
