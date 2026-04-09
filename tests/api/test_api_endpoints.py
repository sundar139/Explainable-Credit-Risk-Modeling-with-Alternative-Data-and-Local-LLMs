"""Offline API tests for Phase 8 service endpoints."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import pandas as pd
from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from credit_risk_altdata.api.app import app
from credit_risk_altdata.api.dependencies import get_runtime_service
from credit_risk_altdata.api.model_store import ModelStore
from credit_risk_altdata.api.services import APIRuntimeService
from credit_risk_altdata.config import Settings
from credit_risk_altdata.llm.ollama_client import OllamaHealthcheckResult
from credit_risk_altdata.llm.workflow import LLMReportingWorkflowResult


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _prepare_api_artifacts(settings: Settings) -> None:
    feature_columns = ["feature_a", "feature_b", "feature_c"]

    x_train = pd.DataFrame(
        {
            "feature_a": [0.1, 0.2, 0.8, 0.9, 1.2, 1.4],
            "feature_b": [1.0, 1.1, 0.4, 0.3, 0.2, 0.1],
            "feature_c": [0.05, 0.08, 0.4, 0.55, 0.7, 0.9],
        }
    )
    y_train = [0, 0, 1, 1, 1, 1]

    model = LogisticRegression(solver="liblinear")
    model.fit(x_train, y_train)

    settings.modeling_final_model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, settings.modeling_final_model_output_path)

    final_candidate_summary = {
        "final_candidate_name": "lightgbm_tuned_none",
        "final_model_family": "lightgbm",
        "primary_metric": "roc_auc",
        "threshold": 0.5,
        "tuned": True,
        "calibrated": False,
        "selected_artifact_path": str(
            settings.modeling_final_model_output_path.relative_to(settings.project_root)
        ),
        "final_model_output_path": str(
            settings.modeling_final_model_output_path.relative_to(settings.project_root)
        ),
        "training_timestamp": "2026-01-01T00:00:00+00:00",
    }
    settings.modeling_final_candidate_summary_path.parent.mkdir(parents=True, exist_ok=True)
    settings.modeling_final_candidate_summary_path.write_text(
        json.dumps(final_candidate_summary, indent=2),
        encoding="utf-8",
    )

    settings.feature_metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(
        [
            {
                "feature_name": "SK_ID_CURR",
                "source_module": "identifier",
                "dtype": "int64",
                "null_fraction": 0.0,
                "is_target": False,
                "is_identifier": True,
            },
            {
                "feature_name": "TARGET",
                "source_module": "target",
                "dtype": "int8",
                "null_fraction": 0.0,
                "is_target": True,
                "is_identifier": False,
            },
            *[
                {
                    "feature_name": feature,
                    "source_module": "synthetic",
                    "dtype": "float64",
                    "null_fraction": 0.0,
                    "is_target": False,
                    "is_identifier": False,
                }
                for feature in feature_columns
            ],
        ]
    )
    manifest.to_csv(settings.feature_metadata_dir / "feature_manifest.csv", index=False)

    _write_jsonl(
        settings.explainability_shap_local_dir / "shap_local_explanations.jsonl",
        [
            {
                "explanation_method": "shap",
                "applicant_id": 700001,
                "cohort_name": "false_positive",
                "predicted_probability": 0.74,
                "predicted_label": 1,
                "actual_label": 0,
                "threshold": 0.5,
                "top_risk_increasing_features": [
                    {"feature_name": "feature_a", "contribution": 0.22}
                ],
                "top_risk_decreasing_features": [
                    {"feature_name": "feature_b", "contribution": -0.08}
                ],
                "explanation_generated": True,
                "failure_reason": None,
            }
        ],
    )

    _write_jsonl(
        settings.explainability_lime_dir / "lime_explanations.jsonl",
        [
            {
                "explanation_method": "lime",
                "applicant_id": 700002,
                "cohort_name": "true_negative",
                "predicted_probability": 0.32,
                "predicted_label": 0,
                "actual_label": 0,
                "threshold": 0.5,
                "top_risk_increasing_features": [
                    {"feature_name": "feature_c", "contribution": 0.11}
                ],
                "top_risk_decreasing_features": [
                    {"feature_name": "feature_a", "contribution": -0.05}
                ],
                "explanation_generated": True,
                "failure_reason": None,
            }
        ],
    )

    _write_jsonl(
        settings.llm_reports_combined_dir / "llm_reports.jsonl",
        [
            {
                "report_id": "rr_1",
                "applicant_id": 700001,
                "report_type": "plain",
                "explanation_method_source": "shap",
                "generated_text": "Existing report text",
                "fallback_generated": False,
                "failure_reason": None,
                "generation_timestamp": "2026-01-01T00:00:00+00:00",
            }
        ],
    )
    (settings.llm_reports_combined_dir / "llm_reports.csv").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (settings.llm_reports_combined_dir / "llm_reports.csv").write_text(
        "report_id,applicant_id,report_type\nrr_1,700001,plain\n",
        encoding="utf-8",
    )
    settings.llm_reports_reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.llm_reports_reports_dir / "llm_reporting_summary.md").write_text(
        "# LLM Reporting Summary\n",
        encoding="utf-8",
    )


def _override_runtime_service(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> Callable[[], None]:
    _prepare_api_artifacts(synthetic_settings)

    class _FakeOllamaClient:
        def __init__(self, **_: Any) -> None:
            pass

        def healthcheck(self, *, required_model: str | None = None) -> OllamaHealthcheckResult:
            del required_model
            return OllamaHealthcheckResult(
                reachable=True,
                available_models=("qwen2.5:7b",),
                failure_reason=None,
            )

    monkeypatch.setattr("credit_risk_altdata.api.services.OllamaClient", _FakeOllamaClient)

    service = APIRuntimeService(synthetic_settings, model_store=ModelStore(synthetic_settings))
    app.dependency_overrides[get_runtime_service] = lambda: service

    def _cleanup() -> None:
        app.dependency_overrides.clear()

    return _cleanup


def test_health_endpoint(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    cleanup = _override_runtime_service(synthetic_settings, monkeypatch)
    try:
        client = TestClient(app)
        response = client.get("/health")
    finally:
        cleanup()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_readiness_endpoint(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    cleanup = _override_runtime_service(synthetic_settings, monkeypatch)
    try:
        client = TestClient(app)
        response = client.get("/readiness")
    finally:
        cleanup()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ready", "degraded"}
    component_names = {component["name"] for component in payload["components"]}
    assert {
        "final_candidate_summary",
        "final_model_artifact",
        "feature_schema",
        "explainability_root",
        "llm_reports_root",
        "ollama",
    }.issubset(component_names)


def test_score_success_and_extra_feature_warning(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    cleanup = _override_runtime_service(synthetic_settings, monkeypatch)
    try:
        client = TestClient(app)
        response = client.post(
            "/score",
            json={
                "applicant_id": 700001,
                "engineered_features": {
                    "feature_a": 0.5,
                    "feature_b": 0.4,
                    "feature_c": 0.3,
                    "extra_feature": 1.0,
                },
            },
        )
    finally:
        cleanup()

    assert response.status_code == 200
    payload = response.json()
    assert payload["applicant_id"] == 700001
    assert 0.0 <= payload["predicted_probability"] <= 1.0
    assert payload["predicted_label"] in {0, 1}
    assert payload["final_candidate_name"] == "lightgbm_tuned_none"
    assert payload["warnings"]


def test_score_missing_required_feature_returns_structured_error(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    cleanup = _override_runtime_service(synthetic_settings, monkeypatch)
    try:
        client = TestClient(app)
        response = client.post(
            "/score",
            json={
                "engineered_features": {
                    "feature_a": 0.5,
                    "feature_b": 0.4,
                }
            },
        )
    finally:
        cleanup()

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "invalid_engineered_features"


def test_explain_retrieval_path(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    cleanup = _override_runtime_service(synthetic_settings, monkeypatch)
    try:
        client = TestClient(app)
        response = client.post(
            "/explain",
            json={
                "applicant_id": 700001,
                "explanation_method": "shap",
                "allow_generate_if_missing": False,
            },
        )
    finally:
        cleanup()

    assert response.status_code == 200
    payload = response.json()
    assert payload["explanation_available"] is True
    assert payload["explanation_method_resolved"] == "shap"
    assert payload["payload"]["applicant_id"] == 700001


def test_explain_generation_path_when_artifact_missing(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    cleanup = _override_runtime_service(synthetic_settings, monkeypatch)

    shap_path = synthetic_settings.explainability_shap_local_dir / "shap_local_explanations.jsonl"
    shap_path.unlink(missing_ok=True)

    def fake_generate_explanations(*_: Any, **__: Any) -> object:
        _write_jsonl(
            shap_path,
            [
                {
                    "explanation_method": "shap",
                    "applicant_id": 700001,
                    "cohort_name": "false_positive",
                    "predicted_probability": 0.74,
                    "predicted_label": 1,
                    "actual_label": 0,
                    "threshold": 0.5,
                    "top_risk_increasing_features": [],
                    "top_risk_decreasing_features": [],
                    "explanation_generated": True,
                    "failure_reason": None,
                }
            ],
        )
        return object()

    monkeypatch.setattr(
        "credit_risk_altdata.api.services.run_explainability_workflow",
        fake_generate_explanations,
    )

    try:
        client = TestClient(app)
        response = client.post(
            "/explain",
            json={
                "applicant_id": 700001,
                "explanation_method": "shap",
                "allow_generate_if_missing": True,
            },
        )
    finally:
        cleanup()

    assert response.status_code == 200
    payload = response.json()
    assert payload["explanation_available"] is True
    assert payload["explanation_generated"] is True


def test_risk_report_retrieval_path(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    cleanup = _override_runtime_service(synthetic_settings, monkeypatch)
    try:
        client = TestClient(app)
        response = client.post(
            "/risk-report",
            json={
                "applicant_id": 700001,
                "explanation_method_source": "auto",
                "report_type": "plain",
                "allow_generate_if_missing": False,
                "allow_fallback": True,
            },
        )
    finally:
        cleanup()

    assert response.status_code == 200
    payload = response.json()
    assert payload["report_available"] is True
    assert payload["reports"][0]["report_id"] == "rr_1"


def test_risk_report_generation_path_with_mocked_workflow(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    cleanup = _override_runtime_service(synthetic_settings, monkeypatch)

    report_jsonl_path = synthetic_settings.llm_reports_combined_dir / "llm_reports.jsonl"
    report_jsonl_path.write_text("", encoding="utf-8")

    def fake_run_llm_reporting_workflow(
        settings: Settings,
        *,
        report_type_selection: str,
        method_source_selection: str,
        model_name_override: str | None = None,
        limit: int | None = None,
        input_path_override: Path | None = None,
        overwrite: bool = False,
    ) -> LLMReportingWorkflowResult:
        del model_name_override
        del input_path_override
        del overwrite
        _write_jsonl(
            settings.llm_reports_combined_dir / "llm_reports.jsonl",
            [
                {
                    "report_id": "rr_generated",
                    "applicant_id": 700001,
                    "report_type": "plain",
                    "explanation_method_source": method_source_selection,
                    "generated_text": "Generated fallback report",
                    "fallback_generated": True,
                    "failure_reason": "Ollama unavailable",
                    "generation_timestamp": "2026-01-01T00:00:00+00:00",
                }
            ],
        )
        return LLMReportingWorkflowResult(
            source_artifact_path=settings.explainability_shap_local_dir
            / "shap_local_explanations.jsonl",
            method_source_used=method_source_selection,  # type: ignore[arg-type]
            reports_jsonl_path=settings.llm_reports_combined_dir / "llm_reports.jsonl",
            reports_csv_path=settings.llm_reports_combined_dir / "llm_reports.csv",
            reporting_summary_path=settings.llm_reports_reports_dir / "llm_reporting_summary.md",
            total_reports=1,
            llm_generated_reports=0,
            fallback_generated_reports=1,
        )

    monkeypatch.setattr(
        "credit_risk_altdata.api.services.run_llm_reporting_workflow",
        fake_run_llm_reporting_workflow,
    )

    try:
        client = TestClient(app)
        response = client.post(
            "/risk-report",
            json={
                "applicant_id": 700001,
                "explanation_method_source": "shap",
                "report_type": "plain",
                "allow_generate_if_missing": True,
                "allow_fallback": True,
            },
        )
    finally:
        cleanup()

    assert response.status_code == 200
    payload = response.json()
    assert payload["report_generated"] is True
    assert payload["report_available"] is True
    assert payload["fallback_generated"] is True


def test_artifacts_summary_endpoint(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    cleanup = _override_runtime_service(synthetic_settings, monkeypatch)
    try:
        client = TestClient(app)
        response = client.get("/artifacts/summary")
    finally:
        cleanup()

    assert response.status_code == 200
    payload = response.json()
    assert payload["final_candidate_name"] == "lightgbm_tuned_none"
    assert payload["paths"]["final_model_artifact"]["exists"] is True


def test_score_invalid_schema_returns_validation_error(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    cleanup = _override_runtime_service(synthetic_settings, monkeypatch)
    try:
        client = TestClient(app)
        response = client.post(
            "/score",
            json={
                "engineered_features": {
                    "feature_a": "bad_value",
                    "feature_b": 0.4,
                    "feature_c": 0.3,
                }
            },
        )
    finally:
        cleanup()

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "request_validation_error"
