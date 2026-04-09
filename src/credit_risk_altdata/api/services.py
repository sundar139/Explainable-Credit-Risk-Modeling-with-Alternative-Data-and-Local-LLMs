"""Service-layer orchestration for API endpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from credit_risk_altdata.api.model_store import ModelStore
from credit_risk_altdata.config import Settings
from credit_risk_altdata.explainability.constants import (
    LIME_LOCAL_EXPLANATIONS_FILE,
    METHOD_ALL,
    METHOD_LIME,
    METHOD_SHAP,
    SHAP_LOCAL_EXPLANATIONS_FILE,
    ExplainabilityMethodSelection,
)
from credit_risk_altdata.explainability.workflow import run_explainability_workflow
from credit_risk_altdata.llm.constants import (
    DRAFT_DISCLAIMER,
    LLM_REPORTS_JSONL_FILE,
    METHOD_SOURCE_LIME,
    METHOD_SOURCE_SHAP,
    REPORT_TYPE_ADVERSE_ACTION,
    REPORT_TYPE_ALL,
    REPORT_TYPE_PLAIN,
    REPORT_TYPE_UNDERWRITER,
    ExplanationMethodSourceSelection,
    ReportTypeSelection,
)
from credit_risk_altdata.llm.ollama_client import OllamaClient
from credit_risk_altdata.llm.workflow import run_llm_reporting_workflow
from credit_risk_altdata.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class APIServiceError(RuntimeError):
    """Structured API service-layer error."""

    status_code: int
    code: str
    message: str
    details: dict[str, Any] | None = None

    def to_detail(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class APIRuntimeService:
    """Runtime API services backed by Phase 5-7 artifacts."""

    def __init__(self, settings: Settings, *, model_store: ModelStore | None = None) -> None:
        self.settings = settings
        self.model_store = model_store if model_store is not None else ModelStore(settings)

    @staticmethod
    def _applicant_key(value: Any) -> str:
        return str(value)

    @staticmethod
    def _path_status(path: Path) -> dict[str, Any]:
        return {
            "path": str(path),
            "exists": path.exists(),
        }

    @staticmethod
    def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []

        rows: list[dict[str, Any]] = []
        for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload_raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL row in {path} at line {line_index}: {exc}"
                ) from exc
            if isinstance(payload_raw, dict):
                rows.append(payload_raw)
        return rows

    def health_payload(self) -> dict[str, Any]:
        """Build /health response payload."""
        return {
            "status": "ok",
            "environment": self.settings.app_env,
            "service": self.settings.app_name,
        }

    def readiness_payload(self) -> dict[str, Any]:
        """Build structured component-level readiness payload."""
        components: list[dict[str, Any]] = []

        def add_component(
            name: str,
            ready: bool,
            *,
            optional: bool = False,
            path: Path | None = None,
            detail: str | None = None,
        ) -> None:
            components.append(
                {
                    "name": name,
                    "ready": ready,
                    "optional": optional,
                    "path": str(path) if path is not None else None,
                    "detail": detail,
                }
            )

        candidate_path = self.settings.modeling_final_candidate_summary_path
        try:
            self.model_store.get_final_candidate_summary()
            add_component(
                "final_candidate_summary",
                True,
                path=candidate_path,
            )
        except Exception as exc:
            add_component(
                "final_candidate_summary",
                False,
                path=candidate_path,
                detail=str(exc),
            )

        try:
            model_path = self.model_store.get_model_artifact_path()
            add_component("final_model_artifact", True, path=model_path)
        except Exception as exc:
            add_component(
                "final_model_artifact",
                False,
                path=self.settings.modeling_final_model_output_path,
                detail=str(exc),
            )

        try:
            feature_columns = self.model_store.get_expected_feature_columns()
            add_component(
                "feature_schema",
                bool(feature_columns),
                path=self.model_store.feature_manifest_path,
                detail=(
                    f"expected_feature_count={len(feature_columns)}"
                    if feature_columns
                    else "Expected feature schema is empty"
                ),
            )
        except Exception as exc:
            add_component(
                "feature_schema",
                False,
                path=self.model_store.feature_manifest_path,
                detail=str(exc),
            )

        explainability_root = self.settings.explainability_root_dir
        add_component(
            "explainability_root",
            explainability_root.exists(),
            path=explainability_root,
            detail=(
                None
                if explainability_root.exists()
                else "Explainability root directory missing"
            ),
        )

        llm_reports_root = self.settings.llm_reports_root_dir
        add_component(
            "llm_reports_root",
            llm_reports_root.exists(),
            path=llm_reports_root,
            detail=None if llm_reports_root.exists() else "LLM reports root directory missing",
        )

        ollama_client = OllamaClient(
            base_url=str(self.settings.ollama_base_url),
            model=self.settings.llm_reports_model_name,
            timeout_seconds=float(self.settings.llm_reports_timeout_seconds),
            max_retries=int(self.settings.llm_reports_retries),
        )
        health = ollama_client.healthcheck(required_model=self.settings.llm_reports_model_name)
        add_component(
            "ollama",
            health.reachable and health.failure_reason is None,
            optional=True,
            detail=(
                None
                if health.reachable and health.failure_reason is None
                else health.failure_reason
            ),
        )

        required_components = [component for component in components if not component["optional"]]
        all_required_ready = all(bool(component["ready"]) for component in required_components)
        all_optional_ready = all(
            bool(component["ready"]) for component in components if component["optional"]
        )

        if not all_required_ready:
            status = "not_ready"
            ready = False
        elif not all_optional_ready:
            status = "degraded"
            ready = True
        else:
            status = "ready"
            ready = True

        return {
            "status": status,
            "ready": ready,
            "components": components,
        }

    def score_payload(
        self,
        *,
        engineered_features: dict[str, float | int | bool | None],
        applicant_id: int | str | None,
        threshold: float | None,
        include_prediction_label: bool,
    ) -> dict[str, Any]:
        """Build /score response payload using cached production model."""
        try:
            candidate_summary = self.model_store.get_final_candidate_summary()
            model = self.model_store.get_model()
            x_frame, warnings = self.model_store.build_scoring_frame(engineered_features)
            predicted_probability = self.model_store.predict_positive_probability(model, x_frame)
        except FileNotFoundError as exc:
            raise APIServiceError(
                status_code=503,
                code="model_artifact_missing",
                message="Model artifacts are unavailable for scoring",
                details={"reason": str(exc)},
            ) from exc
        except ValueError as exc:
            raise APIServiceError(
                status_code=422,
                code="invalid_engineered_features",
                message="Score request contains invalid engineered feature payload",
                details={"reason": str(exc)},
            ) from exc
        except Exception as exc:
            raise APIServiceError(
                status_code=500,
                code="score_failed",
                message="Scoring failed unexpectedly",
                details={"reason": str(exc)},
            ) from exc

        resolved_threshold = (
            float(threshold) if threshold is not None else float(candidate_summary["threshold"])
        )
        if not (0.0 < resolved_threshold < 1.0):
            raise APIServiceError(
                status_code=422,
                code="invalid_threshold",
                message="Threshold must be between 0 and 1",
                details={"threshold": resolved_threshold},
            )

        predicted_label = (
            int(predicted_probability >= resolved_threshold)
            if include_prediction_label
            else None
        )

        return {
            "applicant_id": applicant_id,
            "predicted_probability": predicted_probability,
            "predicted_label": predicted_label,
            "threshold": resolved_threshold,
            "final_model_family": str(candidate_summary["final_model_family"]),
            "final_candidate_name": str(candidate_summary["final_candidate_name"]),
            "model_artifact_path": str(self.model_store.get_model_artifact_path()),
            "warnings": warnings,
        }

    def _explanation_path_for_method(self, method: Literal["shap", "lime"]) -> Path:
        if method == METHOD_SHAP:
            return self.settings.explainability_shap_local_dir / SHAP_LOCAL_EXPLANATIONS_FILE
        if method == METHOD_LIME:
            return self.settings.explainability_lime_dir / LIME_LOCAL_EXPLANATIONS_FILE
        raise ValueError(f"Unsupported explanation method: {method}")

    @staticmethod
    def _explanation_methods_for_selection(
        selection: Literal["shap", "lime", "auto"],
    ) -> list[Literal["shap", "lime"]]:
        if selection == "auto":
            return [METHOD_SHAP, METHOD_LIME]
        if selection == METHOD_SHAP:
            return [METHOD_SHAP]
        if selection == METHOD_LIME:
            return [METHOD_LIME]
        raise ValueError(f"Unsupported explanation method selection: {selection}")

    def _find_explanation_payload(
        self,
        *,
        applicant_id: int | str,
        methods: list[Literal["shap", "lime"]],
    ) -> tuple[dict[str, Any], Path, Literal["shap", "lime"]] | None:
        applicant_key = self._applicant_key(applicant_id)
        for method in methods:
            path = self._explanation_path_for_method(method)
            rows = self._load_jsonl_rows(path)
            for row in rows:
                if self._applicant_key(row.get("applicant_id")) == applicant_key:
                    return row, path, method
        return None

    def explain_payload(
        self,
        *,
        applicant_id: int | str | None,
        explanation_method: Literal["shap", "lime", "auto"],
        allow_generate_if_missing: bool,
        top_k: int | None,
        engineered_features: dict[str, float | int | bool | None] | None,
    ) -> dict[str, Any]:
        """Build /explain response payload using existing or generated artifacts."""
        warnings: list[str] = []
        errors: list[str] = []

        if applicant_id is None:
            raise APIServiceError(
                status_code=422,
                code="missing_applicant_id",
                message=(
                    "Explain endpoint currently supports artifact-backed retrieval by applicant_id."
                ),
                details={
                    "explanation_method": explanation_method,
                },
            )

        if engineered_features is not None:
            warnings.append(
                "engineered_features is currently unused by /explain. "
                "The endpoint uses artifact-backed retrieval/generation only."
            )

        methods = self._explanation_methods_for_selection(explanation_method)
        existing = self._find_explanation_payload(applicant_id=applicant_id, methods=methods)
        generated = False

        if existing is None and allow_generate_if_missing:
            selected_paths = [self._explanation_path_for_method(method) for method in methods]
            any_missing = any(not path.exists() for path in selected_paths)
            if any_missing:
                method_selection: ExplainabilityMethodSelection = (
                    METHOD_ALL if explanation_method == "auto" else explanation_method
                )
                try:
                    run_explainability_workflow(
                        self.settings,
                        method_selection=method_selection,
                        top_k=top_k,
                        overwrite=True,
                    )
                    generated = True
                except Exception as exc:
                    raise APIServiceError(
                        status_code=500,
                        code="explainability_generation_failed",
                        message="Failed to generate explainability artifacts",
                        details={"reason": str(exc)},
                    ) from exc

                existing = self._find_explanation_payload(
                    applicant_id=applicant_id,
                    methods=methods,
                )
            else:
                warnings.append(
                    "Explainability artifacts already exist; applicant was not found in "
                    "stored cohorts."
                )

        if existing is None:
            return {
                "applicant_id": applicant_id,
                "explanation_method_requested": explanation_method,
                "explanation_method_resolved": None,
                "explanation_available": False,
                "explanation_generated": generated,
                "payload": None,
                "payload_path": None,
                "cohort_name": None,
                "warnings": warnings,
                "errors": errors,
            }

        payload, payload_path, resolved_method = existing
        cohort_name_raw = payload.get("cohort_name")
        cohort_name = str(cohort_name_raw) if cohort_name_raw is not None else None

        return {
            "applicant_id": applicant_id,
            "explanation_method_requested": explanation_method,
            "explanation_method_resolved": resolved_method,
            "explanation_available": True,
            "explanation_generated": generated,
            "payload": payload,
            "payload_path": str(payload_path),
            "cohort_name": cohort_name,
            "warnings": warnings,
            "errors": errors,
        }

    @staticmethod
    def _report_types_for_selection(
        selection: ReportTypeSelection,
    ) -> list[Literal["plain", "underwriter", "adverse-action"]]:
        if selection == REPORT_TYPE_ALL:
            return [REPORT_TYPE_PLAIN, REPORT_TYPE_UNDERWRITER, REPORT_TYPE_ADVERSE_ACTION]
        if selection == REPORT_TYPE_PLAIN:
            return [REPORT_TYPE_PLAIN]
        if selection == REPORT_TYPE_UNDERWRITER:
            return [REPORT_TYPE_UNDERWRITER]
        if selection == REPORT_TYPE_ADVERSE_ACTION:
            return [REPORT_TYPE_ADVERSE_ACTION]
        raise ValueError(f"Unsupported report type selection: {selection}")

    def _report_rows_for_applicant(
        self,
        *,
        applicant_id: int | str,
        report_type_selection: ReportTypeSelection,
    ) -> list[dict[str, Any]]:
        report_path = self.settings.llm_reports_combined_dir / LLM_REPORTS_JSONL_FILE
        rows = self._load_jsonl_rows(report_path)

        applicant_key = self._applicant_key(applicant_id)
        selected_types = set(self._report_types_for_selection(report_type_selection))

        filtered: list[dict[str, Any]] = []
        for row in rows:
            if self._applicant_key(row.get("applicant_id")) != applicant_key:
                continue
            row_type = row.get("report_type")
            if not isinstance(row_type, str):
                continue
            if row_type not in selected_types:
                continue
            filtered.append(row)

        return filtered

    def _explanation_position(
        self,
        *,
        applicant_id: int | str,
        method_source: Literal["shap", "lime"],
    ) -> int | None:
        path = self._explanation_path_for_method(method_source)
        rows = self._load_jsonl_rows(path)
        applicant_key = self._applicant_key(applicant_id)

        for index, row in enumerate(rows, start=1):
            if self._applicant_key(row.get("applicant_id")) == applicant_key:
                return index
        return None

    def _resolve_method_source_for_applicant(
        self,
        *,
        applicant_id: int | str,
        method_source_selection: ExplanationMethodSourceSelection,
    ) -> tuple[Literal["shap", "lime"] | None, int | None]:
        if method_source_selection == METHOD_SOURCE_SHAP:
            return METHOD_SOURCE_SHAP, self._explanation_position(
                applicant_id=applicant_id,
                method_source=METHOD_SOURCE_SHAP,
            )

        if method_source_selection == METHOD_SOURCE_LIME:
            return METHOD_SOURCE_LIME, self._explanation_position(
                applicant_id=applicant_id,
                method_source=METHOD_SOURCE_LIME,
            )

        shap_position = self._explanation_position(
            applicant_id=applicant_id,
            method_source=METHOD_SOURCE_SHAP,
        )
        if shap_position is not None:
            return METHOD_SOURCE_SHAP, shap_position

        lime_position = self._explanation_position(
            applicant_id=applicant_id,
            method_source=METHOD_SOURCE_LIME,
        )
        if lime_position is not None:
            return METHOD_SOURCE_LIME, lime_position

        shap_path = self._explanation_path_for_method(METHOD_SOURCE_SHAP)
        lime_path = self._explanation_path_for_method(METHOD_SOURCE_LIME)
        if shap_path.exists():
            return METHOD_SOURCE_SHAP, None
        if lime_path.exists():
            return METHOD_SOURCE_LIME, None
        return None, None

    @staticmethod
    def _to_report_record(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "report_id": str(row.get("report_id", "")),
            "report_type": str(row.get("report_type", "plain")),
            "explanation_method_source": str(row.get("explanation_method_source", "unknown")),
            "generated_text": str(row.get("generated_text", "")),
            "fallback_generated": bool(row.get("fallback_generated", False)),
            "failure_reason": (
                str(row["failure_reason"])
                if row.get("failure_reason") is not None
                else None
            ),
            "generation_timestamp": str(row.get("generation_timestamp", "")),
        }

    def risk_report_payload(
        self,
        *,
        applicant_id: int | str | None,
        explanation_method_source: ExplanationMethodSourceSelection,
        report_type: ReportTypeSelection,
        allow_generate_if_missing: bool,
        allow_fallback: bool,
        top_k: int | None,
    ) -> dict[str, Any]:
        """Build /risk-report response payload from existing or generated narratives."""
        warnings: list[str] = []
        errors: list[str] = []
        generated = False

        if applicant_id is None:
            raise APIServiceError(
                status_code=422,
                code="missing_applicant_id",
                message=(
                    "risk-report currently supports retrieval/generation by applicant_id only."
                ),
                details={
                    "report_type": report_type,
                    "explanation_method_source": explanation_method_source,
                },
            )

        if top_k is not None:
            warnings.append(
                "top_k is currently unused by /risk-report; narratives are grounded in existing "
                "Phase 6 payload feature slices."
            )

        report_rows = self._report_rows_for_applicant(
            applicant_id=applicant_id,
            report_type_selection=report_type,
        )

        if not report_rows and allow_generate_if_missing:
            resolved_method_source, position = self._resolve_method_source_for_applicant(
                applicant_id=applicant_id,
                method_source_selection=explanation_method_source,
            )

            if resolved_method_source is None:
                warnings.append(
                    "No explanation source artifacts are available for narrative generation."
                )
            elif position is None:
                warnings.append(
                    "Applicant was not found in explanation payload artifacts; "
                    "narrative generation is not feasible for this applicant."
                )
            else:
                generated = True
                generation_settings = self.settings
                if allow_fallback != self.settings.llm_reports_enable_fallback:
                    generation_settings = self.settings.model_copy(
                        update={"llm_reports_enable_fallback": allow_fallback}
                    )

                try:
                    run_llm_reporting_workflow(
                        generation_settings,
                        report_type_selection=report_type,
                        method_source_selection=resolved_method_source,
                        limit=position,
                        overwrite=True,
                    )
                except Exception as exc:
                    raise APIServiceError(
                        status_code=503,
                        code="risk_report_generation_failed",
                        message="Failed to generate local narrative risk reports",
                        details={"reason": str(exc)},
                    ) from exc

                report_rows = self._report_rows_for_applicant(
                    applicant_id=applicant_id,
                    report_type_selection=report_type,
                )

        fallback_generated = any(bool(row.get("fallback_generated", False)) for row in report_rows)

        return {
            "applicant_id": applicant_id,
            "requested_report_type": report_type,
            "report_available": bool(report_rows),
            "report_generated": generated,
            "fallback_generated": fallback_generated,
            "payload_path": str(self.settings.llm_reports_combined_dir / LLM_REPORTS_JSONL_FILE),
            "disclaimer": DRAFT_DISCLAIMER,
            "reports": [self._to_report_record(row) for row in report_rows],
            "warnings": warnings,
            "errors": errors,
        }

    def artifacts_summary_payload(self) -> dict[str, Any]:
        """Build machine-readable artifact summary payload."""
        try:
            summary = self.model_store.get_final_candidate_summary()
            final_candidate_name = str(summary.get("final_candidate_name"))
            final_model_family = str(summary.get("final_model_family"))
            threshold_value = summary.get("threshold")
            threshold = float(threshold_value) if threshold_value is not None else None
        except Exception:
            final_candidate_name = None
            final_model_family = None
            threshold = None

        model_path: Path
        try:
            model_path = self.model_store.get_model_artifact_path()
        except Exception:
            model_path = self.settings.modeling_final_model_output_path

        paths = {
            "final_candidate_summary": self._path_status(
                self.settings.modeling_final_candidate_summary_path
            ),
            "final_model_artifact": self._path_status(model_path),
            "feature_manifest": self._path_status(self.model_store.feature_manifest_path),
            "explainability_shap_local": self._path_status(
                self.settings.explainability_shap_local_dir / SHAP_LOCAL_EXPLANATIONS_FILE
            ),
            "explainability_lime_local": self._path_status(
                self.settings.explainability_lime_dir / LIME_LOCAL_EXPLANATIONS_FILE
            ),
            "llm_reports_combined_jsonl": self._path_status(
                self.settings.llm_reports_combined_dir / LLM_REPORTS_JSONL_FILE
            ),
            "llm_reports_summary": self._path_status(
                self.settings.llm_reports_reports_dir / "llm_reporting_summary.md"
            ),
        }

        return {
            "final_candidate_name": final_candidate_name,
            "final_model_family": final_model_family,
            "threshold": threshold,
            "paths": paths,
            "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }
