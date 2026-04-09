"""Artifact contract verification for demo and API readiness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk_altdata.config import Settings
from credit_risk_altdata.explainability.constants import (
    EXPLAINABILITY_SUMMARY_FILE,
    LIME_LOCAL_EXPLANATIONS_FILE,
    SELECTED_EXAMPLES_FILE,
    SHAP_LOCAL_EXPLANATIONS_FILE,
)
from credit_risk_altdata.llm.constants import LLM_REPORTING_SUMMARY_FILE, LLM_REPORTS_JSONL_FILE
from credit_risk_altdata.modeling.constants import (
    BEST_MODEL_SUMMARY_FILE,
    FINAL_PRODUCTION_CANDIDATE_FILE,
    TUNED_MODEL_COMPARISON_FILE,
    TUNING_RESULTS_FILE,
)


@dataclass(frozen=True, slots=True)
class ArtifactCheck:
    """Result of one artifact contract check."""

    name: str
    required: bool
    passed: bool
    path: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "required": self.required,
            "passed": self.passed,
            "path": self.path,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ArtifactVerificationReport:
    """Structured output for Phase 9 artifact verification."""

    checks: tuple[ArtifactCheck, ...]
    generated_at: str

    @property
    def is_valid(self) -> bool:
        return all(check.passed for check in self.checks if check.required)

    @property
    def required_failed_count(self) -> int:
        return sum(1 for check in self.checks if check.required and not check.passed)

    @property
    def optional_failed_count(self) -> int:
        return sum(1 for check in self.checks if (not check.required) and (not check.passed))

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "required_failed_count": self.required_failed_count,
            "optional_failed_count": self.optional_failed_count,
            "generated_at": self.generated_at,
            "checks": [check.to_dict() for check in self.checks],
        }


def _resolve_project_path(settings: Settings, raw_path: str | Path) -> Path:
    path = raw_path if isinstance(raw_path, Path) else Path(str(raw_path))
    if path.is_absolute():
        return path
    return settings.project_root / path


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload_raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, dict):
        raise ValueError(f"{label} must be a JSON object")
    return dict(payload_raw)


def _append_check(
    checks: list[ArtifactCheck],
    *,
    name: str,
    required: bool,
    passed: bool,
    path: Path | None,
    message: str,
) -> None:
    checks.append(
        ArtifactCheck(
            name=name,
            required=required,
            passed=passed,
            path=str(path) if path is not None else None,
            message=message,
        )
    )


def _check_exists(
    checks: list[ArtifactCheck],
    *,
    name: str,
    required: bool,
    path: Path,
    success_message: str,
    failure_message: str,
) -> bool:
    exists = path.exists()
    _append_check(
        checks,
        name=name,
        required=required,
        passed=exists,
        path=path,
        message=success_message if exists else failure_message,
    )
    return exists


def verify_artifact_contracts(settings: Settings) -> ArtifactVerificationReport:
    """Verify core Phase 3-8 artifact contracts for reproducible demos."""
    checks: list[ArtifactCheck] = []

    train_features_path = settings.home_credit_processed_dir / "train_features.parquet"
    test_features_path = settings.home_credit_processed_dir / "test_features.parquet"
    _check_exists(
        checks,
        name="processed_train_features",
        required=True,
        path=train_features_path,
        success_message="Processed train feature matrix is present.",
        failure_message="Missing processed train feature matrix.",
    )
    _check_exists(
        checks,
        name="processed_test_features",
        required=True,
        path=test_features_path,
        success_message="Processed test feature matrix is present.",
        failure_message="Missing processed test feature matrix.",
    )

    feature_manifest_path = settings.feature_metadata_dir / "feature_manifest.csv"
    _check_exists(
        checks,
        name="feature_manifest",
        required=True,
        path=feature_manifest_path,
        success_message="Feature manifest is present for API schema assumptions.",
        failure_message="Missing feature manifest required by API scoring.",
    )

    baseline_summary_path = settings.modeling_reports_dir / BEST_MODEL_SUMMARY_FILE
    if _check_exists(
        checks,
        name="baseline_summary",
        required=True,
        path=baseline_summary_path,
        success_message="Baseline summary artifact is present.",
        failure_message="Missing baseline summary artifact.",
    ):
        try:
            baseline_summary = _load_json_object(
                baseline_summary_path,
                label=BEST_MODEL_SUMMARY_FILE,
            )
            required_keys = {"best_model_name", "primary_metric", "primary_metric_value"}
            missing = sorted(required_keys.difference(baseline_summary))
            _append_check(
                checks,
                name="baseline_summary_contract",
                required=True,
                passed=not missing,
                path=baseline_summary_path,
                message=(
                    "Baseline summary includes required keys."
                    if not missing
                    else f"Missing keys: {missing}"
                ),
            )
        except Exception as exc:
            _append_check(
                checks,
                name="baseline_summary_contract",
                required=True,
                passed=False,
                path=baseline_summary_path,
                message=f"Invalid baseline summary JSON: {exc}",
            )

    tuning_results_path = settings.modeling_tuning_dir / TUNING_RESULTS_FILE
    _check_exists(
        checks,
        name="tuning_results",
        required=True,
        path=tuning_results_path,
        success_message="Tuning results artifact is present.",
        failure_message="Missing tuning results artifact.",
    )

    tuned_comparison_path = settings.modeling_metrics_dir / TUNED_MODEL_COMPARISON_FILE
    tuned_comparison_frame: pd.DataFrame | None = None
    if _check_exists(
        checks,
        name="tuned_model_comparison",
        required=True,
        path=tuned_comparison_path,
        success_message="Tuned model comparison artifact is present.",
        failure_message="Missing tuned model comparison artifact.",
    ):
        try:
            tuned_comparison_frame = pd.read_csv(tuned_comparison_path)
            has_candidate_name = "candidate_name" in tuned_comparison_frame.columns
            _append_check(
                checks,
                name="tuned_model_comparison_contract",
                required=True,
                passed=has_candidate_name,
                path=tuned_comparison_path,
                message=(
                    "Tuned model comparison has candidate_name column."
                    if has_candidate_name
                    else "tuned_model_comparison.csv is missing candidate_name column"
                ),
            )
        except Exception as exc:
            _append_check(
                checks,
                name="tuned_model_comparison_contract",
                required=True,
                passed=False,
                path=tuned_comparison_path,
                message=f"Unable to parse tuned model comparison CSV: {exc}",
            )

    final_candidate_summary_path = settings.modeling_reports_dir / FINAL_PRODUCTION_CANDIDATE_FILE
    final_candidate_summary: dict[str, Any] | None = None
    selected_artifact_path: Path | None = None
    final_model_output_path: Path | None = None

    if _check_exists(
        checks,
        name="final_candidate_summary",
        required=True,
        path=final_candidate_summary_path,
        success_message="Final production candidate summary is present.",
        failure_message="Missing final production candidate summary.",
    ):
        try:
            final_candidate_summary = _load_json_object(
                final_candidate_summary_path,
                label=FINAL_PRODUCTION_CANDIDATE_FILE,
            )
            required_keys = {
                "final_candidate_name",
                "final_model_family",
                "threshold",
                "selected_artifact_path",
                "final_model_output_path",
                "source_comparison_artifact",
            }
            missing = sorted(required_keys.difference(final_candidate_summary))
            _append_check(
                checks,
                name="final_candidate_summary_contract",
                required=True,
                passed=not missing,
                path=final_candidate_summary_path,
                message=(
                    "Final candidate summary includes required keys."
                    if not missing
                    else f"Missing keys: {missing}"
                ),
            )

            threshold_value = final_candidate_summary.get("threshold")
            threshold_ok = isinstance(threshold_value, (float, int)) and 0.0 < float(
                threshold_value
            ) < 1.0
            _append_check(
                checks,
                name="final_candidate_threshold_range",
                required=True,
                passed=threshold_ok,
                path=final_candidate_summary_path,
                message=(
                    "Final candidate threshold is within (0,1)."
                    if threshold_ok
                    else f"Invalid threshold: {threshold_value!r}"
                ),
            )

            selected_raw = final_candidate_summary.get("selected_artifact_path")
            if isinstance(selected_raw, str) and selected_raw.strip():
                selected_artifact_path = _resolve_project_path(settings, selected_raw)
                selected_exists = selected_artifact_path.exists()
                _append_check(
                    checks,
                    name="selected_model_artifact",
                    required=True,
                    passed=selected_exists,
                    path=selected_artifact_path,
                    message=(
                        "Selected tuned model artifact exists."
                        if selected_exists
                        else "Selected tuned model artifact is missing."
                    ),
                )
            else:
                _append_check(
                    checks,
                    name="selected_model_artifact",
                    required=True,
                    passed=False,
                    path=None,
                    message="selected_artifact_path is missing or empty.",
                )

            final_raw = final_candidate_summary.get("final_model_output_path")
            if isinstance(final_raw, str) and final_raw.strip():
                final_model_output_path = _resolve_project_path(settings, final_raw)
                final_exists = final_model_output_path.exists()
                _append_check(
                    checks,
                    name="final_model_output_artifact",
                    required=True,
                    passed=final_exists,
                    path=final_model_output_path,
                    message=(
                        "Final production model output exists."
                        if final_exists
                        else "Final production model output is missing."
                    ),
                )
            else:
                _append_check(
                    checks,
                    name="final_model_output_artifact",
                    required=True,
                    passed=False,
                    path=None,
                    message="final_model_output_path is missing or empty.",
                )

            source_raw = final_candidate_summary.get("source_comparison_artifact")
            if isinstance(source_raw, str) and source_raw.strip():
                source_path = _resolve_project_path(settings, source_raw)
                source_exists = source_path.exists()
                _append_check(
                    checks,
                    name="final_candidate_source_comparison_exists",
                    required=True,
                    passed=source_exists,
                    path=source_path,
                    message=(
                        "source_comparison_artifact path exists."
                        if source_exists
                        else "source_comparison_artifact path is missing."
                    ),
                )

                _append_check(
                    checks,
                    name="final_candidate_source_points_to_tuned_comparison",
                    required=True,
                    passed=source_path.resolve() == tuned_comparison_path.resolve(),
                    path=source_path,
                    message=(
                        "Final candidate source comparison points to tuned_model_comparison.csv."
                        if source_path.resolve() == tuned_comparison_path.resolve()
                        else (
                            "Final candidate source comparison does not point to "
                            "artifacts/modeling/metrics/tuned_model_comparison.csv."
                        )
                    ),
                )
            else:
                _append_check(
                    checks,
                    name="final_candidate_source_comparison_exists",
                    required=True,
                    passed=False,
                    path=None,
                    message="source_comparison_artifact is missing or empty.",
                )
                _append_check(
                    checks,
                    name="final_candidate_source_points_to_tuned_comparison",
                    required=True,
                    passed=False,
                    path=None,
                    message="Cannot validate source comparison target because source is missing.",
                )
        except Exception as exc:
            _append_check(
                checks,
                name="final_candidate_summary_contract",
                required=True,
                passed=False,
                path=final_candidate_summary_path,
                message=f"Invalid final candidate summary JSON: {exc}",
            )

    if selected_artifact_path is not None and final_model_output_path is not None:
        suffix_match = (
            selected_artifact_path.suffix.lower()
            == final_model_output_path.suffix.lower()
        )
        _append_check(
            checks,
            name="final_model_suffix_consistency",
            required=True,
            passed=suffix_match,
            path=final_model_output_path,
            message=(
                "Final production model suffix matches selected tuned artifact suffix."
                if suffix_match
                else (
                    "Final production model suffix does not match selected tuned artifact "
                    "suffix."
                )
            ),
        )

    if final_candidate_summary is not None and tuned_comparison_frame is not None:
        candidate_name = str(final_candidate_summary.get("final_candidate_name", ""))
        if "candidate_name" in tuned_comparison_frame.columns:
            candidate_present = candidate_name in set(
                tuned_comparison_frame["candidate_name"].astype(str)
            )
            _append_check(
                checks,
                name="final_candidate_in_tuned_comparison",
                required=True,
                passed=candidate_present,
                path=tuned_comparison_path,
                message=(
                    "Final candidate is present in tuned_model_comparison.csv."
                    if candidate_present
                    else (
                        "Final candidate is not present in tuned_model_comparison.csv: "
                        f"{candidate_name}"
                    )
                ),
            )

    selected_examples_path = settings.explainability_selected_examples_dir / SELECTED_EXAMPLES_FILE
    _check_exists(
        checks,
        name="selected_examples",
        required=True,
        path=selected_examples_path,
        success_message="Explainability selected examples artifact is present.",
        failure_message="Missing explainability selected examples artifact.",
    )

    explainability_summary_path = settings.explainability_reports_dir / EXPLAINABILITY_SUMMARY_FILE
    _check_exists(
        checks,
        name="explainability_summary",
        required=True,
        path=explainability_summary_path,
        success_message="Explainability summary artifact is present.",
        failure_message="Missing explainability summary artifact.",
    )

    shap_local_path = settings.explainability_shap_local_dir / SHAP_LOCAL_EXPLANATIONS_FILE
    lime_local_path = settings.explainability_lime_dir / LIME_LOCAL_EXPLANATIONS_FILE
    has_any_local_explanations = shap_local_path.exists() or lime_local_path.exists()
    _append_check(
        checks,
        name="local_explanation_payloads",
        required=True,
        passed=has_any_local_explanations,
        path=shap_local_path if shap_local_path.exists() else lime_local_path,
        message=(
            "At least one local explanation payload artifact is present (SHAP or LIME)."
            if has_any_local_explanations
            else "Missing both SHAP and LIME local explanation payload artifacts."
        ),
    )

    llm_reports_jsonl_path = settings.llm_reports_combined_dir / LLM_REPORTS_JSONL_FILE
    _check_exists(
        checks,
        name="llm_reports_jsonl",
        required=True,
        path=llm_reports_jsonl_path,
        success_message="LLM report combined JSONL artifact is present.",
        failure_message="Missing LLM report combined JSONL artifact.",
    )

    llm_reporting_summary_path = settings.llm_reports_reports_dir / LLM_REPORTING_SUMMARY_FILE
    _check_exists(
        checks,
        name="llm_reporting_summary",
        required=True,
        path=llm_reporting_summary_path,
        success_message="LLM reporting summary artifact is present.",
        failure_message="Missing LLM reporting summary artifact.",
    )

    _check_exists(
        checks,
        name="api_assumption_explainability_root",
        required=True,
        path=settings.explainability_root_dir,
        success_message="Explainability root directory exists for API readiness assumptions.",
        failure_message="Explainability root directory is missing.",
    )
    _check_exists(
        checks,
        name="api_assumption_llm_reports_root",
        required=True,
        path=settings.llm_reports_root_dir,
        success_message="LLM reports root directory exists for API readiness assumptions.",
        failure_message="LLM reports root directory is missing.",
    )

    generated_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    return ArtifactVerificationReport(checks=tuple(checks), generated_at=generated_at)
