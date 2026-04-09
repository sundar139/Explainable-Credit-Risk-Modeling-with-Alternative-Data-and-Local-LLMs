"""Phase 7 workflow for local LLM-assisted risk-report generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from credit_risk_altdata.config import Settings
from credit_risk_altdata.explainability.constants import (
    LIME_LOCAL_EXPLANATIONS_FILE,
    SHAP_LOCAL_EXPLANATIONS_FILE,
)
from credit_risk_altdata.llm.constants import (
    ARTIFACT_VERSION,
    DRAFT_DISCLAIMER,
    LLM_REPORTING_SUMMARY_FILE,
    LLM_REPORTS_CSV_FILE,
    LLM_REPORTS_JSONL_FILE,
    METHOD_SOURCE_AUTO,
    METHOD_SOURCE_LIME,
    METHOD_SOURCE_SHAP,
    PROMPT_VERSION,
    REPORT_TYPE_ADVERSE_ACTION,
    REPORT_TYPE_ALL,
    REPORT_TYPE_PLAIN,
    REPORT_TYPE_UNDERWRITER,
    ExplanationMethodSource,
    ExplanationMethodSourceSelection,
    ReportType,
    ReportTypeSelection,
)
from credit_risk_altdata.llm.ollama_client import OllamaClient, OllamaClientError
from credit_risk_altdata.llm.prompts import build_report_prompt
from credit_risk_altdata.llm.rendering import normalize_generated_text, render_fallback_report
from credit_risk_altdata.llm.reporting import (
    LLMReportArtifactPaths,
    resolve_llm_report_artifact_paths,
    write_csv,
    write_jsonl,
    write_markdown,
)
from credit_risk_altdata.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExplanationCase:
    """Normalized local explanation payload for one applicant."""

    applicant_id: int | str
    explanation_method_source: ExplanationMethodSource
    cohort_name: str | None
    predicted_probability: float
    predicted_label: int | None
    actual_label: int | None
    threshold: float
    top_risk_increasing_features: list[dict[str, Any]]
    top_risk_decreasing_features: list[dict[str, Any]]
    source_explanation_generated: bool
    source_failure_reason: str | None


@dataclass(frozen=True, slots=True)
class LLMReportingWorkflowResult:
    """Result references from Phase 7 reporting workflow."""

    source_artifact_path: Path
    method_source_used: ExplanationMethodSource
    reports_jsonl_path: Path
    reports_csv_path: Path
    reporting_summary_path: Path
    total_reports: int
    llm_generated_reports: int
    fallback_generated_reports: int


def _safe_file_token(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)
    token = sanitized.strip("_")
    return token or "case"


def _to_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float field '{field_name}': {value!r}") from exc


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_applicant_id(value: Any) -> int | str:
    text = str(value)
    return int(text) if text.isdigit() else text


def _normalize_feature_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        feature_name_raw = row.get("feature_name")
        if not isinstance(feature_name_raw, str) or not feature_name_raw.strip():
            continue

        contribution_raw = row.get("contribution")
        contribution_value: float | None
        if isinstance(contribution_raw, bool):
            contribution_value = float(int(contribution_raw))
        elif isinstance(contribution_raw, (int, float, str)):
            try:
                contribution_value = float(contribution_raw)
            except ValueError:
                contribution_value = None
        else:
            contribution_value = None

        normalized_row: dict[str, Any] = {"feature_name": feature_name_raw.strip()}
        if contribution_value is not None:
            normalized_row["contribution"] = contribution_value
        normalized.append(normalized_row)

    return normalized


def _resolve_report_types(selection: ReportTypeSelection) -> list[ReportType]:
    if selection == REPORT_TYPE_ALL:
        return [REPORT_TYPE_PLAIN, REPORT_TYPE_UNDERWRITER, REPORT_TYPE_ADVERSE_ACTION]
    if selection == REPORT_TYPE_PLAIN:
        return [REPORT_TYPE_PLAIN]
    if selection == REPORT_TYPE_UNDERWRITER:
        return [REPORT_TYPE_UNDERWRITER]
    if selection == REPORT_TYPE_ADVERSE_ACTION:
        return [REPORT_TYPE_ADVERSE_ACTION]
    raise ValueError(f"Unsupported report type selection: {selection}")


def _resolve_default_source_path(
    settings: Settings,
    method_source_selection: ExplanationMethodSourceSelection,
) -> tuple[Path, ExplanationMethodSource]:
    shap_path = settings.explainability_shap_local_dir / SHAP_LOCAL_EXPLANATIONS_FILE
    lime_path = settings.explainability_lime_dir / LIME_LOCAL_EXPLANATIONS_FILE

    if method_source_selection == METHOD_SOURCE_SHAP:
        return shap_path, METHOD_SOURCE_SHAP
    if method_source_selection == METHOD_SOURCE_LIME:
        return lime_path, METHOD_SOURCE_LIME
    if method_source_selection == METHOD_SOURCE_AUTO:
        if shap_path.exists():
            return shap_path, METHOD_SOURCE_SHAP
        if lime_path.exists():
            return lime_path, METHOD_SOURCE_LIME
        raise FileNotFoundError(
            "No local explanation payload artifact found for method-source auto. "
            f"Checked: {shap_path} and {lime_path}"
        )

    raise ValueError(f"Unsupported method-source selection: {method_source_selection}")


def _resolve_input_source(
    settings: Settings,
    *,
    method_source_selection: ExplanationMethodSourceSelection,
    input_path_override: Path | None,
) -> tuple[Path, ExplanationMethodSource]:
    if input_path_override is None:
        return _resolve_default_source_path(settings, method_source_selection)

    resolved_path = input_path_override
    if not resolved_path.is_absolute():
        resolved_path = settings.project_root / resolved_path

    if method_source_selection == METHOD_SOURCE_SHAP:
        return resolved_path, METHOD_SOURCE_SHAP
    if method_source_selection == METHOD_SOURCE_LIME:
        return resolved_path, METHOD_SOURCE_LIME

    inferred = METHOD_SOURCE_LIME if "lime" in resolved_path.name.lower() else METHOD_SOURCE_SHAP
    return resolved_path, inferred


def _load_explanation_cases(
    *,
    source_path: Path,
    fallback_source: ExplanationMethodSource,
) -> list[ExplanationCase]:
    if not source_path.exists():
        raise FileNotFoundError(f"Explanation payload artifact not found: {source_path}")

    lines = source_path.read_text(encoding="utf-8").splitlines()
    cases: list[ExplanationCase] = []

    for line_index, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        try:
            payload_raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL row at {source_path}:{line_index}: {exc}"
            ) from exc

        if not isinstance(payload_raw, dict):
            raise ValueError(
                f"Explanation row must be an object at {source_path}:{line_index}"
            )

        payload = cast(dict[str, Any], payload_raw)
        if "applicant_id" not in payload:
            raise ValueError(
                f"Explanation row missing applicant_id at {source_path}:{line_index}"
            )
        if "predicted_probability" not in payload:
            raise ValueError(
                f"Explanation row missing predicted_probability at {source_path}:{line_index}"
            )
        if "threshold" not in payload:
            raise ValueError(
                f"Explanation row missing threshold at {source_path}:{line_index}"
            )

        method_raw = payload.get("explanation_method")
        method_source: ExplanationMethodSource
        if method_raw in (METHOD_SOURCE_SHAP, METHOD_SOURCE_LIME):
            method_source = cast(ExplanationMethodSource, method_raw)
        else:
            method_source = fallback_source

        case = ExplanationCase(
            applicant_id=_normalize_applicant_id(payload["applicant_id"]),
            explanation_method_source=method_source,
            cohort_name=(
                str(payload["cohort_name"])
                if payload.get("cohort_name") is not None
                else None
            ),
            predicted_probability=_to_float(
                payload["predicted_probability"],
                "predicted_probability",
            ),
            predicted_label=_to_optional_int(payload.get("predicted_label")),
            actual_label=_to_optional_int(payload.get("actual_label")),
            threshold=_to_float(payload["threshold"], "threshold"),
            top_risk_increasing_features=_normalize_feature_rows(
                payload.get("top_risk_increasing_features")
            ),
            top_risk_decreasing_features=_normalize_feature_rows(
                payload.get("top_risk_decreasing_features")
            ),
            source_explanation_generated=bool(payload.get("explanation_generated", True)),
            source_failure_reason=(
                str(payload["failure_reason"])
                if payload.get("failure_reason") is not None
                else None
            ),
        )
        cases.append(case)

    if not cases:
        raise ValueError(f"No explanation rows were found in {source_path}")
    return cases


def _check_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing_paths = [path for path in paths if path.exists()]
    if existing_paths:
        raise FileExistsError(
            "LLM reporting artifacts already exist. Use overwrite=True to replace them. "
            f"Existing: {[str(path) for path in existing_paths]}"
        )


def _report_directory_for_type(paths: LLMReportArtifactPaths, report_type: ReportType) -> Path:
    if report_type == REPORT_TYPE_PLAIN:
        return paths.plain_language_dir
    if report_type == REPORT_TYPE_UNDERWRITER:
        return paths.underwriter_dir
    if report_type == REPORT_TYPE_ADVERSE_ACTION:
        return paths.adverse_action_dir
    raise ValueError(f"Unsupported report type: {report_type}")


def _supporting_features(case: ExplanationCase) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_row in case.top_risk_increasing_features[:5]:
        rows.append(
            {
                "direction": "risk_increasing",
                "feature_name": feature_row.get("feature_name"),
                "contribution": feature_row.get("contribution"),
            }
        )
    for feature_row in case.top_risk_decreasing_features[:5]:
        rows.append(
            {
                "direction": "risk_decreasing",
                "feature_name": feature_row.get("feature_name"),
                "contribution": feature_row.get("contribution"),
            }
        )
    return rows


def _make_report_id(
    *,
    applicant_id: int | str,
    report_type: ReportType,
    explanation_method_source: ExplanationMethodSource,
    ordinal: int,
) -> str:
    token = f"{applicant_id}|{report_type}|{explanation_method_source}|{ordinal}"
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
    return f"rr_{digest}"


def _build_case_markdown_lines(
    *,
    report_id: str,
    case: ExplanationCase,
    report_type: ReportType,
    generated_text: str,
    fallback_generated: bool,
    failure_reason: str | None,
    llm_model: str,
    disclaimer: str,
) -> list[str]:
    return [
        f"- Report ID: {report_id}",
        f"- Report type: {report_type}",
        f"- Applicant ID: {case.applicant_id}",
        f"- Method source: {case.explanation_method_source}",
        f"- Cohort: {case.cohort_name}",
        f"- Predicted probability: {case.predicted_probability:.6f}",
        f"- Threshold: {case.threshold:.6f}",
        f"- Predicted label: {case.predicted_label}",
        f"- Actual label: {case.actual_label}",
        f"- LLM model: {llm_model}",
        f"- Fallback generated: {fallback_generated}",
        f"- Failure reason: {failure_reason}",
        "",
        "## Narrative",
        generated_text,
        "",
        f"Disclaimer: {disclaimer}",
    ]


def run_llm_reporting_workflow(
    settings: Settings,
    *,
    report_type_selection: ReportTypeSelection,
    method_source_selection: ExplanationMethodSourceSelection,
    model_name_override: str | None = None,
    limit: int | None = None,
    input_path_override: Path | None = None,
    overwrite: bool = False,
) -> LLMReportingWorkflowResult:
    """Generate local risk narratives from structured explainability payloads."""
    resolved_limit = int(limit) if limit is not None else settings.llm_reports_max_cases
    if resolved_limit <= 0:
        raise ValueError("limit must be a positive integer")

    report_types = _resolve_report_types(report_type_selection)
    source_path, method_source_used = _resolve_input_source(
        settings,
        method_source_selection=method_source_selection,
        input_path_override=input_path_override,
    )
    cases = _load_explanation_cases(
        source_path=source_path,
        fallback_source=method_source_used,
    )
    selected_cases = cases[:resolved_limit]

    paths = resolve_llm_report_artifact_paths(settings)
    reports_jsonl_path = paths.combined_dir / LLM_REPORTS_JSONL_FILE
    reports_csv_path = paths.combined_dir / LLM_REPORTS_CSV_FILE
    summary_path = paths.reports_dir / LLM_REPORTING_SUMMARY_FILE

    _check_overwrite(
        [reports_jsonl_path, reports_csv_path, summary_path],
        overwrite=overwrite,
    )

    resolved_model_name = (
        str(model_name_override).strip()
        if model_name_override is not None
        else settings.llm_reports_model_name
    )
    if not resolved_model_name:
        raise ValueError("Resolved LLM report model name is empty")

    client = OllamaClient(
        base_url=str(settings.ollama_base_url),
        model=resolved_model_name,
        timeout_seconds=float(settings.llm_reports_timeout_seconds),
        max_retries=int(settings.llm_reports_retries),
    )

    health = client.healthcheck(required_model=resolved_model_name)
    global_fallback_reason: str | None = None
    if not health.reachable:
        global_fallback_reason = f"Ollama unavailable: {health.failure_reason}"
    elif health.failure_reason is not None:
        global_fallback_reason = health.failure_reason

    if global_fallback_reason is not None:
        if not settings.llm_reports_enable_fallback:
            raise RuntimeError(
                "LLM reporting requires an available local Ollama model when fallback is disabled: "
                f"{global_fallback_reason}"
            )
        LOGGER.warning(
            "Using deterministic fallback narratives for all cases: %s",
            global_fallback_reason,
        )

    LOGGER.info(
        "Starting LLM reporting workflow: source=%s report_types=%s cases=%d model=%s",
        source_path,
        report_types,
        len(selected_cases),
        resolved_model_name,
    )

    generation_timestamp = datetime.now(tz=UTC).isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    llm_generated_reports = 0
    fallback_generated_reports = 0

    for case in selected_cases:
        for report_type in report_types:
            report_id = _make_report_id(
                applicant_id=case.applicant_id,
                report_type=report_type,
                explanation_method_source=case.explanation_method_source,
                ordinal=len(rows) + 1,
            )

            failure_reason: str | None = None
            fallback_generated = False
            generated_text: str

            if not case.source_explanation_generated:
                fallback_generated = True
                failure_reason = (
                    "Source explainability payload indicates explanation was not generated: "
                    f"{case.source_failure_reason or 'unknown reason'}"
                )
                generated_text = render_fallback_report(
                    report_type=report_type,
                    applicant_id=case.applicant_id,
                    explanation_method_source=case.explanation_method_source,
                    cohort_name=case.cohort_name,
                    predicted_probability=case.predicted_probability,
                    threshold=case.threshold,
                    predicted_label=case.predicted_label,
                    actual_label=case.actual_label,
                    top_risk_increasing_features=case.top_risk_increasing_features,
                    top_risk_decreasing_features=case.top_risk_decreasing_features,
                    source_explanation_generated=case.source_explanation_generated,
                    source_failure_reason=case.source_failure_reason,
                    generation_failure_reason=failure_reason,
                    disclaimer=DRAFT_DISCLAIMER,
                )
            elif global_fallback_reason is not None:
                fallback_generated = True
                failure_reason = global_fallback_reason
                generated_text = render_fallback_report(
                    report_type=report_type,
                    applicant_id=case.applicant_id,
                    explanation_method_source=case.explanation_method_source,
                    cohort_name=case.cohort_name,
                    predicted_probability=case.predicted_probability,
                    threshold=case.threshold,
                    predicted_label=case.predicted_label,
                    actual_label=case.actual_label,
                    top_risk_increasing_features=case.top_risk_increasing_features,
                    top_risk_decreasing_features=case.top_risk_decreasing_features,
                    source_explanation_generated=case.source_explanation_generated,
                    source_failure_reason=case.source_failure_reason,
                    generation_failure_reason=failure_reason,
                    disclaimer=DRAFT_DISCLAIMER,
                )
            else:
                prompt = build_report_prompt(
                    report_type=report_type,
                    applicant_id=case.applicant_id,
                    explanation_method_source=case.explanation_method_source,
                    cohort_name=case.cohort_name,
                    predicted_probability=case.predicted_probability,
                    predicted_label=case.predicted_label,
                    actual_label=case.actual_label,
                    threshold=case.threshold,
                    top_risk_increasing_features=case.top_risk_increasing_features,
                    top_risk_decreasing_features=case.top_risk_decreasing_features,
                    source_explanation_generated=case.source_explanation_generated,
                    source_failure_reason=case.source_failure_reason,
                    disclaimer=DRAFT_DISCLAIMER,
                    prompt_version=PROMPT_VERSION,
                )
                try:
                    llm_text = client.generate(prompt=prompt, model=resolved_model_name)
                    generated_text = normalize_generated_text(llm_text)
                    if not generated_text:
                        raise OllamaClientError("Generated text is empty after normalization")
                except Exception as exc:
                    if not settings.llm_reports_enable_fallback:
                        raise
                    fallback_generated = True
                    failure_reason = f"Ollama generation failed: {type(exc).__name__}: {exc}"
                    generated_text = render_fallback_report(
                        report_type=report_type,
                        applicant_id=case.applicant_id,
                        explanation_method_source=case.explanation_method_source,
                        cohort_name=case.cohort_name,
                        predicted_probability=case.predicted_probability,
                        threshold=case.threshold,
                        predicted_label=case.predicted_label,
                        actual_label=case.actual_label,
                        top_risk_increasing_features=case.top_risk_increasing_features,
                        top_risk_decreasing_features=case.top_risk_decreasing_features,
                        source_explanation_generated=case.source_explanation_generated,
                        source_failure_reason=case.source_failure_reason,
                        generation_failure_reason=failure_reason,
                        disclaimer=DRAFT_DISCLAIMER,
                    )

            if fallback_generated:
                fallback_generated_reports += 1
            else:
                llm_generated_reports += 1

            output_dir = _report_directory_for_type(paths, report_type)
            case_file = output_dir / (
                f"{report_type}_{_safe_file_token(str(case.applicant_id))}_{report_id}.md"
            )
            write_markdown(
                case_file,
                title="Risk Narrative Report",
                lines=_build_case_markdown_lines(
                    report_id=report_id,
                    case=case,
                    report_type=report_type,
                    generated_text=generated_text,
                    fallback_generated=fallback_generated,
                    failure_reason=failure_reason,
                    llm_model=resolved_model_name,
                    disclaimer=DRAFT_DISCLAIMER,
                ),
            )

            rows.append(
                {
                    "report_id": report_id,
                    "applicant_id": case.applicant_id,
                    "report_type": report_type,
                    "explanation_method_source": case.explanation_method_source,
                    "cohort_name": case.cohort_name,
                    "predicted_probability": case.predicted_probability,
                    "predicted_label": case.predicted_label,
                    "actual_label": case.actual_label,
                    "threshold": case.threshold,
                    "supporting_features": _supporting_features(case),
                    "generated_text": generated_text,
                    "fallback_generated": fallback_generated,
                    "failure_reason": failure_reason,
                    "llm_model": resolved_model_name,
                    "model_name": resolved_model_name,
                    "prompt_version": PROMPT_VERSION,
                    "artifact_version": ARTIFACT_VERSION,
                    "explanation_generated": case.source_explanation_generated,
                    "source_failure_reason": case.source_failure_reason,
                    "source_artifact_path": str(source_path),
                    "generation_timestamp": generation_timestamp,
                    "disclaimer": DRAFT_DISCLAIMER,
                }
            )

    reports_jsonl_path = write_jsonl(reports_jsonl_path, rows)
    reports_csv_path = write_csv(reports_csv_path, rows)

    summary_lines = [
        "## Configuration",
        f"- Source artifact: {source_path}",
        f"- Method source used: {method_source_used}",
        f"- Report types: {', '.join(report_types)}",
        f"- Case limit: {resolved_limit}",
        f"- Cases processed: {len(selected_cases)}",
        f"- Model name: {resolved_model_name}",
        f"- Fallback enabled: {settings.llm_reports_enable_fallback}",
        "",
        "## Generation Results",
        f"- Total reports: {len(rows)}",
        f"- LLM generated: {llm_generated_reports}",
        f"- Fallback generated: {fallback_generated_reports}",
        (
            "- Global fallback reason: "
            f"{global_fallback_reason if global_fallback_reason is not None else 'none'}"
        ),
        "",
        "## Artifacts",
        f"- JSONL: {reports_jsonl_path}",
        f"- CSV: {reports_csv_path}",
        f"- Plain-language markdown dir: {paths.plain_language_dir}",
        f"- Underwriter markdown dir: {paths.underwriter_dir}",
        f"- Adverse-action markdown dir: {paths.adverse_action_dir}",
        f"- Disclaimer: {DRAFT_DISCLAIMER}",
    ]
    summary_path = write_markdown(
        summary_path,
        title="LLM Reporting Summary",
        lines=summary_lines,
    )

    LOGGER.info(
        "LLM reporting completed. total_reports=%d llm_generated=%d fallback=%d summary=%s",
        len(rows),
        llm_generated_reports,
        fallback_generated_reports,
        summary_path,
    )

    return LLMReportingWorkflowResult(
        source_artifact_path=source_path,
        method_source_used=method_source_used,
        reports_jsonl_path=reports_jsonl_path,
        reports_csv_path=reports_csv_path,
        reporting_summary_path=summary_path,
        total_reports=len(rows),
        llm_generated_reports=llm_generated_reports,
        fallback_generated_reports=fallback_generated_reports,
    )
