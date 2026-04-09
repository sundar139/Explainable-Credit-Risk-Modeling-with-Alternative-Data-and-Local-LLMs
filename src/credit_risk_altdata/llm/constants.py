"""Constants for local LLM risk-report generation."""

from __future__ import annotations

from typing import Final, Literal

REPORT_TYPE_ALL: Final[Literal["all"]] = "all"
REPORT_TYPE_PLAIN: Final[Literal["plain"]] = "plain"
REPORT_TYPE_UNDERWRITER: Final[Literal["underwriter"]] = "underwriter"
REPORT_TYPE_ADVERSE_ACTION: Final[Literal["adverse-action"]] = "adverse-action"

ReportTypeSelection = Literal["all", "plain", "underwriter", "adverse-action"]
ReportType = Literal["plain", "underwriter", "adverse-action"]

METHOD_SOURCE_AUTO: Final[Literal["auto"]] = "auto"
METHOD_SOURCE_SHAP: Final[Literal["shap"]] = "shap"
METHOD_SOURCE_LIME: Final[Literal["lime"]] = "lime"

ExplanationMethodSourceSelection = Literal["auto", "shap", "lime"]
ExplanationMethodSource = Literal["shap", "lime"]

SUPPORTED_REPORT_TYPE_SELECTIONS: Final[tuple[ReportTypeSelection, ...]] = (
    REPORT_TYPE_ALL,
    REPORT_TYPE_PLAIN,
    REPORT_TYPE_UNDERWRITER,
    REPORT_TYPE_ADVERSE_ACTION,
)
SUPPORTED_REPORT_TYPES: Final[tuple[ReportType, ...]] = (
    REPORT_TYPE_PLAIN,
    REPORT_TYPE_UNDERWRITER,
    REPORT_TYPE_ADVERSE_ACTION,
)
SUPPORTED_METHOD_SOURCE_SELECTIONS: Final[tuple[ExplanationMethodSourceSelection, ...]] = (
    METHOD_SOURCE_AUTO,
    METHOD_SOURCE_SHAP,
    METHOD_SOURCE_LIME,
)
SUPPORTED_METHOD_SOURCES: Final[tuple[ExplanationMethodSource, ...]] = (
    METHOD_SOURCE_SHAP,
    METHOD_SOURCE_LIME,
)

PROMPT_VERSION: Final[str] = "phase7_v1"
ARTIFACT_VERSION: Final[str] = "phase7_v1"

LLM_REPORTS_JSONL_FILE: Final[str] = "llm_reports.jsonl"
LLM_REPORTS_CSV_FILE: Final[str] = "llm_reports.csv"
LLM_REPORTING_SUMMARY_FILE: Final[str] = "llm_reporting_summary.md"

DRAFT_DISCLAIMER: Final[str] = (
    "Draft narrative output for internal review only. "
    "This text is not legal advice and is not a final adverse action notice."
)
