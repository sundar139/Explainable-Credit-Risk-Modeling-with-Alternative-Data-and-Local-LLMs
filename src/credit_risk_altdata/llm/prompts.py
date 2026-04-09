"""Prompt templates for local risk-report generation."""

from __future__ import annotations

from typing import Any

from credit_risk_altdata.llm.constants import (
    DRAFT_DISCLAIMER,
    PROMPT_VERSION,
    REPORT_TYPE_ADVERSE_ACTION,
    REPORT_TYPE_PLAIN,
    REPORT_TYPE_UNDERWRITER,
    ReportType,
)


def _render_feature_evidence(
    rows: list[dict[str, Any]],
    *,
    max_rows: int,
) -> list[str]:
    if not rows:
        return ["- none"]

    lines: list[str] = []
    for row in rows[:max_rows]:
        feature_name = str(row.get("feature_name", "unknown_feature"))
        contribution_raw = row.get("contribution")
        if isinstance(contribution_raw, bool):
            contribution = float(int(contribution_raw))
            lines.append(f"- {feature_name} (contribution={contribution:.6f})")
            continue
        if isinstance(contribution_raw, (int, float, str)):
            try:
                contribution = float(contribution_raw)
                lines.append(f"- {feature_name} (contribution={contribution:.6f})")
                continue
            except ValueError:
                pass
        if contribution_raw is None:
            lines.append(f"- {feature_name}")
        else:
            lines.append(f"- {feature_name}")
    return lines


def _style_instruction(report_type: ReportType) -> str:
    if report_type == REPORT_TYPE_PLAIN:
        return (
            "Write a clear plain-language explanation in 3 to 5 short sentences. "
            "Avoid jargon where possible."
        )
    if report_type == REPORT_TYPE_UNDERWRITER:
        return (
            "Write a concise underwriter-style summary in up to 5 bullet points. "
            "Focus on factors most relevant to risk assessment."
        )
    if report_type == REPORT_TYPE_ADVERSE_ACTION:
        return (
            "Write an adverse-action-style draft using neutral language in 4 to 6 sentences. "
            "State that this is draft explanatory text for review and not a final legal notice."
        )
    raise ValueError(f"Unsupported report type: {report_type}")


def build_report_prompt(
    *,
    report_type: ReportType,
    applicant_id: int | str,
    explanation_method_source: str,
    cohort_name: str | None,
    predicted_probability: float,
    predicted_label: int | None,
    actual_label: int | None,
    threshold: float,
    top_risk_increasing_features: list[dict[str, Any]],
    top_risk_decreasing_features: list[dict[str, Any]],
    source_explanation_generated: bool,
    source_failure_reason: str | None,
    disclaimer: str = DRAFT_DISCLAIMER,
    prompt_version: str = PROMPT_VERSION,
) -> str:
    """Build a grounded prompt from structured explainability evidence."""
    lines: list[str] = [
        f"Prompt version: {prompt_version}",
        "You are a credit risk analyst writing evidence-grounded narrative text.",
        "",
        "Guardrails:",
        "1. Use only the evidence provided below.",
        "2. Do not invent features, values, policies, or facts.",
        "3. If evidence is missing, explicitly say that details are limited.",
        "4. Keep language neutral, professional, and concise.",
        "5. Do not claim legal compliance or regulatory sufficiency.",
        f"6. Include this disclaimer exactly once: {disclaimer}",
        "",
        "Output style:",
        _style_instruction(report_type),
        "",
        "Case evidence:",
        f"- applicant_id: {applicant_id}",
        f"- explanation_method_source: {explanation_method_source}",
        f"- cohort_name: {cohort_name if cohort_name is not None else 'unknown'}",
        f"- predicted_probability: {predicted_probability:.6f}",
        f"- predicted_label: {predicted_label}",
        f"- actual_label: {actual_label}",
        f"- threshold: {threshold:.6f}",
        f"- source_explanation_generated: {source_explanation_generated}",
        (
            "- source_failure_reason: "
            f"{source_failure_reason if source_failure_reason is not None else 'none'}"
        ),
        "",
        "Top risk-increasing features:",
        *_render_feature_evidence(top_risk_increasing_features, max_rows=6),
        "",
        "Top risk-decreasing features:",
        *_render_feature_evidence(top_risk_decreasing_features, max_rows=6),
        "",
        "Task:",
        (
            f"Generate one {report_type} report for this applicant grounded strictly in "
            "the evidence above."
        ),
    ]

    return "\n".join(lines)
