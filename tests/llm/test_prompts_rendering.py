"""Tests for prompt construction and deterministic fallback rendering."""

from __future__ import annotations

from credit_risk_altdata.llm.prompts import build_report_prompt
from credit_risk_altdata.llm.rendering import normalize_generated_text, render_fallback_report


def test_build_report_prompt_includes_evidence_and_guardrails() -> None:
    prompt = build_report_prompt(
        report_type="plain",
        applicant_id=700001,
        explanation_method_source="shap",
        cohort_name="false_positive",
        predicted_probability=0.72,
        predicted_label=1,
        actual_label=0,
        threshold=0.5,
        top_risk_increasing_features=[
            {"feature_name": "feature_a", "contribution": 0.21},
            {"feature_name": "feature_b", "contribution": 0.11},
        ],
        top_risk_decreasing_features=[
            {"feature_name": "feature_c", "contribution": -0.09},
        ],
        source_explanation_generated=True,
        source_failure_reason=None,
    )

    assert "Do not invent features" in prompt
    assert "applicant_id: 700001" in prompt
    assert "predicted_probability: 0.720000" in prompt
    assert "feature_a" in prompt
    assert "feature_c" in prompt


def test_render_fallback_report_plain_and_underwriter() -> None:
    plain_text = render_fallback_report(
        report_type="plain",
        applicant_id=700123,
        explanation_method_source="lime",
        cohort_name="true_negative",
        predicted_probability=0.31,
        threshold=0.5,
        predicted_label=0,
        actual_label=0,
        top_risk_increasing_features=[{"feature_name": "feature_1", "contribution": 0.1}],
        top_risk_decreasing_features=[{"feature_name": "feature_2", "contribution": -0.2}],
        source_explanation_generated=True,
        source_failure_reason=None,
        generation_failure_reason="Ollama unavailable",
    )
    assert "deterministic fallback" in plain_text
    assert "feature_1" in plain_text
    assert "feature_2" in plain_text

    underwriter_text = render_fallback_report(
        report_type="underwriter",
        applicant_id=700123,
        explanation_method_source="lime",
        cohort_name="true_negative",
        predicted_probability=0.31,
        threshold=0.5,
        predicted_label=0,
        actual_label=0,
        top_risk_increasing_features=[{"feature_name": "feature_1", "contribution": 0.1}],
        top_risk_decreasing_features=[{"feature_name": "feature_2", "contribution": -0.2}],
        source_explanation_generated=False,
        source_failure_reason="source failed",
        generation_failure_reason="Ollama unavailable",
    )
    assert underwriter_text.splitlines()[0].startswith("- Applicant ID:")
    assert "fallback" in underwriter_text


def test_normalize_generated_text_strips_blank_lines() -> None:
    text = "\n\n Hello there.\n\n  This is a test.  \n\n"
    normalized = normalize_generated_text(text)
    assert normalized == "Hello there.\nThis is a test."
