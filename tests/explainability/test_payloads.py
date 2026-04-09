"""Tests for local explanation payload formatting."""

from __future__ import annotations

from credit_risk_altdata.explainability.payloads import build_local_explanation_payload


def test_build_local_explanation_payload_sorts_and_limits_contributions() -> None:
    payload = build_local_explanation_payload(
        explanation_method="shap",
        applicant_id="123456",
        cohort_name="false_positive",
        split_name="train_oof",
        predicted_probability=0.84,
        predicted_label=1,
        actual_label=0,
        threshold=0.5,
        feature_contributions={
            "feature_a": 0.6,
            "feature_b": -0.5,
            "feature_c": 0.3,
            "feature_d": -0.2,
            "feature_e": 0.1,
        },
        top_k=2,
        metadata={"model_family": "lightgbm"},
    )

    assert payload["explanation_method"] == "shap"
    assert payload["applicant_id"] == 123456
    assert payload["cohort_name"] == "false_positive"

    assert [row["feature_name"] for row in payload["top_risk_increasing_features"]] == [
        "feature_a",
        "feature_c",
    ]
    assert [row["feature_name"] for row in payload["top_risk_decreasing_features"]] == [
        "feature_b",
        "feature_d",
    ]

    assert len(payload["feature_contributions"]) == 4
    assert payload["feature_contributions"][0]["feature_name"] == "feature_a"
    assert payload["explanation_generated"] is True
    assert payload["failure_reason"] is None
    assert payload["failed_feature_count"] == 0
    assert payload["metadata"]["model_family"] == "lightgbm"
    assert "generated_at" in payload["metadata"]


def test_build_local_explanation_payload_preserves_non_numeric_applicant_id() -> None:
    payload = build_local_explanation_payload(
        explanation_method="lime",
        applicant_id="A-1001",
        cohort_name="borderline_threshold",
        split_name="train_oof",
        predicted_probability=0.49,
        predicted_label=0,
        actual_label=None,
        threshold=0.5,
        feature_contributions={"rule_x": -0.12},
        top_k=3,
        metadata={},
    )

    assert payload["applicant_id"] == "A-1001"
    assert payload["actual_label"] is None
    assert payload["top_risk_increasing_features"] == []
    assert payload["top_risk_decreasing_features"][0]["feature_name"] == "rule_x"


def test_build_local_explanation_payload_supports_structured_failure_fields() -> None:
    payload = build_local_explanation_payload(
        explanation_method="lime",
        applicant_id=700001,
        cohort_name="true_positive",
        split_name="train_oof",
        predicted_probability=0.88,
        predicted_label=1,
        actual_label=1,
        threshold=0.5,
        feature_contributions={},
        top_k=5,
        metadata={"model_family": "lightgbm"},
        explanation_generated=False,
        failure_reason="ValueError: synthetic truncnorm scale failure",
        failed_feature_count=12,
    )

    assert payload["explanation_generated"] is False
    assert "truncnorm" in str(payload["failure_reason"])
    assert payload["failed_feature_count"] == 12
    assert payload["feature_contributions"] == []
