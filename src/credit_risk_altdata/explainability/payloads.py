"""Payload-shaping helpers for explainability artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from credit_risk_altdata.explainability.constants import CohortName, ExplanationMethod


def _sorted_positive_contributions(
    feature_contributions: Mapping[str, float],
) -> list[tuple[str, float]]:
    positive = [
        (name, float(value))
        for name, value in feature_contributions.items()
        if value > 0.0
    ]
    return sorted(positive, key=lambda item: (-item[1], item[0]))


def _sorted_negative_contributions(
    feature_contributions: Mapping[str, float],
) -> list[tuple[str, float]]:
    negative = [
        (name, float(value))
        for name, value in feature_contributions.items()
        if value < 0.0
    ]
    return sorted(negative, key=lambda item: (item[1], item[0]))


def _key_feature_contributions(
    feature_contributions: Mapping[str, float],
    *,
    top_k: int,
) -> list[dict[str, float | str]]:
    ranked = sorted(
        ((name, float(value)) for name, value in feature_contributions.items()),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    limit = max(top_k * 2, top_k)
    return [
        {
            "feature_name": name,
            "contribution": value,
        }
        for name, value in ranked[:limit]
    ]


def build_local_explanation_payload(
    *,
    explanation_method: ExplanationMethod,
    applicant_id: int | str,
    cohort_name: CohortName,
    split_name: str,
    predicted_probability: float,
    predicted_label: int,
    actual_label: int | None,
    threshold: float,
    feature_contributions: Mapping[str, float],
    top_k: int,
    metadata: Mapping[str, Any],
    explanation_generated: bool = True,
    failure_reason: str | None = None,
    failed_feature_count: int = 0,
) -> dict[str, Any]:
    """Build a machine-readable local explanation payload."""
    top_positive = _sorted_positive_contributions(feature_contributions)
    top_negative = _sorted_negative_contributions(feature_contributions)

    payload = {
        "explanation_method": explanation_method,
        "applicant_id": int(applicant_id) if str(applicant_id).isdigit() else str(applicant_id),
        "split_name": split_name,
        "cohort_name": cohort_name,
        "predicted_probability": float(predicted_probability),
        "predicted_label": int(predicted_label),
        "actual_label": int(actual_label) if actual_label is not None else None,
        "threshold": float(threshold),
        "explanation_generated": bool(explanation_generated),
        "failure_reason": str(failure_reason) if failure_reason is not None else None,
        "failed_feature_count": int(failed_feature_count),
        "top_risk_increasing_features": [
            {
                "feature_name": name,
                "contribution": value,
            }
            for name, value in top_positive[:top_k]
        ],
        "top_risk_decreasing_features": [
            {
                "feature_name": name,
                "contribution": value,
            }
            for name, value in top_negative[:top_k]
        ],
        "feature_contributions": _key_feature_contributions(
            feature_contributions,
            top_k=top_k,
        ),
        "metadata": {
            **{key: value for key, value in metadata.items()},
            "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        },
    }
    return payload
