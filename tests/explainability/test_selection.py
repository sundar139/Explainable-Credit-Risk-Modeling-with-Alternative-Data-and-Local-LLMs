"""Tests for deterministic representative-example selection."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_risk_altdata.explainability.selection import (
    build_prediction_frame,
    select_representative_examples,
)


def test_build_prediction_frame_validates_lengths() -> None:
    with pytest.raises(ValueError):
        build_prediction_frame(
            applicant_ids=[1, 2],
            actual_labels=[0],
            predicted_probabilities=[0.1, 0.9],
            threshold=0.5,
        )


def test_select_representative_examples_is_deterministic_by_cohort() -> None:
    prediction_frame = build_prediction_frame(
        applicant_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        actual_labels=[1, 1, 0, 0, 0, 1, 1, 0, 1, 0],
        predicted_probabilities=[0.95, 0.85, 0.10, 0.20, 0.70, 0.30, 0.60, 0.80, 0.49, 0.51],
        threshold=0.5,
        split_name="train_oof",
    )

    selected = select_representative_examples(
        prediction_frame=prediction_frame,
        threshold=0.5,
        true_positive_count=2,
        true_negative_count=1,
        false_positive_count=2,
        false_negative_count=1,
        borderline_count=2,
    )

    assert len(selected) == 8
    assert selected["applicant_id"].is_unique

    grouped = {
        cohort: sorted(rows["applicant_id"].astype(int).tolist())
        for cohort, rows in selected.groupby("cohort_name")
    }
    assert grouped["true_positive"] == [1, 2]
    assert grouped["true_negative"] == [3]
    assert grouped["false_positive"] == [5, 8]
    assert grouped["false_negative"] == [6]
    assert grouped["borderline_threshold"] == [9, 10]

    borderline_rows = selected[selected["cohort_name"] == "borderline_threshold"]
    assert borderline_rows["distance_to_threshold"].tolist() == pytest.approx([0.01, 0.01])

    expected_columns = [
        "applicant_id",
        "split_name",
        "cohort_name",
        "actual_label",
        "predicted_label",
        "predicted_probability",
        "distance_to_threshold",
        "selection_rank",
    ]
    assert list(selected.columns) == expected_columns


def test_select_representative_examples_requires_expected_columns() -> None:
    prediction_frame = pd.DataFrame(
        {
            "applicant_id": [1],
            "actual_label": [1],
            "predicted_probability": [0.8],
        }
    )

    with pytest.raises(ValueError):
        select_representative_examples(
            prediction_frame=prediction_frame,
            threshold=0.5,
            true_positive_count=1,
            true_negative_count=0,
            false_positive_count=0,
            false_negative_count=0,
            borderline_count=0,
        )
