"""Representative-example selection logic for explainability."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame

from credit_risk_altdata.explainability.constants import (
    COHORT_BORDERLINE,
    COHORT_FALSE_NEGATIVE,
    COHORT_FALSE_POSITIVE,
    COHORT_TRUE_NEGATIVE,
    COHORT_TRUE_POSITIVE,
)


def build_prediction_frame(
    *,
    applicant_ids: Sequence[int],
    actual_labels: Sequence[int],
    predicted_probabilities: Sequence[float],
    threshold: float,
    split_name: str = "train_oof",
) -> DataFrame:
    """Construct normalized prediction frame for cohort selection."""
    if not (len(applicant_ids) == len(actual_labels) == len(predicted_probabilities)):
        raise ValueError("Applicant IDs, labels, and probabilities must have the same length")

    frame = pd.DataFrame(
        {
            "applicant_id": np.asarray(applicant_ids, dtype=int),
            "actual_label": np.asarray(actual_labels, dtype=int),
            "predicted_probability": np.asarray(predicted_probabilities, dtype=float),
            "split_name": split_name,
        }
    )
    frame["predicted_label"] = (frame["predicted_probability"] >= float(threshold)).astype(int)
    frame["distance_to_threshold"] = (
        frame["predicted_probability"] - float(threshold)
    ).abs()
    return frame


def _select_cohort(
    *,
    frame: DataFrame,
    cohort_name: str,
    mask: pd.Series,
    sort_by: list[str],
    ascending: list[bool],
    count: int,
    selected_applicant_ids: set[int],
) -> DataFrame:
    if count <= 0:
        return pd.DataFrame(columns=frame.columns.tolist() + ["cohort_name", "selection_rank"])

    available = frame[mask & ~frame["applicant_id"].isin(selected_applicant_ids)].copy()
    if available.empty:
        return pd.DataFrame(columns=frame.columns.tolist() + ["cohort_name", "selection_rank"])

    selected = available.sort_values(sort_by, ascending=ascending).head(count).copy()
    selected["cohort_name"] = cohort_name
    selected["selection_rank"] = range(1, len(selected) + 1)
    selected_applicant_ids.update(selected["applicant_id"].astype(int).tolist())
    return selected


def select_representative_examples(
    *,
    prediction_frame: DataFrame,
    threshold: float,
    true_positive_count: int,
    true_negative_count: int,
    false_positive_count: int,
    false_negative_count: int,
    borderline_count: int,
) -> DataFrame:
    """Select deterministic cohorts for local explainability."""
    required_columns = {
        "applicant_id",
        "actual_label",
        "predicted_probability",
        "predicted_label",
        "split_name",
    }
    missing_columns = sorted(required_columns.difference(prediction_frame.columns))
    if missing_columns:
        raise ValueError(f"Prediction frame missing required columns: {missing_columns}")

    frame = prediction_frame.copy()
    frame["applicant_id"] = frame["applicant_id"].astype(int)
    frame["actual_label"] = frame["actual_label"].astype(int)
    frame["predicted_label"] = frame["predicted_label"].astype(int)
    frame["predicted_probability"] = frame["predicted_probability"].astype(float)
    frame["distance_to_threshold"] = (
        frame["predicted_probability"] - float(threshold)
    ).abs()

    selected_ids: set[int] = set()
    selected_parts: list[DataFrame] = []

    selected_parts.append(
        _select_cohort(
            frame=frame,
            cohort_name=COHORT_TRUE_POSITIVE,
            mask=(frame["actual_label"] == 1) & (frame["predicted_label"] == 1),
            sort_by=["predicted_probability", "applicant_id"],
            ascending=[False, True],
            count=true_positive_count,
            selected_applicant_ids=selected_ids,
        )
    )
    selected_parts.append(
        _select_cohort(
            frame=frame,
            cohort_name=COHORT_TRUE_NEGATIVE,
            mask=(frame["actual_label"] == 0) & (frame["predicted_label"] == 0),
            sort_by=["predicted_probability", "applicant_id"],
            ascending=[True, True],
            count=true_negative_count,
            selected_applicant_ids=selected_ids,
        )
    )
    selected_parts.append(
        _select_cohort(
            frame=frame,
            cohort_name=COHORT_FALSE_POSITIVE,
            mask=(frame["actual_label"] == 0) & (frame["predicted_label"] == 1),
            sort_by=["predicted_probability", "applicant_id"],
            ascending=[False, True],
            count=false_positive_count,
            selected_applicant_ids=selected_ids,
        )
    )
    selected_parts.append(
        _select_cohort(
            frame=frame,
            cohort_name=COHORT_FALSE_NEGATIVE,
            mask=(frame["actual_label"] == 1) & (frame["predicted_label"] == 0),
            sort_by=["predicted_probability", "applicant_id"],
            ascending=[True, True],
            count=false_negative_count,
            selected_applicant_ids=selected_ids,
        )
    )
    selected_parts.append(
        _select_cohort(
            frame=frame,
            cohort_name=COHORT_BORDERLINE,
            mask=pd.Series(True, index=frame.index),
            sort_by=["distance_to_threshold", "applicant_id"],
            ascending=[True, True],
            count=borderline_count,
            selected_applicant_ids=selected_ids,
        )
    )

    selected = pd.concat(selected_parts, axis=0, ignore_index=True)
    if selected.empty:
        return selected

    ordered = selected.sort_values(
        ["cohort_name", "selection_rank", "applicant_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return ordered[
        [
            "applicant_id",
            "split_name",
            "cohort_name",
            "actual_label",
            "predicted_label",
            "predicted_probability",
            "distance_to_threshold",
            "selection_rank",
        ]
    ]
