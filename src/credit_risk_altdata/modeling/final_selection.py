"""Final production-candidate selection helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from pandas import DataFrame


def select_final_candidate(
    candidate_comparison: DataFrame,
    *,
    primary_metric: str,
    source_comparison_artifact: str | None = None,
) -> dict[str, object]:
    """Select the final production candidate by primary metric and tie-breakers."""
    required_columns = {
        "candidate_name",
        "model_family",
        "is_tuned",
        "is_calibrated",
        "calibration_method",
        "threshold",
        "artifact_path",
        "pr_auc",
        "f1",
        primary_metric,
    }
    missing_columns = sorted(required_columns.difference(candidate_comparison.columns))
    if missing_columns:
        raise ValueError(f"Candidate comparison missing required columns: {missing_columns}")
    if candidate_comparison.empty:
        raise ValueError("Candidate comparison is empty")

    sorted_candidates = candidate_comparison.sort_values(
        by=[primary_metric, "pr_auc", "f1"],
        ascending=False,
    ).reset_index(drop=True)
    best_row = sorted_candidates.iloc[0]

    metric_value = float(best_row[primary_metric])
    justification = (
        f"Selected highest {primary_metric} candidate "
        f"({best_row['candidate_name']}, {metric_value:.6f}) "
        "with tie-breakers on PR AUC and F1."
    )
    summary = {
        "final_model_family": str(best_row["model_family"]),
        "final_candidate_name": str(best_row["candidate_name"]),
        "tuned": bool(best_row["is_tuned"]),
        "calibrated": bool(best_row["is_calibrated"]),
        "calibration_method": str(best_row["calibration_method"]),
        "primary_metric": primary_metric,
        "primary_metric_value": metric_value,
        "threshold": float(best_row["threshold"]),
        "selected_artifact_path": str(best_row["artifact_path"]),
        "training_timestamp": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "justification": justification,
        "brief_justification": justification,
        "source_comparison_artifact": source_comparison_artifact,
    }
    return summary
