"""Model registry and best-model selection helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from pandas import DataFrame


def select_best_model(
    model_comparison: DataFrame,
    *,
    primary_metric: str,
    threshold: float,
    folds: int,
    random_seed: int,
) -> dict[str, object]:
    """Select the best model by primary metric and return a machine-readable summary."""
    if model_comparison.empty:
        raise ValueError("Model comparison dataframe is empty")
    if primary_metric not in model_comparison.columns:
        raise ValueError(f"Primary metric not found in model comparison: {primary_metric}")

    sorted_comparison = model_comparison.sort_values(
        by=[primary_metric, "pr_auc", "f1"],
        ascending=False,
    ).reset_index(drop=True)
    best_row = sorted_comparison.iloc[0]

    summary = {
        "best_model_name": str(best_row["model_name"]),
        "primary_metric": primary_metric,
        "primary_metric_value": float(best_row[primary_metric]),
        "threshold": float(threshold),
        "folds": int(folds),
        "random_seed": int(random_seed),
        "training_timestamp": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
    return summary
