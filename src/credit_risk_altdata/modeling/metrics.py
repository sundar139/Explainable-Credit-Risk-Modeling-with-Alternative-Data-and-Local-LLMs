"""Metric calculation helpers for baseline modeling."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(callable_obj: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
    try:
        return float(callable_obj(*args, **kwargs))
    except ValueError:
        return float("nan")


def compute_classification_metrics(
    *,
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    threshold: float,
) -> dict[str, float]:
    """Compute baseline classification metrics at a fixed probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = _safe_metric(roc_auc_score, y_true, y_prob)
    pr_auc = _safe_metric(average_precision_score, y_true, y_prob)
    precision = _safe_metric(precision_score, y_true, y_pred, zero_division=0)
    recall = _safe_metric(recall_score, y_true, y_pred, zero_division=0)
    f1 = _safe_metric(f1_score, y_true, y_pred, zero_division=0)
    accuracy = _safe_metric(accuracy_score, y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "threshold": float(threshold),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def summarize_fold_metrics(fold_metrics: DataFrame) -> DataFrame:
    """Summarize fold metrics by model using mean and std statistics."""
    metric_columns = [
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    grouped = fold_metrics.groupby("model_name", as_index=False)[metric_columns].agg(
        ["mean", "std"]
    )
    flattened_columns: list[str] = []
    for column in grouped.columns:
        if isinstance(column, tuple):
            if column[0] == "model_name":
                flattened_columns.append("model_name")
            else:
                flattened_columns.append(f"{column[0]}_{column[1]}")
        else:
            flattened_columns.append(str(column))
    grouped.columns = flattened_columns
    return grouped
