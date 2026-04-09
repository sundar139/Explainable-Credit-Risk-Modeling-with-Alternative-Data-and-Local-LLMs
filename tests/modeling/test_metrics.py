"""Tests for baseline metric utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk_altdata.modeling.metrics import (
    compute_classification_metrics,
    summarize_fold_metrics,
)


def test_compute_classification_metrics_returns_expected_keys() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_prob = np.array([0.1, 0.4, 0.6, 0.9], dtype=float)

    metrics = compute_classification_metrics(y_true=y_true, y_prob=y_prob, threshold=0.5)

    expected_keys = {
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "threshold",
        "tn",
        "fp",
        "fn",
        "tp",
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["tn"] == 2.0
    assert metrics["tp"] == 2.0
    assert metrics["fp"] == 0.0
    assert metrics["fn"] == 0.0


def test_summarize_fold_metrics_groups_by_model() -> None:
    fold_metrics = pd.DataFrame(
        [
            {
                "model_name": "lightgbm",
                "fold": "1",
                "roc_auc": 0.71,
                "pr_auc": 0.45,
                "precision": 0.5,
                "recall": 0.4,
                "f1": 0.44,
                "accuracy": 0.7,
                "tn": 30,
                "fp": 10,
                "fn": 12,
                "tp": 8,
            },
            {
                "model_name": "lightgbm",
                "fold": "2",
                "roc_auc": 0.73,
                "pr_auc": 0.47,
                "precision": 0.52,
                "recall": 0.42,
                "f1": 0.46,
                "accuracy": 0.72,
                "tn": 31,
                "fp": 9,
                "fn": 11,
                "tp": 9,
            },
            {
                "model_name": "catboost",
                "fold": "1",
                "roc_auc": 0.75,
                "pr_auc": 0.50,
                "precision": 0.55,
                "recall": 0.44,
                "f1": 0.49,
                "accuracy": 0.74,
                "tn": 32,
                "fp": 8,
                "fn": 10,
                "tp": 10,
            },
        ]
    )

    summary = summarize_fold_metrics(fold_metrics)

    assert "model_name" in summary.columns
    assert "roc_auc_mean" in summary.columns
    assert "roc_auc_std" in summary.columns
    assert set(summary["model_name"]) == {"lightgbm", "catboost"}
