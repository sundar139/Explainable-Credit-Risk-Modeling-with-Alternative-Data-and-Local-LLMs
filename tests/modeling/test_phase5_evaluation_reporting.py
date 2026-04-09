"""Tests for Phase 5 evaluation reporting utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from credit_risk_altdata.config import Settings
from credit_risk_altdata.modeling.evaluation_reporting import (
    build_threshold_grid,
    generate_evaluation_artifacts,
    generate_threshold_analysis,
)


def test_generate_threshold_analysis_contains_expected_columns() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0], dtype=int)
    y_prob = np.array([0.1, 0.8, 0.3, 0.75, 0.6, 0.2], dtype=float)
    thresholds = np.array([0.3, 0.5, 0.7], dtype=float)

    threshold_table = generate_threshold_analysis(
        y_true=y_true,
        y_prob=y_prob,
        thresholds=thresholds,
    )

    expected_columns = {
        "threshold",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "positive_prediction_rate",
    }
    assert expected_columns.issubset(set(threshold_table.columns))
    assert threshold_table.shape[0] == 3


def test_generate_evaluation_artifacts_writes_required_outputs(tmp_path: Path) -> None:
    y_true = np.array([0, 1] * 40, dtype=int)
    y_prob = np.linspace(0.05, 0.95, num=80, dtype=float)

    settings = Settings(app_env="test", project_root=tmp_path)
    thresholds = build_threshold_grid(settings)
    summary, artifact_paths = generate_evaluation_artifacts(
        y_true=y_true,
        y_prob=y_prob,
        thresholds=thresholds,
        evaluation_dir=tmp_path / "artifacts" / "modeling" / "evaluation",
    )

    assert 0.0 <= float(summary["roc_auc"]) <= 1.0
    assert 0.0 <= float(summary["pr_auc"]) <= 1.0
    for path in artifact_paths.values():
        assert path.exists()
