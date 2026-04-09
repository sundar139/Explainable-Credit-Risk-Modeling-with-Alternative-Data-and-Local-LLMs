"""Tests for Phase 5 calibration candidate evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from credit_risk_altdata.config import Settings
from credit_risk_altdata.modeling.calibration import (
    build_calibration_comparison,
    evaluate_tuned_candidates,
)
from credit_risk_altdata.modeling.constants import (
    CALIBRATION_NONE,
    CALIBRATION_SIGMOID,
    MODEL_LIGHTGBM,
)
from credit_risk_altdata.modeling.data_prep import prepare_modeling_dataset
from credit_risk_altdata.modeling.reporting import resolve_modeling_artifact_paths


def test_evaluate_tuned_candidates_with_sigmoid_calibration(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
) -> None:
    settings = synthetic_settings.model_copy(update={"modeling_folds": 3})
    write_processed_features(settings, n_train=90, n_test=20)
    dataset = prepare_modeling_dataset(settings)
    artifact_paths = resolve_modeling_artifact_paths(settings)

    tuned_params = {
        "n_estimators": 60,
        "learning_rate": 0.08,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
    }

    candidates = evaluate_tuned_candidates(
        settings=settings,
        dataset=dataset,
        model_family=MODEL_LIGHTGBM,
        tuned_params=tuned_params,
        calibration_methods=[CALIBRATION_NONE, CALIBRATION_SIGMOID],
        artifact_paths=artifact_paths,
    )

    assert len(candidates) == 2
    for candidate in candidates:
        assert candidate.model_artifact_path.exists()
        assert candidate.fold_metrics["fold"].isin(["1", "2", "3", "overall"]).all()
        assert np.min(candidate.oof_predictions) >= 0.0
        assert np.max(candidate.oof_predictions) <= 1.0

    comparison = build_calibration_comparison(candidates)
    assert set(comparison["calibration_method"]) == {CALIBRATION_NONE, CALIBRATION_SIGMOID}
    assert "brier_score" in comparison.columns
    assert "log_loss" in comparison.columns
