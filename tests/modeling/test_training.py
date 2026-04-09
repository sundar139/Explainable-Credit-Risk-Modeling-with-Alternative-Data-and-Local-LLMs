"""Integration tests for baseline training orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd
import pytest

from credit_risk_altdata.config import Settings
from credit_risk_altdata.modeling.constants import MODEL_ALL, MODEL_CATBOOST, MODEL_LIGHTGBM
from credit_risk_altdata.modeling.training import run_baseline_training


def test_run_baseline_training_lightgbm_only(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
) -> None:
    write_processed_features(synthetic_settings, n_train=90, n_test=30)

    result = run_baseline_training(
        synthetic_settings,
        model_selection=MODEL_LIGHTGBM,
        overwrite=True,
    )

    assert result.fold_metrics_path.exists()
    assert result.model_comparison_path.exists()
    assert result.oof_predictions_path.exists()
    assert result.test_predictions_path.exists()
    assert result.best_model_summary_path.exists()

    comparison = pd.read_csv(result.model_comparison_path)
    assert list(comparison["model_name"]) == [MODEL_LIGHTGBM]


def test_run_baseline_training_catboost_only(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
) -> None:
    write_processed_features(synthetic_settings, n_train=90, n_test=30)

    result = run_baseline_training(
        synthetic_settings,
        model_selection=MODEL_CATBOOST,
        overwrite=True,
    )

    comparison = pd.read_csv(result.model_comparison_path)
    assert list(comparison["model_name"]) == [MODEL_CATBOOST]
    assert (synthetic_settings.modeling_models_dir / MODEL_CATBOOST / "final_model.cbm").exists()


def test_run_baseline_training_all_models_writes_selection_summary(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
) -> None:
    write_processed_features(synthetic_settings, n_train=100, n_test=30)

    result = run_baseline_training(
        synthetic_settings,
        model_selection=MODEL_ALL,
        overwrite=True,
    )

    comparison = pd.read_csv(result.model_comparison_path)
    assert set(comparison["model_name"]) == {MODEL_LIGHTGBM, MODEL_CATBOOST}

    best_summary = json.loads(result.best_model_summary_path.read_text(encoding="utf-8"))
    assert best_summary["best_model_name"] in {MODEL_LIGHTGBM, MODEL_CATBOOST}
    assert best_summary["primary_metric"] == synthetic_settings.modeling_primary_metric
    assert "trained_model_paths" in best_summary

    oof_predictions = pd.read_parquet(result.oof_predictions_path)
    assert "oof_pred_lightgbm" in oof_predictions.columns
    assert "oof_pred_catboost" in oof_predictions.columns

    assert (
        synthetic_settings.modeling_feature_importance_dir / "lightgbm_feature_importance.csv"
    ).exists()
    assert (
        synthetic_settings.modeling_feature_importance_dir / "catboost_feature_importance.csv"
    ).exists()


def test_run_baseline_training_fails_with_invalid_input(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
) -> None:
    write_processed_features(synthetic_settings, drop_target=True)

    with pytest.raises(ValueError):
        run_baseline_training(
            synthetic_settings,
            model_selection=MODEL_LIGHTGBM,
            overwrite=True,
        )
