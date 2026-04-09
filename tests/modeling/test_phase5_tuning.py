"""Tests for Phase 5 tuning workflow."""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd

from credit_risk_altdata.config import Settings
from credit_risk_altdata.modeling.constants import (
    CALIBRATION_NONE,
    FINAL_PRODUCTION_CANDIDATE_FILE,
    MODEL_CATBOOST,
    MODEL_LIGHTGBM,
)
from credit_risk_altdata.modeling.data_prep import prepare_modeling_dataset
from credit_risk_altdata.modeling.tuning import (
    run_tuned_modeling,
    tune_model_hyperparameters,
)


def _phase5_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "modeling_folds": 3,
            "modeling_threshold_grid_min": 0.2,
            "modeling_threshold_grid_max": 0.8,
            "modeling_threshold_grid_step": 0.2,
        }
    )


def test_tune_model_hyperparameters_lightgbm(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
) -> None:
    settings = _phase5_settings(synthetic_settings)
    write_processed_features(settings, n_train=80, n_test=20)
    dataset = prepare_modeling_dataset(settings)

    result = tune_model_hyperparameters(
        settings=settings,
        dataset=dataset,
        model_family=MODEL_LIGHTGBM,
        n_trials=2,
    )

    assert result.model_family == MODEL_LIGHTGBM
    assert 0.0 <= result.best_score <= 1.0
    assert result.best_params
    assert not result.trial_results.empty


def test_tune_model_hyperparameters_catboost(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
) -> None:
    settings = _phase5_settings(synthetic_settings)
    write_processed_features(settings, n_train=80, n_test=20)
    dataset = prepare_modeling_dataset(settings)

    result = tune_model_hyperparameters(
        settings=settings,
        dataset=dataset,
        model_family=MODEL_CATBOOST,
        n_trials=1,
    )

    assert result.model_family == MODEL_CATBOOST
    assert 0.0 <= result.best_score <= 1.0
    assert result.best_params
    assert not result.trial_results.empty


def test_run_tuned_modeling_writes_phase5_artifacts(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
) -> None:
    settings = _phase5_settings(synthetic_settings)
    write_processed_features(settings, n_train=90, n_test=25)

    result = run_tuned_modeling(
        settings,
        model_selection=MODEL_LIGHTGBM,
        n_trials=1,
        calibration_selection=CALIBRATION_NONE,
        overwrite=True,
    )

    assert result.tuning_results_path.exists()
    assert result.calibration_comparison_path.exists()
    assert result.tuned_model_comparison_path.exists()
    assert result.threshold_analysis_path.exists()
    assert result.evaluation_summary_path.exists()
    assert result.tuned_modeling_summary_path.exists()
    assert result.final_candidate_summary_path.exists()
    assert result.final_candidate_summary_path == settings.modeling_final_candidate_summary_path
    assert result.final_candidate_summary_path.name == FINAL_PRODUCTION_CANDIDATE_FILE

    tuning_results = pd.read_csv(result.tuning_results_path)
    assert set(tuning_results["model_family"]) == {MODEL_LIGHTGBM}

    final_candidate_payload = json.loads(
        result.final_candidate_summary_path.read_text(encoding="utf-8")
    )
    required_keys = {
        "final_model_family",
        "tuned",
        "calibrated",
        "primary_metric",
        "threshold",
        "training_timestamp",
        "selected_artifact_path",
        "justification",
        "source_comparison_artifact",
    }
    assert required_keys.issubset(set(final_candidate_payload.keys()))
    assert final_candidate_payload["source_comparison_artifact"] == str(
        result.tuned_model_comparison_path
    )

    best_lightgbm = json.loads(
        (settings.modeling_tuning_dir / "best_params_lightgbm.json").read_text(encoding="utf-8")
    )
    assert best_lightgbm["model_family"] == MODEL_LIGHTGBM
    assert best_lightgbm["tuned"] is True
