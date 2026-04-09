"""CLI tests for Phase 5 tune-models command."""

from __future__ import annotations

from collections.abc import Callable

from pytest import MonkeyPatch

from credit_risk_altdata import cli
from credit_risk_altdata.config import Settings


def _phase5_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "modeling_folds": 3,
            "modeling_threshold_grid_min": 0.2,
            "modeling_threshold_grid_max": 0.8,
            "modeling_threshold_grid_step": 0.2,
        }
    )


def test_cli_tune_models_lightgbm_success(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _phase5_settings(synthetic_settings)
    write_processed_features(settings, n_train=90, n_test=20)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    exit_code = cli.main(
        [
            "tune-models",
            "--model",
            "lightgbm",
            "--n-trials",
            "1",
            "--calibration",
            "none",
            "--overwrite",
        ]
    )

    assert exit_code == 0
    assert (settings.modeling_tuning_dir / "tuning_results.csv").exists()
    assert (settings.modeling_calibration_dir / "calibration_comparison.csv").exists()
    assert (settings.modeling_evaluation_dir / "evaluation_summary.json").exists()


def test_cli_tune_models_fails_when_input_missing(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _phase5_settings(synthetic_settings)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    exit_code = cli.main(
        [
            "tune-models",
            "--model",
            "lightgbm",
            "--n-trials",
            "1",
            "--calibration",
            "none",
            "--overwrite",
        ]
    )

    assert exit_code == 1
