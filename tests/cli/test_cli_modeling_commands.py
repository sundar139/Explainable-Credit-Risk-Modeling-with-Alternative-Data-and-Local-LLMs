"""CLI tests for baseline modeling commands."""

from __future__ import annotations

from collections.abc import Callable

from pytest import MonkeyPatch

from credit_risk_altdata import cli
from credit_risk_altdata.config import Settings


def test_cli_train_baselines_lightgbm_success(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
    monkeypatch: MonkeyPatch,
) -> None:
    write_processed_features(synthetic_settings, n_train=90, n_test=30)
    monkeypatch.setattr(cli, "get_settings", lambda: synthetic_settings)

    exit_code = cli.main(["train-baselines", "--model", "lightgbm", "--overwrite"])

    assert exit_code == 0
    assert (synthetic_settings.modeling_metrics_dir / "model_comparison.csv").exists()
    assert (synthetic_settings.modeling_reports_dir / "best_model_summary.json").exists()


def test_cli_train_baselines_failure_when_input_missing(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: synthetic_settings)

    exit_code = cli.main(["train-baselines", "--model", "catboost", "--overwrite"])

    assert exit_code == 1
