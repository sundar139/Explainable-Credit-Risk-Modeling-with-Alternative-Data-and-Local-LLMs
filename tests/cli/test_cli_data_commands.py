"""CLI tests for Phase 2 data commands."""

from __future__ import annotations

from collections.abc import Callable

from pytest import MonkeyPatch

from credit_risk_altdata import cli
from credit_risk_altdata.config import Settings
from credit_risk_altdata.data.download import DataDownloadError, DownloadResult


def test_cli_validate_raw_data_success(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
    monkeypatch: MonkeyPatch,
) -> None:
    write_raw_dataset(synthetic_settings)
    monkeypatch.setattr(cli, "get_settings", lambda: synthetic_settings)

    exit_code = cli.main(["validate-raw-data"])

    assert exit_code == 0
    assert (synthetic_settings.data_validation_dir / "data_quality_summary.json").exists()


def test_cli_validate_raw_data_failure_when_files_missing(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
    monkeypatch: MonkeyPatch,
) -> None:
    write_raw_dataset(synthetic_settings, missing_files={"application_test.csv"})
    monkeypatch.setattr(cli, "get_settings", lambda: synthetic_settings)

    exit_code = cli.main(["validate-raw-data"])

    assert exit_code == 1


def test_cli_build_interim_parquet_success(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
    monkeypatch: MonkeyPatch,
) -> None:
    write_raw_dataset(synthetic_settings)
    monkeypatch.setattr(cli, "get_settings", lambda: synthetic_settings)

    exit_code = cli.main(["build-interim-parquet"])

    assert exit_code == 0
    assert (synthetic_settings.home_credit_interim_dir / "application_train.parquet").exists()


def test_cli_download_data_success_with_mocked_downloader(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    def fake_download_data(*, settings: Settings, force: bool) -> DownloadResult:
        return DownloadResult(
            destination=settings.home_credit_raw_dir,
            downloaded=force,
            extracted_files=tuple(),
            skipped=not force,
        )

    monkeypatch.setattr(cli, "get_settings", lambda: synthetic_settings)
    monkeypatch.setattr(cli, "download_home_credit_dataset", fake_download_data)

    exit_code = cli.main(["download-data"])

    assert exit_code == 0


def test_cli_download_data_failure_with_mocked_error(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    def failing_download(*, settings: Settings, force: bool) -> DownloadResult:
        del settings
        del force
        raise DataDownloadError("simulated download failure")

    monkeypatch.setattr(cli, "get_settings", lambda: synthetic_settings)
    monkeypatch.setattr(cli, "download_home_credit_dataset", failing_download)

    exit_code = cli.main(["download-data", "--force"])

    assert exit_code == 1


def test_cli_build_features_success(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
    monkeypatch: MonkeyPatch,
) -> None:
    write_raw_dataset(synthetic_settings)
    monkeypatch.setattr(cli, "get_settings", lambda: synthetic_settings)

    exit_code = cli.main(["build-features", "--overwrite"])

    assert exit_code == 0
    assert (synthetic_settings.home_credit_processed_dir / "train_features.parquet").exists()
    assert (synthetic_settings.home_credit_processed_dir / "test_features.parquet").exists()
    assert (synthetic_settings.feature_metadata_dir / "feature_manifest.csv").exists()


def test_cli_build_features_fails_when_required_data_missing(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
    monkeypatch: MonkeyPatch,
) -> None:
    write_raw_dataset(synthetic_settings, missing_files={"application_test.csv"})
    monkeypatch.setattr(cli, "get_settings", lambda: synthetic_settings)

    exit_code = cli.main(["build-features", "--overwrite"])

    assert exit_code == 1
