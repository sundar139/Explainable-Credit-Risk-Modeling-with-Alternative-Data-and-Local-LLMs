"""Tests for Home Credit data loading and parquet conversion."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from credit_risk_altdata.config import Settings
from credit_risk_altdata.data.loaders import (
    build_interim_parquet,
    load_application_train,
    missing_core_raw_files,
)


def test_load_application_train_returns_dataframe(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
) -> None:
    write_raw_dataset(synthetic_settings)

    dataframe = load_application_train(synthetic_settings)

    assert dataframe.shape[0] == 2
    assert "SK_ID_CURR" in dataframe.columns


def test_build_interim_parquet_writes_expected_files(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
) -> None:
    write_raw_dataset(synthetic_settings)

    first_run = build_interim_parquet(synthetic_settings)

    assert len(first_run) == 8
    assert all(record.written for record in first_run)
    assert all(record.parquet_path.exists() for record in first_run)

    second_run = build_interim_parquet(synthetic_settings)
    assert all(not record.written for record in second_run)


def test_build_interim_parquet_fails_on_missing_required_file(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
) -> None:
    write_raw_dataset(synthetic_settings, missing_files={"application_train.csv"})

    missing_files = missing_core_raw_files(synthetic_settings)
    assert "application_train.csv" in missing_files

    with pytest.raises(FileNotFoundError):
        build_interim_parquet(synthetic_settings)
