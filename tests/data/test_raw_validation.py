"""Tests for raw data validation logic."""

from __future__ import annotations

from collections.abc import Callable

from credit_risk_altdata.config import Settings
from credit_risk_altdata.data.raw_validation import validate_raw_data, validate_required_raw_files


def test_validate_required_raw_files_detects_missing(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
) -> None:
    write_raw_dataset(synthetic_settings, missing_files={"bureau.csv"})

    missing_files = validate_required_raw_files(synthetic_settings)

    assert "bureau.csv" in missing_files


def test_validate_raw_data_detects_required_column_issue(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
) -> None:
    write_raw_dataset(synthetic_settings, drop_bureau_key_column=True)

    result = validate_raw_data(synthetic_settings)

    assert result.is_valid is False
    assert any(
        "bureau.csv is missing required key columns" in message for message in result.errors
    )


def test_validate_raw_data_detects_duplicate_unique_key(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
) -> None:
    write_raw_dataset(synthetic_settings, duplicate_application_curr=True)

    result = validate_raw_data(synthetic_settings)

    assert result.is_valid is False
    assert any(
        "application_train.csv has duplicate keys for [SK_ID_CURR]" in message
        for message in result.errors
    )


def test_validate_raw_data_writes_reports(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
) -> None:
    write_raw_dataset(synthetic_settings)

    result = validate_raw_data(synthetic_settings)

    assert result.is_valid is True
    for report_path in result.report_paths.values():
        assert report_path.exists()
