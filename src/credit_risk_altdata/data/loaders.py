"""Data loading utilities for Home Credit raw and interim tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pandas import DataFrame

from credit_risk_altdata.config import Settings
from credit_risk_altdata.data.constants import (
    CORE_RAW_FILES,
    HOME_CREDIT_INTERIM_SUBDIR,
    HOME_CREDIT_RAW_SUBDIR,
)
from credit_risk_altdata.utils.filesystem import ensure_directories


@dataclass(frozen=True, slots=True)
class HomeCreditPaths:
    """Resolved paths for Home Credit datasets."""

    raw_dir: Path
    interim_dir: Path


@dataclass(frozen=True, slots=True)
class ParquetBuildRecord:
    """Result record for a parquet conversion operation."""

    source_csv: str
    parquet_path: Path
    row_count: int
    column_count: int
    written: bool


def resolve_home_credit_paths(settings: Settings) -> HomeCreditPaths:
    """Resolve raw and interim directories for Home Credit tables."""
    return HomeCreditPaths(
        raw_dir=settings.raw_data_dir / HOME_CREDIT_RAW_SUBDIR,
        interim_dir=settings.interim_data_dir / HOME_CREDIT_INTERIM_SUBDIR,
    )


def get_core_raw_file_paths(settings: Settings) -> dict[str, Path]:
    """Return expected raw CSV paths indexed by file name."""
    paths = resolve_home_credit_paths(settings)
    return {file_name: paths.raw_dir / file_name for file_name in CORE_RAW_FILES}


def missing_core_raw_files(settings: Settings) -> list[str]:
    """Return list of required raw files that are missing."""
    file_paths = get_core_raw_file_paths(settings)
    return [file_name for file_name, path in file_paths.items() if not path.exists()]


def read_home_credit_table(
    settings: Settings,
    file_name: str,
    **read_csv_kwargs: Any,
) -> DataFrame:
    """Load a Home Credit CSV table into a pandas DataFrame."""
    if file_name not in CORE_RAW_FILES:
        raise ValueError(f"Unsupported Home Credit table: {file_name}")

    file_path = get_core_raw_file_paths(settings)[file_name]
    if not file_path.exists():
        raise FileNotFoundError(f"Required raw file not found: {file_path}")

    csv_kwargs: dict[str, Any] = {"low_memory": False}
    csv_kwargs.update(read_csv_kwargs)
    return cast(DataFrame, pd.read_csv(file_path, **csv_kwargs))


def load_application_train(settings: Settings) -> DataFrame:
    return read_home_credit_table(settings=settings, file_name="application_train.csv")


def load_application_test(settings: Settings) -> DataFrame:
    return read_home_credit_table(settings=settings, file_name="application_test.csv")


def load_bureau(settings: Settings) -> DataFrame:
    return read_home_credit_table(settings=settings, file_name="bureau.csv")


def load_bureau_balance(settings: Settings) -> DataFrame:
    return read_home_credit_table(settings=settings, file_name="bureau_balance.csv")


def load_previous_application(settings: Settings) -> DataFrame:
    return read_home_credit_table(settings=settings, file_name="previous_application.csv")


def load_pos_cash_balance(settings: Settings) -> DataFrame:
    return read_home_credit_table(settings=settings, file_name="POS_CASH_balance.csv")


def load_credit_card_balance(settings: Settings) -> DataFrame:
    return read_home_credit_table(settings=settings, file_name="credit_card_balance.csv")


def load_installments_payments(settings: Settings) -> DataFrame:
    return read_home_credit_table(settings=settings, file_name="installments_payments.csv")


def build_interim_parquet(
    settings: Settings,
    *,
    force: bool = False,
) -> list[ParquetBuildRecord]:
    """Build parquet copies from raw CSVs for all core Home Credit tables."""
    missing_files = missing_core_raw_files(settings)
    if missing_files:
        missing = ", ".join(sorted(missing_files))
        raise FileNotFoundError(f"Cannot build parquet. Missing required raw files: {missing}")

    paths = resolve_home_credit_paths(settings)
    ensure_directories([paths.interim_dir])

    results: list[ParquetBuildRecord] = []
    for file_name in CORE_RAW_FILES:
        dataframe = read_home_credit_table(settings=settings, file_name=file_name)
        parquet_path = paths.interim_dir / f"{Path(file_name).stem}.parquet"
        should_write = force or not parquet_path.exists()
        if should_write:
            dataframe.to_parquet(parquet_path, index=False)

        results.append(
            ParquetBuildRecord(
                source_csv=file_name,
                parquet_path=parquet_path,
                row_count=int(dataframe.shape[0]),
                column_count=int(dataframe.shape[1]),
                written=should_write,
            )
        )

    return results
