"""Constants and helper utilities for feature engineering."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from pandas import DataFrame, Series

ENTITY_ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
SPECIAL_DAY_PLACEHOLDER = 365243

FEATURE_INPUT_TABLES: dict[str, str] = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "installments_payments": "installments_payments.csv",
}

FEATURE_PREFIX_BY_MODULE: dict[str, str] = {
    "application_base": "app_",
    "bureau": "bureau_",
    "previous_application": "prev_",
    "pos_cash": "pos_",
    "credit_card": "cc_",
    "installments": "inst_",
}

NUMERIC_AGGREGATIONS: tuple[str, ...] = ("mean", "median", "min", "max", "std")


def sanitize_token(value: str) -> str:
    """Convert arbitrary text into a safe feature-name token."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())
    return cleaned.strip("_") or "unknown"


def require_columns(frame: DataFrame, columns: Iterable[str], context: str) -> None:
    """Ensure required columns are present in a DataFrame."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def assert_unique_entity_rows(frame: DataFrame, context: str) -> None:
    """Ensure DataFrame has one row per SK_ID_CURR."""
    require_columns(frame, [ENTITY_ID_COLUMN], context)
    duplicate_count = int(frame.duplicated(subset=[ENTITY_ID_COLUMN]).sum())
    if duplicate_count > 0:
        raise ValueError(f"{context} produced duplicate {ENTITY_ID_COLUMN} rows: {duplicate_count}")


def safe_divide(numerator: Series, denominator: Series) -> Series:
    """Divide two series while protecting against divide-by-zero."""
    denominator_non_zero = denominator.where(denominator != 0)
    return numerator.astype("float64") / denominator_non_zero.astype("float64")


def normalize_object_columns(frame: DataFrame) -> DataFrame:
    """Convert object columns to pandas string dtype for stable parquet output."""
    normalized = frame.copy()
    object_columns = normalized.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        normalized[column] = normalized[column].astype("string")
    return normalized


def aggregated_numeric_features(
    frame: DataFrame,
    *,
    group_key: str,
    numeric_columns: Sequence[str],
    prefix: str,
    aggregations: Sequence[str] = NUMERIC_AGGREGATIONS,
) -> DataFrame:
    """Aggregate available numeric columns and return flattened feature names."""
    available_columns = [column for column in numeric_columns if column in frame.columns]
    if not available_columns:
        return frame[[group_key]].drop_duplicates().reset_index(drop=True)

    grouped = frame.groupby(group_key, dropna=False)[available_columns].agg(list(aggregations))
    flattened_columns: list[str] = []
    for column in grouped.columns:
        if isinstance(column, tuple):
            source_column, aggregation = column
        else:
            source_column, aggregation = str(column), "value"
        flattened_columns.append(
            f"{prefix}{sanitize_token(str(source_column))}_{sanitize_token(str(aggregation))}"
        )
    grouped.columns = flattened_columns
    return grouped.reset_index()
