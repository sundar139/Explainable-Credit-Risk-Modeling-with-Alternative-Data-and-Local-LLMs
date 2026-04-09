"""Data preparation utilities for baseline model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]

from credit_risk_altdata.config import Settings
from credit_risk_altdata.modeling.constants import IDENTIFIER_COLUMN, LABEL_COLUMN


@dataclass(frozen=True, slots=True)
class ModelingDataset:
    """Validated and encoded dataset for baseline model training."""

    train_ids: Series
    test_ids: Series
    y: Series
    x_train: DataFrame
    x_test: DataFrame
    feature_columns: list[str]
    categorical_columns: list[str]
    train_input_path: Path
    test_input_path: Path


def resolve_feature_input_paths(
    settings: Settings,
    input_path_override: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve train and test feature matrix paths."""
    if input_path_override is None:
        base_dir = settings.home_credit_processed_dir
        return base_dir / "train_features.parquet", base_dir / "test_features.parquet"

    if input_path_override.is_dir():
        return (
            input_path_override / "train_features.parquet",
            input_path_override / "test_features.parquet",
        )

    return input_path_override, input_path_override.parent / "test_features.parquet"


def load_feature_frames(train_path: Path, test_path: Path) -> tuple[DataFrame, DataFrame]:
    """Load train and test feature matrices from parquet files."""
    if not train_path.exists():
        raise FileNotFoundError(f"Training feature file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test feature file not found: {test_path}")

    train_frame = pd.read_parquet(train_path)
    test_frame = pd.read_parquet(test_path)
    return train_frame, test_frame


def validate_feature_frames(
    train_frame: DataFrame,
    test_frame: DataFrame,
) -> tuple[DataFrame, DataFrame, list[str]]:
    """Validate train/test matrices and return aligned feature column list."""
    if IDENTIFIER_COLUMN not in train_frame.columns:
        raise ValueError(f"Training frame is missing identifier column: {IDENTIFIER_COLUMN}")
    if LABEL_COLUMN not in train_frame.columns:
        raise ValueError(f"Training frame is missing label column: {LABEL_COLUMN}")
    if IDENTIFIER_COLUMN not in test_frame.columns:
        raise ValueError(f"Test frame is missing identifier column: {IDENTIFIER_COLUMN}")
    if LABEL_COLUMN in test_frame.columns:
        raise ValueError("Test frame must not include TARGET")

    train_duplicate_ids = int(train_frame.duplicated(subset=[IDENTIFIER_COLUMN]).sum())
    if train_duplicate_ids > 0:
        raise ValueError(f"Training frame has duplicate SK_ID_CURR rows: {train_duplicate_ids}")

    test_duplicate_ids = int(test_frame.duplicated(subset=[IDENTIFIER_COLUMN]).sum())
    if test_duplicate_ids > 0:
        raise ValueError(f"Test frame has duplicate SK_ID_CURR rows: {test_duplicate_ids}")

    if train_frame[LABEL_COLUMN].isna().any():
        raise ValueError("Training labels contain missing TARGET values")
    if train_frame[LABEL_COLUMN].nunique(dropna=True) < 2:
        raise ValueError("Training labels must contain at least two classes")

    train_feature_columns = [
        column
        for column in train_frame.columns
        if column not in (IDENTIFIER_COLUMN, LABEL_COLUMN)
    ]
    test_feature_columns = [column for column in test_frame.columns if column != IDENTIFIER_COLUMN]

    if not train_feature_columns:
        raise ValueError("Training feature set is empty")

    if set(train_feature_columns) != set(test_feature_columns):
        missing_in_test = sorted(set(train_feature_columns).difference(test_feature_columns))
        extra_in_test = sorted(set(test_feature_columns).difference(train_feature_columns))
        raise ValueError(
            "Train/test feature columns mismatch. "
            f"Missing in test: {missing_in_test}; extra in test: {extra_in_test}"
        )

    aligned_train = train_frame[[IDENTIFIER_COLUMN] + train_feature_columns + [LABEL_COLUMN]].copy()
    aligned_test = test_frame[[IDENTIFIER_COLUMN] + train_feature_columns].copy()
    return aligned_train, aligned_test, train_feature_columns


def encode_feature_columns(
    train_frame: DataFrame,
    test_frame: DataFrame,
    feature_columns: list[str],
) -> tuple[DataFrame, DataFrame, list[str]]:
    """Encode non-numeric columns consistently into numeric values."""
    encoded_train_columns: dict[str, Series] = {}
    encoded_test_columns: dict[str, Series] = {}
    categorical_columns: list[str] = []

    for column in feature_columns:
        train_values = train_frame[column]
        test_values = test_frame[column]

        if is_bool_dtype(train_values):
            encoded_train_columns[column] = train_values.astype("int8")
            encoded_test_columns[column] = test_values.astype("int8")
            continue

        if is_numeric_dtype(train_values):
            encoded_train_columns[column] = pd.to_numeric(train_values, errors="coerce").astype(
                "float32"
            )
            encoded_test_columns[column] = pd.to_numeric(test_values, errors="coerce").astype(
                "float32"
            )
            continue

        combined = pd.concat([train_values, test_values], axis=0, ignore_index=True)
        categorical = pd.Categorical(combined.astype("string").fillna("__MISSING__"))
        encoded_codes = pd.Series(categorical.codes, dtype="int32")
        encoded_train_columns[column] = pd.Series(
            encoded_codes.iloc[: len(train_values)].to_numpy(),
            index=train_frame.index,
            dtype="int32",
        )
        encoded_test_columns[column] = pd.Series(
            encoded_codes.iloc[len(train_values) :].to_numpy(),
            index=test_frame.index,
            dtype="int32",
        )
        categorical_columns.append(column)

    encoded_train = pd.DataFrame(encoded_train_columns, index=train_frame.index)
    encoded_test = pd.DataFrame(encoded_test_columns, index=test_frame.index)

    non_numeric_columns = encoded_train.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_columns:
        raise ValueError(
            f"Encoded training features still contain non-numeric columns: {non_numeric_columns}"
        )

    return encoded_train[feature_columns], encoded_test[feature_columns], categorical_columns


def build_stratified_folds(
    y: Series,
    *,
    n_splits: int,
    random_seed: int,
) -> list[tuple[list[int], list[int]]]:
    """Create stratified train/validation folds."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    folds: list[tuple[list[int], list[int]]] = []
    for train_indices, valid_indices in splitter.split(X=y.to_numpy(), y=y.to_numpy()):
        folds.append((train_indices.tolist(), valid_indices.tolist()))
    return folds


def prepare_modeling_dataset(
    settings: Settings,
    *,
    input_path_override: Path | None = None,
) -> ModelingDataset:
    """Load, validate, align, and encode processed features for model training."""
    train_path, test_path = resolve_feature_input_paths(settings, input_path_override)
    train_frame, test_frame = load_feature_frames(train_path, test_path)
    aligned_train, aligned_test, feature_columns = validate_feature_frames(train_frame, test_frame)

    x_train, x_test, categorical_columns = encode_feature_columns(
        aligned_train,
        aligned_test,
        feature_columns,
    )

    y = aligned_train[LABEL_COLUMN].astype("int8")
    return ModelingDataset(
        train_ids=aligned_train[IDENTIFIER_COLUMN].copy(),
        test_ids=aligned_test[IDENTIFIER_COLUMN].copy(),
        y=y,
        x_train=x_train,
        x_test=x_test,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        train_input_path=train_path,
        test_input_path=test_path,
    )
