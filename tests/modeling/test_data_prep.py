"""Tests for baseline modeling data preparation."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from credit_risk_altdata.config import Settings
from credit_risk_altdata.modeling.data_prep import (
    build_stratified_folds,
    prepare_modeling_dataset,
)


def test_prepare_modeling_dataset_encodes_non_numeric_columns(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
) -> None:
    write_processed_features(synthetic_settings, include_categorical=True)

    dataset = prepare_modeling_dataset(synthetic_settings)

    assert dataset.x_train.shape[0] == 120
    assert dataset.x_test.shape[0] == 40
    assert dataset.y.shape[0] == 120
    assert "app_contract_type" in dataset.categorical_columns
    assert dataset.x_train.select_dtypes(exclude=["number"]).empty


def test_prepare_modeling_dataset_fails_when_target_missing(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
) -> None:
    write_processed_features(synthetic_settings, drop_target=True)

    with pytest.raises(ValueError):
        prepare_modeling_dataset(synthetic_settings)


def test_prepare_modeling_dataset_fails_on_duplicate_identifier(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
) -> None:
    write_processed_features(synthetic_settings, duplicate_train_ids=True)

    with pytest.raises(ValueError):
        prepare_modeling_dataset(synthetic_settings)


def test_prepare_modeling_dataset_fails_on_mismatched_columns(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
) -> None:
    write_processed_features(synthetic_settings, mismatched_test_columns=True)

    with pytest.raises(ValueError):
        prepare_modeling_dataset(synthetic_settings)


def test_build_stratified_folds_maintains_class_coverage() -> None:
    labels = pd.Series([0, 1] * 20, dtype="int8")

    folds = build_stratified_folds(labels, n_splits=5, random_seed=42)

    assert len(folds) == 5
    for train_indices, valid_indices in folds:
        train_labels = labels.take(train_indices)
        valid_labels = labels.take(valid_indices)
        assert train_labels.nunique() == 2
        assert valid_labels.nunique() == 2
