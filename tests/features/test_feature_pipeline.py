"""Integration-style tests for feature pipeline outputs and metadata."""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd
import pytest

from credit_risk_altdata.config import Settings
from credit_risk_altdata.data.loaders import build_interim_parquet
from credit_risk_altdata.features.constants import ENTITY_ID_COLUMN, TARGET_COLUMN
from credit_risk_altdata.features.pipeline import build_feature_matrices


def test_build_feature_matrices_from_raw(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
) -> None:
    write_raw_dataset(synthetic_settings)

    result = build_feature_matrices(synthetic_settings, input_source="raw", overwrite=True)

    assert result.train_output_path.exists()
    assert result.test_output_path.exists()
    assert result.manifest_path.exists()
    assert result.join_summary_path.exists()
    assert result.summary_path.exists()

    train = pd.read_parquet(result.train_output_path)
    test = pd.read_parquet(result.test_output_path)
    manifest = pd.read_csv(result.manifest_path)

    assert train[ENTITY_ID_COLUMN].is_unique
    assert test[ENTITY_ID_COLUMN].is_unique
    assert TARGET_COLUMN in train.columns
    assert TARGET_COLUMN not in test.columns

    train_feature_columns = [column for column in train.columns if column != TARGET_COLUMN]
    assert train_feature_columns == list(test.columns)

    for prefix in ("bureau_", "prev_", "pos_", "cc_", "inst_"):
        assert any(column.startswith(prefix) for column in test.columns)

    manifest_columns = {
        "feature_name",
        "source_module",
        "dtype",
        "null_fraction",
        "is_target",
        "is_identifier",
    }
    assert manifest_columns.issubset(set(manifest.columns))
    non_target_features = manifest.loc[~manifest["is_target"], "feature_name"]
    assert not non_target_features.str.contains("target", case=False).any()

    join_payload = json.loads(result.join_summary_path.read_text(encoding="utf-8"))
    assert join_payload["input_source"] == "raw"
    assert len(join_payload["modules"]) == 5


def test_build_feature_matrices_from_interim(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
) -> None:
    write_raw_dataset(synthetic_settings)
    build_interim_parquet(synthetic_settings, force=True)

    result = build_feature_matrices(synthetic_settings, input_source="interim", overwrite=True)

    assert result.train_output_path.exists()
    assert result.test_output_path.exists()


def test_build_feature_matrices_respects_overwrite_flag(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
) -> None:
    write_raw_dataset(synthetic_settings)

    build_feature_matrices(synthetic_settings, input_source="raw", overwrite=True)
    with pytest.raises(FileExistsError):
        build_feature_matrices(synthetic_settings, input_source="raw", overwrite=False)
