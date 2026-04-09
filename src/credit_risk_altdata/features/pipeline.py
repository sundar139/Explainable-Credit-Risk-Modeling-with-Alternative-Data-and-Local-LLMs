"""End-to-end feature engineering pipeline for Home Credit tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from pandas import DataFrame

from credit_risk_altdata.config import Settings
from credit_risk_altdata.data.loaders import read_home_credit_table
from credit_risk_altdata.data.reporting import write_json_report, write_markdown_report
from credit_risk_altdata.features.base_application import build_application_base_features
from credit_risk_altdata.features.bureau import build_bureau_features
from credit_risk_altdata.features.constants import (
    ENTITY_ID_COLUMN,
    FEATURE_INPUT_TABLES,
    FEATURE_PREFIX_BY_MODULE,
    TARGET_COLUMN,
    assert_unique_entity_rows,
    normalize_object_columns,
    require_columns,
)
from credit_risk_altdata.features.credit_card import build_credit_card_features
from credit_risk_altdata.features.installments import build_installments_features
from credit_risk_altdata.features.manifest import (
    build_feature_manifest,
    build_feature_summary_lines,
    write_feature_manifest_csv,
)
from credit_risk_altdata.features.pos_cash import build_pos_cash_features
from credit_risk_altdata.features.previous_application import build_previous_application_features
from credit_risk_altdata.utils.filesystem import ensure_directories

InputSource = Literal["raw", "interim"]


class FeaturePipelineError(RuntimeError):
    """Raised when feature engineering fails."""


@dataclass(frozen=True, slots=True)
class FeaturePipelineResult:
    """Result payload for feature pipeline execution."""

    train_output_path: Path
    test_output_path: Path
    manifest_path: Path
    join_summary_path: Path
    summary_path: Path
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]


def _utc_timestamp() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _load_tables_from_raw(settings: Settings) -> dict[str, DataFrame]:
    return {
        table_name: read_home_credit_table(settings=settings, file_name=file_name)
        for table_name, file_name in FEATURE_INPUT_TABLES.items()
    }


def _load_tables_from_interim(settings: Settings) -> dict[str, DataFrame]:
    tables: dict[str, DataFrame] = {}
    for table_name, file_name in FEATURE_INPUT_TABLES.items():
        parquet_name = f"{Path(file_name).stem}.parquet"
        parquet_path = settings.home_credit_interim_dir / parquet_name
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Interim parquet file is missing for {table_name}: {parquet_path}"
            )
        tables[table_name] = pd.read_parquet(parquet_path)
    return tables


def _load_input_tables(settings: Settings, input_source: InputSource) -> dict[str, DataFrame]:
    if input_source == "raw":
        return _load_tables_from_raw(settings)
    if input_source == "interim":
        return _load_tables_from_interim(settings)
    raise FeaturePipelineError(f"Unsupported input source: {input_source}")


def _merge_feature_block(
    base: DataFrame,
    *,
    block: DataFrame,
    module_name: str,
) -> DataFrame:
    require_columns(block, [ENTITY_ID_COLUMN], f"{module_name} feature block")
    assert_unique_entity_rows(block, f"{module_name} feature block")

    feature_columns = [column for column in block.columns if column != ENTITY_ID_COLUMN]
    if not feature_columns:
        return base

    duplicate_columns = sorted(set(feature_columns).intersection(base.columns))
    if duplicate_columns:
        raise FeaturePipelineError(
            f"{module_name} feature block has duplicate columns: {duplicate_columns}"
        )

    expected_prefix = FEATURE_PREFIX_BY_MODULE[module_name]
    invalid_columns = [
        column for column in feature_columns if not column.startswith(expected_prefix)
    ]
    if invalid_columns:
        raise FeaturePipelineError(
            f"{module_name} features must start with '{expected_prefix}': {invalid_columns}"
        )

    return base.merge(
        block,
        on=ENTITY_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )


def _validate_train_test_alignment(train: DataFrame, test: DataFrame) -> None:
    if TARGET_COLUMN not in train.columns:
        raise FeaturePipelineError("Train matrix does not contain TARGET")
    if TARGET_COLUMN in test.columns:
        raise FeaturePipelineError("Test matrix must not contain TARGET")

    train_feature_columns = [column for column in train.columns if column != TARGET_COLUMN]
    test_feature_columns = list(test.columns)
    if train_feature_columns != test_feature_columns:
        missing_in_test = sorted(set(train_feature_columns).difference(test_feature_columns))
        extra_in_test = sorted(set(test_feature_columns).difference(train_feature_columns))
        raise FeaturePipelineError(
            "Train/test feature schemas are not aligned. "
            f"Missing in test: {missing_in_test}; extra in test: {extra_in_test}"
        )

    non_target_columns = [column for column in train_feature_columns if column != ENTITY_ID_COLUMN]
    leaked_columns = [
        column for column in non_target_columns if "target" in column.lower()
    ]
    if leaked_columns:
        raise FeaturePipelineError(
            f"Potential leakage columns found in feature matrix: {leaked_columns}"
        )


def build_feature_matrices(
    settings: Settings,
    *,
    input_source: InputSource = "raw",
    overwrite: bool = False,
) -> FeaturePipelineResult:
    """Build model-ready train/test matrices with one row per SK_ID_CURR."""
    tables = _load_input_tables(settings=settings, input_source=input_source)

    train_features, test_features = build_application_base_features(
        application_train=tables["application_train"],
        application_test=tables["application_test"],
    )

    assert_unique_entity_rows(train_features, "base train matrix")
    assert_unique_entity_rows(test_features, "base test matrix")

    module_feature_map: dict[str, list[str]] = {
        "application_base": [
            column
            for column in train_features.columns
            if column not in (ENTITY_ID_COLUMN, TARGET_COLUMN)
        ]
    }

    feature_blocks: list[tuple[str, DataFrame]] = [
        (
            "bureau",
            build_bureau_features(
                bureau=tables["bureau"],
                bureau_balance=tables["bureau_balance"],
            ),
        ),
        (
            "previous_application",
            build_previous_application_features(
                previous_application=tables["previous_application"],
            ),
        ),
        ("pos_cash", build_pos_cash_features(pos_cash_balance=tables["pos_cash_balance"])),
        (
            "credit_card",
            build_credit_card_features(credit_card_balance=tables["credit_card_balance"]),
        ),
        (
            "installments",
            build_installments_features(installments_payments=tables["installments_payments"]),
        ),
    ]

    join_summary_modules: list[dict[str, object]] = []
    for module_name, block in feature_blocks:
        module_columns = [column for column in block.columns if column != ENTITY_ID_COLUMN]
        module_feature_map[module_name] = module_columns

        train_match_rate = float(
            train_features[ENTITY_ID_COLUMN].isin(block[ENTITY_ID_COLUMN]).mean()
        )
        test_match_rate = float(
            test_features[ENTITY_ID_COLUMN].isin(block[ENTITY_ID_COLUMN]).mean()
        )

        train_features = _merge_feature_block(
            base=train_features,
            block=block,
            module_name=module_name,
        )
        test_features = _merge_feature_block(
            base=test_features,
            block=block,
            module_name=module_name,
        )

        join_summary_modules.append(
            {
                "module": module_name,
                "feature_count": len(module_columns),
                "train_entity_match_rate": round(train_match_rate, 8),
                "test_entity_match_rate": round(test_match_rate, 8),
            }
        )

    train_features = normalize_object_columns(train_features)
    test_features = normalize_object_columns(test_features)

    _validate_train_test_alignment(train_features, test_features)
    assert_unique_entity_rows(train_features, "final train feature matrix")
    assert_unique_entity_rows(test_features, "final test feature matrix")

    ensure_directories([settings.home_credit_processed_dir, settings.feature_metadata_dir])

    train_output_path = settings.home_credit_processed_dir / "train_features.parquet"
    test_output_path = settings.home_credit_processed_dir / "test_features.parquet"
    if not overwrite and (train_output_path.exists() or test_output_path.exists()):
        raise FileExistsError(
            "Feature outputs already exist. Use overwrite=True to replace existing files."
        )

    train_features.to_parquet(train_output_path, index=False)
    test_features.to_parquet(test_output_path, index=False)

    manifest = build_feature_manifest(
        train_features=train_features,
        test_features=test_features,
        module_feature_map=module_feature_map,
    )
    manifest_path = write_feature_manifest_csv(manifest, settings.feature_metadata_dir)

    join_summary_payload = {
        "generated_at": _utc_timestamp(),
        "input_source": input_source,
        "train_shape": list(train_features.shape),
        "test_shape": list(test_features.shape),
        "modules": join_summary_modules,
    }
    join_summary_path = write_json_report(
        settings.feature_metadata_dir / "join_summary.json",
        join_summary_payload,
    )

    summary_lines = build_feature_summary_lines(
        train_features=train_features,
        test_features=test_features,
        module_feature_map=module_feature_map,
    )
    summary_lines.extend(
        [
            "",
            "## Outputs",
            f"- Train features: {train_output_path}",
            f"- Test features: {test_output_path}",
            f"- Feature manifest: {manifest_path}",
            f"- Join summary: {join_summary_path}",
        ]
    )
    summary_path = write_markdown_report(
        settings.feature_metadata_dir / "feature_summary.md",
        title="Feature Engineering Summary",
        lines=summary_lines,
    )

    return FeaturePipelineResult(
        train_output_path=train_output_path,
        test_output_path=test_output_path,
        manifest_path=manifest_path,
        join_summary_path=join_summary_path,
        summary_path=summary_path,
        train_shape=(int(train_features.shape[0]), int(train_features.shape[1])),
        test_shape=(int(test_features.shape[0]), int(test_features.shape[1])),
    )
