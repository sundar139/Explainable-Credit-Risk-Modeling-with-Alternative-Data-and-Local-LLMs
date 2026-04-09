"""Feature manifest generation for processed train/test matrices."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame

from credit_risk_altdata.features.constants import ENTITY_ID_COLUMN, TARGET_COLUMN


def build_feature_manifest(
    train_features: DataFrame,
    test_features: DataFrame,
    *,
    module_feature_map: dict[str, list[str]],
) -> DataFrame:
    """Build feature-level metadata for train/test matrices."""
    train_columns = list(train_features.columns)
    test_columns = list(test_features.columns)

    if TARGET_COLUMN not in train_columns:
        raise ValueError("train_features must contain TARGET")
    if TARGET_COLUMN in test_columns:
        raise ValueError("test_features must not contain TARGET")

    module_by_feature: dict[str, str] = {}
    for module_name, features in module_feature_map.items():
        for feature_name in features:
            module_by_feature[feature_name] = module_name

    all_columns = sorted(set(train_columns).union(test_columns))
    records: list[dict[str, object]] = []
    for column in all_columns:
        is_target = column == TARGET_COLUMN
        is_identifier = column == ENTITY_ID_COLUMN

        if column in train_features.columns and column in test_features.columns:
            combined_series = pd.concat(
                [train_features[column], test_features[column]],
                axis=0,
                ignore_index=True,
            )
            null_fraction = float(combined_series.isna().mean())
            dtype = str(train_features[column].dtype)
        elif column in train_features.columns:
            null_fraction = float(train_features[column].isna().mean())
            dtype = str(train_features[column].dtype)
        else:
            null_fraction = float(test_features[column].isna().mean())
            dtype = str(test_features[column].dtype)

        if is_target:
            source_module = "target"
        elif is_identifier:
            source_module = "identifier"
        else:
            source_module = module_by_feature.get(column, "unknown")

        records.append(
            {
                "feature_name": column,
                "source_module": source_module,
                "dtype": dtype,
                "null_fraction": round(null_fraction, 8),
                "is_target": is_target,
                "is_identifier": is_identifier,
            }
        )

    manifest = pd.DataFrame(records)
    return manifest.sort_values("feature_name").reset_index(drop=True)


def write_feature_manifest_csv(manifest: DataFrame, output_dir: Path) -> Path:
    """Write feature manifest as CSV."""
    output_path = output_dir / "feature_manifest.csv"
    manifest.to_csv(output_path, index=False)
    return output_path


def build_feature_summary_lines(
    *,
    train_features: DataFrame,
    test_features: DataFrame,
    module_feature_map: dict[str, list[str]],
) -> list[str]:
    """Create summary lines for markdown/text output."""
    lines: list[str] = [
        "## Matrix Shapes",
        f"- Train: {train_features.shape[0]} rows x {train_features.shape[1]} columns",
        f"- Test: {test_features.shape[0]} rows x {test_features.shape[1]} columns",
        "",
        "## Per-Module Feature Counts",
    ]

    for module_name in sorted(module_feature_map):
        lines.append(f"- {module_name}: {len(module_feature_map[module_name])}")

    return lines
