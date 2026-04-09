"""Base application feature engineering for train/test assembly."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame

from credit_risk_altdata.features.constants import (
    ENTITY_ID_COLUMN,
    FEATURE_PREFIX_BY_MODULE,
    SPECIAL_DAY_PLACEHOLDER,
    TARGET_COLUMN,
    assert_unique_entity_rows,
    require_columns,
    safe_divide,
)

APPLICATION_PREFIX = FEATURE_PREFIX_BY_MODULE["application_base"]


def _add_ratio_feature(
    frame: DataFrame,
    *,
    output_column: str,
    numerator_column: str,
    denominator_column: str,
) -> None:
    if numerator_column in frame.columns and denominator_column in frame.columns:
        frame[output_column] = safe_divide(frame[numerator_column], frame[denominator_column])


def _add_binary_flag(
    frame: DataFrame,
    *,
    source_column: str,
    output_column: str,
    positive_value: str,
) -> None:
    if source_column not in frame.columns:
        return
    values = frame[source_column].astype("string").fillna("UNKNOWN")
    frame[output_column] = (values == positive_value).astype("int8")


def _add_frequency_feature(frame: DataFrame, *, source_column: str, output_column: str) -> None:
    if source_column not in frame.columns:
        return
    values = frame[source_column].astype("string").fillna("UNKNOWN")
    frequencies = values.value_counts(dropna=False, normalize=True)
    frame[output_column] = values.map(frequencies).astype("float32")


def _engineer_application_features(frame: DataFrame) -> DataFrame:
    engineered = frame.copy()

    for day_column in ("DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE"):
        if day_column in engineered.columns:
            placeholder_mask = engineered[day_column] == SPECIAL_DAY_PLACEHOLDER
            engineered[f"{APPLICATION_PREFIX}{day_column.lower()}_placeholder_flag"] = (
                placeholder_mask.astype("int8")
            )
            engineered.loc[placeholder_mask, day_column] = pd.NA

    for ext_column in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        if ext_column in engineered.columns:
            engineered[f"{APPLICATION_PREFIX}{ext_column.lower()}_missing_flag"] = (
                engineered[ext_column].isna().astype("int8")
            )

    ext_columns = [
        column
        for column in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
        if column in engineered.columns
    ]
    if ext_columns:
        engineered[f"{APPLICATION_PREFIX}ext_source_mean"] = engineered[ext_columns].mean(axis=1)
        engineered[f"{APPLICATION_PREFIX}ext_source_std"] = engineered[ext_columns].std(axis=1)

    _add_ratio_feature(
        engineered,
        output_column=f"{APPLICATION_PREFIX}credit_income_ratio",
        numerator_column="AMT_CREDIT",
        denominator_column="AMT_INCOME_TOTAL",
    )
    _add_ratio_feature(
        engineered,
        output_column=f"{APPLICATION_PREFIX}annuity_income_ratio",
        numerator_column="AMT_ANNUITY",
        denominator_column="AMT_INCOME_TOTAL",
    )
    _add_ratio_feature(
        engineered,
        output_column=f"{APPLICATION_PREFIX}credit_annuity_ratio",
        numerator_column="AMT_CREDIT",
        denominator_column="AMT_ANNUITY",
    )
    _add_ratio_feature(
        engineered,
        output_column=f"{APPLICATION_PREFIX}goods_credit_ratio",
        numerator_column="AMT_GOODS_PRICE",
        denominator_column="AMT_CREDIT",
    )
    _add_ratio_feature(
        engineered,
        output_column=f"{APPLICATION_PREFIX}children_family_ratio",
        numerator_column="CNT_CHILDREN",
        denominator_column="CNT_FAM_MEMBERS",
    )
    _add_ratio_feature(
        engineered,
        output_column=f"{APPLICATION_PREFIX}employed_age_ratio",
        numerator_column="DAYS_EMPLOYED",
        denominator_column="DAYS_BIRTH",
    )
    _add_ratio_feature(
        engineered,
        output_column=f"{APPLICATION_PREFIX}obs30_obs60_ratio",
        numerator_column="OBS_30_CNT_SOCIAL_CIRCLE",
        denominator_column="OBS_60_CNT_SOCIAL_CIRCLE",
    )
    _add_ratio_feature(
        engineered,
        output_column=f"{APPLICATION_PREFIX}def30_def60_ratio",
        numerator_column="DEF_30_CNT_SOCIAL_CIRCLE",
        denominator_column="DEF_60_CNT_SOCIAL_CIRCLE",
    )

    _add_binary_flag(
        engineered,
        source_column="CODE_GENDER",
        output_column=f"{APPLICATION_PREFIX}code_gender_f_flag",
        positive_value="F",
    )
    _add_binary_flag(
        engineered,
        source_column="FLAG_OWN_CAR",
        output_column=f"{APPLICATION_PREFIX}own_car_flag",
        positive_value="Y",
    )
    _add_binary_flag(
        engineered,
        source_column="FLAG_OWN_REALTY",
        output_column=f"{APPLICATION_PREFIX}own_realty_flag",
        positive_value="Y",
    )
    _add_binary_flag(
        engineered,
        source_column="NAME_CONTRACT_TYPE",
        output_column=f"{APPLICATION_PREFIX}contract_cash_loans_flag",
        positive_value="Cash loans",
    )

    _add_frequency_feature(
        engineered,
        source_column="CODE_GENDER",
        output_column=f"{APPLICATION_PREFIX}code_gender_freq",
    )
    _add_frequency_feature(
        engineered,
        source_column="NAME_CONTRACT_TYPE",
        output_column=f"{APPLICATION_PREFIX}contract_type_freq",
    )
    _add_frequency_feature(
        engineered,
        source_column="FLAG_OWN_CAR",
        output_column=f"{APPLICATION_PREFIX}own_car_freq",
    )

    return engineered


def build_application_base_features(
    application_train: DataFrame,
    application_test: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    """Build leakage-safe base train and test matrices from application tables."""
    require_columns(
        application_train,
        [ENTITY_ID_COLUMN, TARGET_COLUMN],
        "application_train base table",
    )
    require_columns(application_test, [ENTITY_ID_COLUMN], "application_test base table")
    if TARGET_COLUMN in application_test.columns:
        raise ValueError("application_test must not contain TARGET")

    train_copy = application_train.copy()
    test_copy = application_test.copy()
    assert_unique_entity_rows(train_copy, "application_train base table")
    assert_unique_entity_rows(test_copy, "application_test base table")

    train_target = train_copy[TARGET_COLUMN].reset_index(drop=True)
    train_no_target = train_copy.drop(columns=[TARGET_COLUMN]).copy()

    train_no_target["_split"] = "train"
    train_no_target["_row_order"] = range(len(train_no_target))
    test_copy["_split"] = "test"
    test_copy["_row_order"] = range(len(test_copy))

    combined = pd.concat([train_no_target, test_copy], axis=0, ignore_index=True, sort=False)
    engineered = _engineer_application_features(combined)

    train_features = (
        engineered[engineered["_split"] == "train"]
        .sort_values("_row_order")
        .drop(columns=["_split", "_row_order"])
        .reset_index(drop=True)
    )
    test_features = (
        engineered[engineered["_split"] == "test"]
        .sort_values("_row_order")
        .drop(columns=["_split", "_row_order"])
        .reset_index(drop=True)
    )

    train_features[TARGET_COLUMN] = train_target.values

    train_feature_columns = [column for column in train_features.columns if column != TARGET_COLUMN]
    test_features = test_features.reindex(columns=train_feature_columns)
    train_features = train_features.reindex(columns=train_feature_columns + [TARGET_COLUMN])

    assert_unique_entity_rows(train_features, "application_base train matrix")
    assert_unique_entity_rows(test_features, "application_base test matrix")
    return train_features, test_features
