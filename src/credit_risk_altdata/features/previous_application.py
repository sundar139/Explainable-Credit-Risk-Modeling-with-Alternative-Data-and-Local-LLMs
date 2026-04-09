"""Previous application relational feature builder."""

from __future__ import annotations

from pandas import DataFrame

from credit_risk_altdata.features.constants import (
    ENTITY_ID_COLUMN,
    FEATURE_PREFIX_BY_MODULE,
    SPECIAL_DAY_PLACEHOLDER,
    aggregated_numeric_features,
    assert_unique_entity_rows,
    require_columns,
    safe_divide,
    sanitize_token,
)

PREVIOUS_PREFIX = FEATURE_PREFIX_BY_MODULE["previous_application"]


def build_previous_application_features(previous_application: DataFrame) -> DataFrame:
    """Build one-row-per-applicant features from previous applications."""
    require_columns(previous_application, [ENTITY_ID_COLUMN], "previous_application table")
    if previous_application.empty:
        return previous_application[[ENTITY_ID_COLUMN]].drop_duplicates().reset_index(drop=True)

    previous = previous_application.copy()
    features = previous[[ENTITY_ID_COLUMN]].drop_duplicates().reset_index(drop=True)

    for day_column in (
        "DAYS_DECISION",
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ):
        if day_column in previous.columns:
            previous.loc[previous[day_column] == SPECIAL_DAY_PLACEHOLDER, day_column] = None

    if "AMT_CREDIT" in previous.columns and "AMT_APPLICATION" in previous.columns:
        previous[f"{PREVIOUS_PREFIX}credit_application_ratio"] = safe_divide(
            previous["AMT_CREDIT"],
            previous["AMT_APPLICATION"],
        )

    if "AMT_ANNUITY" in previous.columns and "AMT_CREDIT" in previous.columns:
        previous[f"{PREVIOUS_PREFIX}annuity_credit_ratio"] = safe_divide(
            previous["AMT_ANNUITY"],
            previous["AMT_CREDIT"],
        )

    record_count = (
        previous.groupby(ENTITY_ID_COLUMN, dropna=False)
        .size()
        .rename(f"{PREVIOUS_PREFIX}record_count")
        .reset_index()
    )
    features = features.merge(record_count, on=ENTITY_ID_COLUMN, how="left", validate="one_to_one")

    if "NAME_CONTRACT_STATUS" in previous.columns:
        status_counts = (
            previous.assign(
                _status=previous["NAME_CONTRACT_STATUS"].astype("string").fillna("UNKNOWN")
            )
            .groupby([ENTITY_ID_COLUMN, "_status"], dropna=False)
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        status_counts.columns = [
            (
                ENTITY_ID_COLUMN
                if column == ENTITY_ID_COLUMN
                else f"{PREVIOUS_PREFIX}status_{sanitize_token(str(column))}_count"
            )
            for column in status_counts.columns
        ]
        features = features.merge(
            status_counts,
            on=ENTITY_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )

    previous_numeric_columns = [
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_DOWN_PAYMENT",
        "RATE_DOWN_PAYMENT",
        "CNT_PAYMENT",
        "DAYS_DECISION",
        f"{PREVIOUS_PREFIX}credit_application_ratio",
        f"{PREVIOUS_PREFIX}annuity_credit_ratio",
    ]
    previous_agg = aggregated_numeric_features(
        previous,
        group_key=ENTITY_ID_COLUMN,
        numeric_columns=previous_numeric_columns,
        prefix=PREVIOUS_PREFIX,
    )
    features = features.merge(previous_agg, on=ENTITY_ID_COLUMN, how="left", validate="one_to_one")

    assert_unique_entity_rows(features, "previous_application feature block")
    return features
