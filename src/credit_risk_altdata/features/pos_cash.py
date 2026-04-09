"""POS_CASH_balance relational feature builder."""

from __future__ import annotations

from pandas import DataFrame

from credit_risk_altdata.features.constants import (
    ENTITY_ID_COLUMN,
    FEATURE_PREFIX_BY_MODULE,
    aggregated_numeric_features,
    assert_unique_entity_rows,
    require_columns,
    safe_divide,
    sanitize_token,
)

POS_PREFIX = FEATURE_PREFIX_BY_MODULE["pos_cash"]


def build_pos_cash_features(pos_cash_balance: DataFrame) -> DataFrame:
    """Build one-row-per-applicant features from POS_CASH_balance."""
    require_columns(pos_cash_balance, [ENTITY_ID_COLUMN], "POS_CASH_balance table")
    if pos_cash_balance.empty:
        return pos_cash_balance[[ENTITY_ID_COLUMN]].drop_duplicates().reset_index(drop=True)

    pos_cash = pos_cash_balance.copy()
    features = pos_cash[[ENTITY_ID_COLUMN]].drop_duplicates().reset_index(drop=True)

    if "SK_DPD" in pos_cash.columns and "SK_DPD_DEF" in pos_cash.columns:
        pos_cash[f"{POS_PREFIX}delinquency_flag"] = (
            (pos_cash["SK_DPD"] > 0) | (pos_cash["SK_DPD_DEF"] > 0)
        ).astype("int8")

    if "CNT_INSTALMENT_FUTURE" in pos_cash.columns and "CNT_INSTALMENT" in pos_cash.columns:
        pos_cash[f"{POS_PREFIX}future_to_total_inst_ratio"] = safe_divide(
            pos_cash["CNT_INSTALMENT_FUTURE"],
            pos_cash["CNT_INSTALMENT"],
        )

    record_count = (
        pos_cash.groupby(ENTITY_ID_COLUMN, dropna=False)
        .size()
        .rename(f"{POS_PREFIX}record_count")
        .reset_index()
    )
    features = features.merge(record_count, on=ENTITY_ID_COLUMN, how="left", validate="one_to_one")

    if "NAME_CONTRACT_STATUS" in pos_cash.columns:
        status_counts = (
            pos_cash.assign(
                _status=pos_cash["NAME_CONTRACT_STATUS"].astype("string").fillna("UNKNOWN")
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
                else f"{POS_PREFIX}status_{sanitize_token(str(column))}_count"
            )
            for column in status_counts.columns
        ]
        features = features.merge(
            status_counts,
            on=ENTITY_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )

    numeric_columns = [
        "MONTHS_BALANCE",
        "SK_DPD",
        "SK_DPD_DEF",
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        f"{POS_PREFIX}delinquency_flag",
        f"{POS_PREFIX}future_to_total_inst_ratio",
    ]
    pos_agg = aggregated_numeric_features(
        pos_cash,
        group_key=ENTITY_ID_COLUMN,
        numeric_columns=numeric_columns,
        prefix=POS_PREFIX,
    )
    features = features.merge(pos_agg, on=ENTITY_ID_COLUMN, how="left", validate="one_to_one")

    assert_unique_entity_rows(features, "pos_cash feature block")
    return features
