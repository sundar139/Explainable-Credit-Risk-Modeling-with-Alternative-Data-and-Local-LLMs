"""credit_card_balance relational feature builder."""

from __future__ import annotations

from pandas import DataFrame

from credit_risk_altdata.features.constants import (
    ENTITY_ID_COLUMN,
    FEATURE_PREFIX_BY_MODULE,
    aggregated_numeric_features,
    assert_unique_entity_rows,
    require_columns,
    safe_divide,
)

CREDIT_CARD_PREFIX = FEATURE_PREFIX_BY_MODULE["credit_card"]


def build_credit_card_features(credit_card_balance: DataFrame) -> DataFrame:
    """Build one-row-per-applicant features from credit_card_balance."""
    require_columns(credit_card_balance, [ENTITY_ID_COLUMN], "credit_card_balance table")
    if credit_card_balance.empty:
        return credit_card_balance[[ENTITY_ID_COLUMN]].drop_duplicates().reset_index(drop=True)

    credit_card = credit_card_balance.copy()
    features = credit_card[[ENTITY_ID_COLUMN]].drop_duplicates().reset_index(drop=True)

    if "AMT_BALANCE" in credit_card.columns and "AMT_CREDIT_LIMIT_ACTUAL" in credit_card.columns:
        credit_card[f"{CREDIT_CARD_PREFIX}utilization_ratio"] = safe_divide(
            credit_card["AMT_BALANCE"],
            credit_card["AMT_CREDIT_LIMIT_ACTUAL"],
        )

    if (
        "AMT_PAYMENT_CURRENT" in credit_card.columns
        and "AMT_TOTAL_RECEIVABLE" in credit_card.columns
    ):
        credit_card[f"{CREDIT_CARD_PREFIX}payment_receivable_ratio"] = safe_divide(
            credit_card["AMT_PAYMENT_CURRENT"],
            credit_card["AMT_TOTAL_RECEIVABLE"],
        )

    if "SK_DPD" in credit_card.columns and "SK_DPD_DEF" in credit_card.columns:
        credit_card[f"{CREDIT_CARD_PREFIX}delinquency_flag"] = (
            (credit_card["SK_DPD"] > 0) | (credit_card["SK_DPD_DEF"] > 0)
        ).astype("int8")

    record_count = (
        credit_card.groupby(ENTITY_ID_COLUMN, dropna=False)
        .size()
        .rename(f"{CREDIT_CARD_PREFIX}record_count")
        .reset_index()
    )
    features = features.merge(record_count, on=ENTITY_ID_COLUMN, how="left", validate="one_to_one")

    numeric_columns = [
        "MONTHS_BALANCE",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_CURRENT",
        "AMT_PAYMENT_CURRENT",
        "AMT_TOTAL_RECEIVABLE",
        "SK_DPD",
        "SK_DPD_DEF",
        f"{CREDIT_CARD_PREFIX}utilization_ratio",
        f"{CREDIT_CARD_PREFIX}payment_receivable_ratio",
        f"{CREDIT_CARD_PREFIX}delinquency_flag",
    ]
    credit_card_agg = aggregated_numeric_features(
        credit_card,
        group_key=ENTITY_ID_COLUMN,
        numeric_columns=numeric_columns,
        prefix=CREDIT_CARD_PREFIX,
    )
    features = features.merge(
        credit_card_agg,
        on=ENTITY_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    assert_unique_entity_rows(features, "credit_card feature block")
    return features
