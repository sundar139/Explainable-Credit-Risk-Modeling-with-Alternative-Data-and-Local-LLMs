"""installments_payments relational feature builder."""

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

INSTALLMENTS_PREFIX = FEATURE_PREFIX_BY_MODULE["installments"]


def build_installments_features(installments_payments: DataFrame) -> DataFrame:
    """Build one-row-per-applicant features from installments_payments."""
    require_columns(installments_payments, [ENTITY_ID_COLUMN], "installments_payments table")
    if installments_payments.empty:
        return installments_payments[[ENTITY_ID_COLUMN]].drop_duplicates().reset_index(drop=True)

    installments = installments_payments.copy()
    features = installments[[ENTITY_ID_COLUMN]].drop_duplicates().reset_index(drop=True)

    if "AMT_PAYMENT" in installments.columns and "AMT_INSTALMENT" in installments.columns:
        installments[f"{INSTALLMENTS_PREFIX}payment_gap"] = (
            installments["AMT_PAYMENT"] - installments["AMT_INSTALMENT"]
        )
        installments[f"{INSTALLMENTS_PREFIX}payment_ratio"] = safe_divide(
            installments["AMT_PAYMENT"],
            installments["AMT_INSTALMENT"],
        )
        installments[f"{INSTALLMENTS_PREFIX}underpayment_flag"] = (
            installments[f"{INSTALLMENTS_PREFIX}payment_gap"] < 0
        ).astype("int8")
        installments[f"{INSTALLMENTS_PREFIX}overpayment_flag"] = (
            installments[f"{INSTALLMENTS_PREFIX}payment_gap"] > 0
        ).astype("int8")

    if "DAYS_ENTRY_PAYMENT" in installments.columns and "DAYS_INSTALMENT" in installments.columns:
        payment_day_gap = installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
        installments[f"{INSTALLMENTS_PREFIX}days_late"] = payment_day_gap.clip(lower=0)
        installments[f"{INSTALLMENTS_PREFIX}days_early"] = (-payment_day_gap).clip(lower=0)
        installments[f"{INSTALLMENTS_PREFIX}late_payment_flag"] = (
            payment_day_gap > 0
        ).astype("int8")

    record_count = (
        installments.groupby(ENTITY_ID_COLUMN, dropna=False)
        .size()
        .rename(f"{INSTALLMENTS_PREFIX}record_count")
        .reset_index()
    )
    features = features.merge(record_count, on=ENTITY_ID_COLUMN, how="left", validate="one_to_one")

    numeric_columns = [
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        f"{INSTALLMENTS_PREFIX}payment_gap",
        f"{INSTALLMENTS_PREFIX}payment_ratio",
        f"{INSTALLMENTS_PREFIX}underpayment_flag",
        f"{INSTALLMENTS_PREFIX}overpayment_flag",
        f"{INSTALLMENTS_PREFIX}days_late",
        f"{INSTALLMENTS_PREFIX}days_early",
        f"{INSTALLMENTS_PREFIX}late_payment_flag",
    ]
    installments_agg = aggregated_numeric_features(
        installments,
        group_key=ENTITY_ID_COLUMN,
        numeric_columns=numeric_columns,
        prefix=INSTALLMENTS_PREFIX,
    )
    features = features.merge(
        installments_agg,
        on=ENTITY_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    assert_unique_entity_rows(features, "installments feature block")
    return features
