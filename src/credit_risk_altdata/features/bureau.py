"""Bureau and bureau_balance relational feature builder."""

from __future__ import annotations

from pandas import DataFrame

from credit_risk_altdata.features.constants import (
    ENTITY_ID_COLUMN,
    FEATURE_PREFIX_BY_MODULE,
    aggregated_numeric_features,
    assert_unique_entity_rows,
    require_columns,
    sanitize_token,
)

BUREAU_PREFIX = FEATURE_PREFIX_BY_MODULE["bureau"]


def build_bureau_features(bureau: DataFrame, bureau_balance: DataFrame) -> DataFrame:
    """Build one-row-per-applicant bureau feature aggregates."""
    require_columns(bureau, [ENTITY_ID_COLUMN], "bureau table")
    if bureau.empty:
        return bureau[[ENTITY_ID_COLUMN]].drop_duplicates().reset_index(drop=True)

    bureau_frame = bureau.copy()
    features = bureau_frame[[ENTITY_ID_COLUMN]].drop_duplicates().reset_index(drop=True)

    record_count = (
        bureau_frame.groupby(ENTITY_ID_COLUMN, dropna=False)
        .size()
        .rename(f"{BUREAU_PREFIX}record_count")
        .reset_index()
    )
    features = features.merge(record_count, on=ENTITY_ID_COLUMN, how="left", validate="one_to_one")

    if "CREDIT_ACTIVE" in bureau_frame.columns:
        status_counts = (
            bureau_frame.assign(_status=bureau_frame["CREDIT_ACTIVE"].astype("string").fillna("UNKNOWN"))
            .groupby([ENTITY_ID_COLUMN, "_status"], dropna=False)
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        status_counts.columns = [
            (
                ENTITY_ID_COLUMN
                if column == ENTITY_ID_COLUMN
                else f"{BUREAU_PREFIX}credit_active_{sanitize_token(str(column))}_count"
            )
            for column in status_counts.columns
        ]
        features = features.merge(
            status_counts,
            on=ENTITY_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )

    bureau_numeric_columns = [
        "DAYS_CREDIT",
        "DAYS_CREDIT_ENDDATE",
        "CREDIT_DAY_OVERDUE",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_MAX_OVERDUE",
    ]
    bureau_agg = aggregated_numeric_features(
        bureau_frame,
        group_key=ENTITY_ID_COLUMN,
        numeric_columns=bureau_numeric_columns,
        prefix=f"{BUREAU_PREFIX}",
    )
    features = features.merge(bureau_agg, on=ENTITY_ID_COLUMN, how="left", validate="one_to_one")

    if "SK_ID_BUREAU" in bureau_frame.columns and "SK_ID_BUREAU" in bureau_balance.columns:
        balance_frame = bureau_balance.copy()
        if "STATUS" in balance_frame.columns:
            balance_frame[f"{BUREAU_PREFIX}bb_delinquent_flag"] = (
                ~balance_frame["STATUS"].astype("string").isin(["0", "C", "X"])
            ).astype("int8")

        bb_numeric_columns = ["MONTHS_BALANCE", f"{BUREAU_PREFIX}bb_delinquent_flag"]
        bb_per_bureau = aggregated_numeric_features(
            balance_frame,
            group_key="SK_ID_BUREAU",
            numeric_columns=bb_numeric_columns,
            prefix=f"{BUREAU_PREFIX}bb_",
        )
        if "STATUS" in balance_frame.columns:
            bb_status_nunique = (
                balance_frame.groupby("SK_ID_BUREAU", dropna=False)["STATUS"]
                .nunique(dropna=True)
                .rename(f"{BUREAU_PREFIX}bb_status_nunique")
                .reset_index()
            )
            bb_per_bureau = bb_per_bureau.merge(
                bb_status_nunique,
                on="SK_ID_BUREAU",
                how="left",
                validate="one_to_one",
            )

        bureau_enriched = bureau_frame.merge(
            bb_per_bureau,
            on="SK_ID_BUREAU",
            how="left",
            validate="many_to_one",
        )
        bb_related_columns = [
            column for column in bureau_enriched.columns if column.startswith(f"{BUREAU_PREFIX}bb_")
        ]
        bb_by_curr = aggregated_numeric_features(
            bureau_enriched,
            group_key=ENTITY_ID_COLUMN,
            numeric_columns=bb_related_columns,
            prefix=f"{BUREAU_PREFIX}linked_",
        )
        features = features.merge(
            bb_by_curr,
            on=ENTITY_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )

    assert_unique_entity_rows(features, "bureau feature block")
    return features
