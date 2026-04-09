"""Unit tests for modular feature builders."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from credit_risk_altdata.config import Settings
from credit_risk_altdata.data.loaders import read_home_credit_table
from credit_risk_altdata.features.base_application import build_application_base_features
from credit_risk_altdata.features.bureau import build_bureau_features
from credit_risk_altdata.features.constants import ENTITY_ID_COLUMN, TARGET_COLUMN
from credit_risk_altdata.features.credit_card import build_credit_card_features
from credit_risk_altdata.features.installments import build_installments_features
from credit_risk_altdata.features.pos_cash import build_pos_cash_features
from credit_risk_altdata.features.previous_application import build_previous_application_features


def test_base_application_builder_preserves_target_and_alignment(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
) -> None:
    write_raw_dataset(synthetic_settings)
    train_raw = read_home_credit_table(synthetic_settings, "application_train.csv")
    test_raw = read_home_credit_table(synthetic_settings, "application_test.csv")

    train_features, test_features = build_application_base_features(train_raw, test_raw)

    assert TARGET_COLUMN in train_features.columns
    assert TARGET_COLUMN not in test_features.columns
    assert train_features[ENTITY_ID_COLUMN].is_unique
    assert test_features[ENTITY_ID_COLUMN].is_unique
    assert [column for column in train_features.columns if column != TARGET_COLUMN] == list(
        test_features.columns
    )
    assert "app_credit_income_ratio" in train_features.columns


def test_base_application_builder_blocks_target_in_test(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
) -> None:
    write_raw_dataset(synthetic_settings)
    train_raw = read_home_credit_table(synthetic_settings, "application_train.csv")
    test_raw = read_home_credit_table(synthetic_settings, "application_test.csv")
    test_raw[TARGET_COLUMN] = 0

    with pytest.raises(ValueError):
        build_application_base_features(train_raw, test_raw)


def test_bureau_builder_outputs_prefixed_one_row_per_entity(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
) -> None:
    write_raw_dataset(synthetic_settings)
    bureau = read_home_credit_table(synthetic_settings, "bureau.csv")
    bureau_balance = read_home_credit_table(synthetic_settings, "bureau_balance.csv")

    features = build_bureau_features(bureau, bureau_balance)

    assert features[ENTITY_ID_COLUMN].is_unique
    feature_columns = [column for column in features.columns if column != ENTITY_ID_COLUMN]
    assert feature_columns
    assert all(column.startswith("bureau_") for column in feature_columns)


def test_previous_builder_outputs_prefixed_one_row_per_entity(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
) -> None:
    write_raw_dataset(synthetic_settings)
    previous = read_home_credit_table(synthetic_settings, "previous_application.csv")

    features = build_previous_application_features(previous)

    assert features[ENTITY_ID_COLUMN].is_unique
    feature_columns = [column for column in features.columns if column != ENTITY_ID_COLUMN]
    assert feature_columns
    assert all(column.startswith("prev_") for column in feature_columns)


def test_pos_credit_and_installments_builders_are_entity_level(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
) -> None:
    write_raw_dataset(synthetic_settings)
    pos = read_home_credit_table(synthetic_settings, "POS_CASH_balance.csv")
    credit_card = read_home_credit_table(synthetic_settings, "credit_card_balance.csv")
    installments = read_home_credit_table(synthetic_settings, "installments_payments.csv")

    pos_features = build_pos_cash_features(pos)
    cc_features = build_credit_card_features(credit_card)
    inst_features = build_installments_features(installments)

    assert pos_features[ENTITY_ID_COLUMN].is_unique
    assert cc_features[ENTITY_ID_COLUMN].is_unique
    assert inst_features[ENTITY_ID_COLUMN].is_unique

    assert all(
        column == ENTITY_ID_COLUMN or column.startswith("pos_")
        for column in pos_features.columns
    )
    assert all(
        column == ENTITY_ID_COLUMN or column.startswith("cc_")
        for column in cc_features.columns
    )
    assert all(
        column == ENTITY_ID_COLUMN or column.startswith("inst_")
        for column in inst_features.columns
    )


def test_relational_blocks_can_be_joined_without_entity_duplication(
    synthetic_settings: Settings,
    write_raw_dataset: Callable[..., None],
) -> None:
    write_raw_dataset(synthetic_settings)
    train_raw = read_home_credit_table(synthetic_settings, "application_train.csv")
    test_raw = read_home_credit_table(synthetic_settings, "application_test.csv")
    bureau = read_home_credit_table(synthetic_settings, "bureau.csv")
    bureau_balance = read_home_credit_table(synthetic_settings, "bureau_balance.csv")

    train_base, test_base = build_application_base_features(train_raw, test_raw)
    bureau_block = build_bureau_features(bureau, bureau_balance)

    merged_train = train_base.merge(
        bureau_block,
        on=ENTITY_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    merged_test = test_base.merge(
        bureau_block,
        on=ENTITY_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    assert merged_train[ENTITY_ID_COLUMN].is_unique
    assert merged_test[ENTITY_ID_COLUMN].is_unique
