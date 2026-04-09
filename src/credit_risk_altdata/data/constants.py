"""Constants for Home Credit dataset processing."""

from __future__ import annotations

from dataclasses import dataclass

HOME_CREDIT_COMPETITION = "home-credit-default-risk"
HOME_CREDIT_RAW_SUBDIR = "home_credit"
HOME_CREDIT_INTERIM_SUBDIR = "home_credit"

CORE_RAW_FILES: tuple[str, ...] = (
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
)

REQUIRED_KEY_COLUMNS: dict[str, tuple[str, ...]] = {
    "application_train.csv": ("SK_ID_CURR",),
    "application_test.csv": ("SK_ID_CURR",),
    "bureau.csv": ("SK_ID_BUREAU", "SK_ID_CURR"),
    "bureau_balance.csv": ("SK_ID_BUREAU",),
    "previous_application.csv": ("SK_ID_PREV", "SK_ID_CURR"),
    "POS_CASH_balance.csv": ("SK_ID_PREV", "SK_ID_CURR"),
    "credit_card_balance.csv": ("SK_ID_PREV", "SK_ID_CURR"),
    "installments_payments.csv": ("SK_ID_PREV", "SK_ID_CURR"),
}


@dataclass(frozen=True, slots=True)
class DuplicateCheckRule:
    """Duplicate-check rule for a key set."""

    keys: tuple[str, ...]
    expect_unique: bool


DUPLICATE_CHECK_RULES: dict[str, tuple[DuplicateCheckRule, ...]] = {
    "application_train.csv": (
        DuplicateCheckRule(keys=("SK_ID_CURR",), expect_unique=True),
    ),
    "application_test.csv": (
        DuplicateCheckRule(keys=("SK_ID_CURR",), expect_unique=True),
    ),
    "bureau.csv": (
        DuplicateCheckRule(keys=("SK_ID_BUREAU",), expect_unique=True),
        DuplicateCheckRule(keys=("SK_ID_CURR",), expect_unique=False),
    ),
    "bureau_balance.csv": (
        DuplicateCheckRule(keys=("SK_ID_BUREAU",), expect_unique=False),
        DuplicateCheckRule(keys=("SK_ID_BUREAU", "MONTHS_BALANCE"), expect_unique=True),
    ),
    "previous_application.csv": (
        DuplicateCheckRule(keys=("SK_ID_PREV",), expect_unique=True),
        DuplicateCheckRule(keys=("SK_ID_CURR",), expect_unique=False),
    ),
    "POS_CASH_balance.csv": (
        DuplicateCheckRule(keys=("SK_ID_CURR",), expect_unique=False),
        DuplicateCheckRule(keys=("SK_ID_PREV", "MONTHS_BALANCE"), expect_unique=True),
    ),
    "credit_card_balance.csv": (
        DuplicateCheckRule(keys=("SK_ID_CURR",), expect_unique=False),
        DuplicateCheckRule(keys=("SK_ID_PREV", "MONTHS_BALANCE"), expect_unique=True),
    ),
    "installments_payments.csv": (
        DuplicateCheckRule(keys=("SK_ID_CURR",), expect_unique=False),
        DuplicateCheckRule(keys=("SK_ID_PREV",), expect_unique=False),
    ),
}
