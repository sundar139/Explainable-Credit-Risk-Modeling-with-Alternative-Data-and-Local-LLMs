"""Shared test fixtures for synthetic Home Credit datasets."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk_altdata.config import Settings


def _base_rows_by_file() -> dict[str, list[dict[str, object]]]:
    return {
        "application_train.csv": [
            {
                "SK_ID_CURR": 100001,
                "TARGET": 0,
                "AMT_CREDIT": 500000.0,
                "AMT_INCOME_TOTAL": 180000.0,
                "AMT_ANNUITY": 25000.0,
                "AMT_GOODS_PRICE": 450000.0,
                "CNT_CHILDREN": 1,
                "CNT_FAM_MEMBERS": 3.0,
                "OBS_30_CNT_SOCIAL_CIRCLE": 2.0,
                "OBS_60_CNT_SOCIAL_CIRCLE": 3.0,
                "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
                "DEF_60_CNT_SOCIAL_CIRCLE": 1.0,
                "EXT_SOURCE_1": 0.45,
                "EXT_SOURCE_2": 0.55,
                "EXT_SOURCE_3": 0.65,
                "DAYS_BIRTH": -12000,
                "DAYS_EMPLOYED": -2000,
                "DAYS_LAST_PHONE_CHANGE": -400,
                "CODE_GENDER": "F",
                "FLAG_OWN_CAR": "Y",
                "FLAG_OWN_REALTY": "Y",
                "NAME_CONTRACT_TYPE": "Cash loans",
            },
            {
                "SK_ID_CURR": 100002,
                "TARGET": 1,
                "AMT_CREDIT": 300000.0,
                "AMT_INCOME_TOTAL": 120000.0,
                "AMT_ANNUITY": 18000.0,
                "AMT_GOODS_PRICE": 280000.0,
                "CNT_CHILDREN": 0,
                "CNT_FAM_MEMBERS": 2.0,
                "OBS_30_CNT_SOCIAL_CIRCLE": 1.0,
                "OBS_60_CNT_SOCIAL_CIRCLE": 1.0,
                "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
                "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
                "EXT_SOURCE_1": 0.35,
                "EXT_SOURCE_2": 0.40,
                "EXT_SOURCE_3": None,
                "DAYS_BIRTH": -14000,
                "DAYS_EMPLOYED": 365243,
                "DAYS_LAST_PHONE_CHANGE": -100,
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "N",
                "NAME_CONTRACT_TYPE": "Revolving loans",
            },
        ],
        "application_test.csv": [
            {
                "SK_ID_CURR": 200001,
                "AMT_CREDIT": 420000.0,
                "AMT_INCOME_TOTAL": 150000.0,
                "AMT_ANNUITY": 23000.0,
                "AMT_GOODS_PRICE": 400000.0,
                "CNT_CHILDREN": 2,
                "CNT_FAM_MEMBERS": 4.0,
                "OBS_30_CNT_SOCIAL_CIRCLE": 2.0,
                "OBS_60_CNT_SOCIAL_CIRCLE": 2.0,
                "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
                "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
                "EXT_SOURCE_1": 0.50,
                "EXT_SOURCE_2": 0.60,
                "EXT_SOURCE_3": 0.58,
                "DAYS_BIRTH": -13000,
                "DAYS_EMPLOYED": -1500,
                "DAYS_LAST_PHONE_CHANGE": -250,
                "CODE_GENDER": "F",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
                "NAME_CONTRACT_TYPE": "Cash loans",
            },
            {
                "SK_ID_CURR": 200002,
                "AMT_CREDIT": 250000.0,
                "AMT_INCOME_TOTAL": 90000.0,
                "AMT_ANNUITY": 16000.0,
                "AMT_GOODS_PRICE": 220000.0,
                "CNT_CHILDREN": 0,
                "CNT_FAM_MEMBERS": 1.0,
                "OBS_30_CNT_SOCIAL_CIRCLE": 0.0,
                "OBS_60_CNT_SOCIAL_CIRCLE": 1.0,
                "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
                "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
                "EXT_SOURCE_1": 0.25,
                "EXT_SOURCE_2": 0.30,
                "EXT_SOURCE_3": 0.35,
                "DAYS_BIRTH": -15000,
                "DAYS_EMPLOYED": -800,
                "DAYS_LAST_PHONE_CHANGE": -90,
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "N",
                "NAME_CONTRACT_TYPE": "Cash loans",
            },
        ],
        "bureau.csv": [
            {
                "SK_ID_BUREAU": 300001,
                "SK_ID_CURR": 100001,
                "CREDIT_ACTIVE": "Active",
                "DAYS_CREDIT": -120,
                "DAYS_CREDIT_ENDDATE": 300,
                "CREDIT_DAY_OVERDUE": 0,
                "AMT_CREDIT_SUM": 150000.0,
                "AMT_CREDIT_SUM_DEBT": 70000.0,
                "AMT_CREDIT_MAX_OVERDUE": 0.0,
            },
            {
                "SK_ID_BUREAU": 300002,
                "SK_ID_CURR": 100001,
                "CREDIT_ACTIVE": "Closed",
                "DAYS_CREDIT": -300,
                "DAYS_CREDIT_ENDDATE": -30,
                "CREDIT_DAY_OVERDUE": 12,
                "AMT_CREDIT_SUM": 80000.0,
                "AMT_CREDIT_SUM_DEBT": 0.0,
                "AMT_CREDIT_MAX_OVERDUE": 1200.0,
            },
            {
                "SK_ID_BUREAU": 300003,
                "SK_ID_CURR": 100002,
                "CREDIT_ACTIVE": "Active",
                "DAYS_CREDIT": -60,
                "DAYS_CREDIT_ENDDATE": 500,
                "CREDIT_DAY_OVERDUE": 0,
                "AMT_CREDIT_SUM": 60000.0,
                "AMT_CREDIT_SUM_DEBT": 15000.0,
                "AMT_CREDIT_MAX_OVERDUE": 0.0,
            },
            {
                "SK_ID_BUREAU": 300004,
                "SK_ID_CURR": 200001,
                "CREDIT_ACTIVE": "Sold",
                "DAYS_CREDIT": -200,
                "DAYS_CREDIT_ENDDATE": 50,
                "CREDIT_DAY_OVERDUE": 5,
                "AMT_CREDIT_SUM": 50000.0,
                "AMT_CREDIT_SUM_DEBT": 10000.0,
                "AMT_CREDIT_MAX_OVERDUE": 500.0,
            },
        ],
        "bureau_balance.csv": [
            {"SK_ID_BUREAU": 300001, "MONTHS_BALANCE": -1, "STATUS": "0"},
            {"SK_ID_BUREAU": 300001, "MONTHS_BALANCE": -2, "STATUS": "1"},
            {"SK_ID_BUREAU": 300002, "MONTHS_BALANCE": -1, "STATUS": "C"},
            {"SK_ID_BUREAU": 300003, "MONTHS_BALANCE": -1, "STATUS": "2"},
            {"SK_ID_BUREAU": 300004, "MONTHS_BALANCE": -1, "STATUS": "0"},
        ],
        "previous_application.csv": [
            {
                "SK_ID_PREV": 400001,
                "SK_ID_CURR": 100001,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "NAME_CONTRACT_STATUS": "Approved",
                "AMT_APPLICATION": 200000.0,
                "AMT_CREDIT": 210000.0,
                "AMT_ANNUITY": 12000.0,
                "AMT_DOWN_PAYMENT": 10000.0,
                "RATE_DOWN_PAYMENT": 0.05,
                "CNT_PAYMENT": 24,
                "DAYS_DECISION": -100,
            },
            {
                "SK_ID_PREV": 400002,
                "SK_ID_CURR": 100001,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "NAME_CONTRACT_STATUS": "Refused",
                "AMT_APPLICATION": 150000.0,
                "AMT_CREDIT": 130000.0,
                "AMT_ANNUITY": 9000.0,
                "AMT_DOWN_PAYMENT": 5000.0,
                "RATE_DOWN_PAYMENT": 0.03,
                "CNT_PAYMENT": 12,
                "DAYS_DECISION": -300,
            },
            {
                "SK_ID_PREV": 400003,
                "SK_ID_CURR": 100002,
                "NAME_CONTRACT_TYPE": "Consumer loans",
                "NAME_CONTRACT_STATUS": "Approved",
                "AMT_APPLICATION": 100000.0,
                "AMT_CREDIT": 98000.0,
                "AMT_ANNUITY": 8000.0,
                "AMT_DOWN_PAYMENT": 2000.0,
                "RATE_DOWN_PAYMENT": 0.02,
                "CNT_PAYMENT": 10,
                "DAYS_DECISION": -200,
            },
            {
                "SK_ID_PREV": 400004,
                "SK_ID_CURR": 200001,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "NAME_CONTRACT_STATUS": "Approved",
                "AMT_APPLICATION": 180000.0,
                "AMT_CREDIT": 175000.0,
                "AMT_ANNUITY": 11000.0,
                "AMT_DOWN_PAYMENT": 7000.0,
                "RATE_DOWN_PAYMENT": 0.04,
                "CNT_PAYMENT": 20,
                "DAYS_DECISION": -90,
            },
            {
                "SK_ID_PREV": 400005,
                "SK_ID_CURR": 200002,
                "NAME_CONTRACT_TYPE": "Consumer loans",
                "NAME_CONTRACT_STATUS": "Canceled",
                "AMT_APPLICATION": 120000.0,
                "AMT_CREDIT": 100000.0,
                "AMT_ANNUITY": 7000.0,
                "AMT_DOWN_PAYMENT": 3000.0,
                "RATE_DOWN_PAYMENT": 0.025,
                "CNT_PAYMENT": 18,
                "DAYS_DECISION": -60,
            },
        ],
        "POS_CASH_balance.csv": [
            {
                "SK_ID_PREV": 400001,
                "SK_ID_CURR": 100001,
                "MONTHS_BALANCE": -1,
                "CNT_INSTALMENT": 12,
                "CNT_INSTALMENT_FUTURE": 8,
                "SK_DPD": 0,
                "SK_DPD_DEF": 0,
                "NAME_CONTRACT_STATUS": "Active",
            },
            {
                "SK_ID_PREV": 400001,
                "SK_ID_CURR": 100001,
                "MONTHS_BALANCE": -2,
                "CNT_INSTALMENT": 12,
                "CNT_INSTALMENT_FUTURE": 9,
                "SK_DPD": 5,
                "SK_DPD_DEF": 1,
                "NAME_CONTRACT_STATUS": "Active",
            },
            {
                "SK_ID_PREV": 400003,
                "SK_ID_CURR": 100002,
                "MONTHS_BALANCE": -1,
                "CNT_INSTALMENT": 10,
                "CNT_INSTALMENT_FUTURE": 7,
                "SK_DPD": 0,
                "SK_DPD_DEF": 0,
                "NAME_CONTRACT_STATUS": "Completed",
            },
            {
                "SK_ID_PREV": 400004,
                "SK_ID_CURR": 200001,
                "MONTHS_BALANCE": -1,
                "CNT_INSTALMENT": 15,
                "CNT_INSTALMENT_FUTURE": 11,
                "SK_DPD": 3,
                "SK_DPD_DEF": 0,
                "NAME_CONTRACT_STATUS": "Active",
            },
        ],
        "credit_card_balance.csv": [
            {
                "SK_ID_PREV": 400001,
                "SK_ID_CURR": 100001,
                "MONTHS_BALANCE": -1,
                "AMT_BALANCE": 1000.0,
                "AMT_CREDIT_LIMIT_ACTUAL": 10000.0,
                "AMT_DRAWINGS_CURRENT": 500.0,
                "AMT_PAYMENT_CURRENT": 400.0,
                "AMT_TOTAL_RECEIVABLE": 800.0,
                "SK_DPD": 0,
                "SK_DPD_DEF": 0,
            },
            {
                "SK_ID_PREV": 400001,
                "SK_ID_CURR": 100001,
                "MONTHS_BALANCE": -2,
                "AMT_BALANCE": 1200.0,
                "AMT_CREDIT_LIMIT_ACTUAL": 10000.0,
                "AMT_DRAWINGS_CURRENT": 600.0,
                "AMT_PAYMENT_CURRENT": 300.0,
                "AMT_TOTAL_RECEIVABLE": 950.0,
                "SK_DPD": 2,
                "SK_DPD_DEF": 1,
            },
            {
                "SK_ID_PREV": 400003,
                "SK_ID_CURR": 100002,
                "MONTHS_BALANCE": -1,
                "AMT_BALANCE": 500.0,
                "AMT_CREDIT_LIMIT_ACTUAL": 8000.0,
                "AMT_DRAWINGS_CURRENT": 300.0,
                "AMT_PAYMENT_CURRENT": 200.0,
                "AMT_TOTAL_RECEIVABLE": 400.0,
                "SK_DPD": 0,
                "SK_DPD_DEF": 0,
            },
            {
                "SK_ID_PREV": 400004,
                "SK_ID_CURR": 200001,
                "MONTHS_BALANCE": -1,
                "AMT_BALANCE": 700.0,
                "AMT_CREDIT_LIMIT_ACTUAL": 9000.0,
                "AMT_DRAWINGS_CURRENT": 350.0,
                "AMT_PAYMENT_CURRENT": 220.0,
                "AMT_TOTAL_RECEIVABLE": 600.0,
                "SK_DPD": 0,
                "SK_DPD_DEF": 0,
            },
        ],
        "installments_payments.csv": [
            {
                "SK_ID_PREV": 400001,
                "SK_ID_CURR": 100001,
                "NUM_INSTALMENT_VERSION": 1,
                "NUM_INSTALMENT_NUMBER": 1,
                "AMT_INSTALMENT": 520.0,
                "AMT_PAYMENT": 500.0,
                "DAYS_INSTALMENT": -30,
                "DAYS_ENTRY_PAYMENT": -28,
            },
            {
                "SK_ID_PREV": 400001,
                "SK_ID_CURR": 100001,
                "NUM_INSTALMENT_VERSION": 1,
                "NUM_INSTALMENT_NUMBER": 2,
                "AMT_INSTALMENT": 560.0,
                "AMT_PAYMENT": 550.0,
                "DAYS_INSTALMENT": -60,
                "DAYS_ENTRY_PAYMENT": -62,
            },
            {
                "SK_ID_PREV": 400003,
                "SK_ID_CURR": 100002,
                "NUM_INSTALMENT_VERSION": 1,
                "NUM_INSTALMENT_NUMBER": 1,
                "AMT_INSTALMENT": 410.0,
                "AMT_PAYMENT": 420.0,
                "DAYS_INSTALMENT": -20,
                "DAYS_ENTRY_PAYMENT": -20,
            },
            {
                "SK_ID_PREV": 400004,
                "SK_ID_CURR": 200001,
                "NUM_INSTALMENT_VERSION": 1,
                "NUM_INSTALMENT_NUMBER": 1,
                "AMT_INSTALMENT": 600.0,
                "AMT_PAYMENT": 500.0,
                "DAYS_INSTALMENT": -15,
                "DAYS_ENTRY_PAYMENT": -10,
            },
        ],
    }


@pytest.fixture
def synthetic_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_env="test",
        project_root=tmp_path,
        data_dir=Path("data"),
        artifacts_dir=Path("artifacts"),
    )


@pytest.fixture
def write_raw_dataset() -> Callable[..., None]:
    def _write(
        settings: Settings,
        *,
        missing_files: set[str] | None = None,
        duplicate_application_curr: bool = False,
        drop_bureau_key_column: bool = False,
    ) -> None:
        raw_dir = settings.home_credit_raw_dir
        raw_dir.mkdir(parents=True, exist_ok=True)

        rows_by_file = _base_rows_by_file()
        if duplicate_application_curr:
            rows_by_file["application_train.csv"].append(
                {"SK_ID_CURR": 100001, "TARGET": 0}
            )

        if drop_bureau_key_column:
            for row in rows_by_file["bureau.csv"]:
                row.pop("SK_ID_BUREAU", None)

        skipped = missing_files or set()
        for file_name, rows in rows_by_file.items():
            if file_name in skipped:
                continue
            dataframe = pd.DataFrame(rows)
            dataframe.to_csv(raw_dir / file_name, index=False)

    return _write


@pytest.fixture
def write_processed_features() -> Callable[..., None]:
    def _write(
        settings: Settings,
        *,
        n_train: int = 120,
        n_test: int = 40,
        include_categorical: bool = True,
        duplicate_train_ids: bool = False,
        drop_target: bool = False,
        mismatched_test_columns: bool = False,
    ) -> None:
        processed_dir = settings.home_credit_processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(seed=42)
        total_rows = n_train + n_test
        feature_1 = rng.normal(loc=0.0, scale=1.0, size=total_rows)
        feature_2 = rng.normal(loc=0.2, scale=1.2, size=total_rows)
        feature_3 = rng.normal(loc=-0.3, scale=0.8, size=total_rows)
        category = rng.choice(["A", "B", "C"], size=total_rows)

        linear_term = (
            1.2 * feature_1
            - 0.8 * feature_2
            + 0.4 * feature_3
            + (category == "A") * 0.6
            + rng.normal(loc=0.0, scale=0.5, size=total_rows)
        )
        probability = 1.0 / (1.0 + np.exp(-linear_term))
        train_target = (probability[:n_train] > 0.5).astype(int)
        if train_target.min() == train_target.max():
            train_target[: max(1, n_train // 2)] = 0
            train_target[max(1, n_train // 2) :] = 1

        train_ids = np.arange(700000, 700000 + n_train, dtype=int)
        test_ids = np.arange(900000, 900000 + n_test, dtype=int)
        if duplicate_train_ids and n_train > 1:
            train_ids[1] = train_ids[0]

        train_frame = pd.DataFrame(
            {
                "SK_ID_CURR": train_ids,
                "app_credit_income_ratio": feature_1[:n_train],
                "bureau_record_count": np.abs(feature_2[:n_train] * 3.0) + 1.0,
                "inst_payment_gap_mean": feature_3[:n_train],
            }
        )
        test_frame = pd.DataFrame(
            {
                "SK_ID_CURR": test_ids,
                "app_credit_income_ratio": feature_1[n_train:],
                "bureau_record_count": np.abs(feature_2[n_train:] * 3.0) + 1.0,
                "inst_payment_gap_mean": feature_3[n_train:],
            }
        )

        if include_categorical:
            train_frame["app_contract_type"] = category[:n_train]
            test_frame["app_contract_type"] = category[n_train:]

        if not drop_target:
            train_frame["TARGET"] = train_target

        if mismatched_test_columns:
            test_frame = test_frame.drop(columns=["inst_payment_gap_mean"])

        train_frame.to_parquet(processed_dir / "train_features.parquet", index=False)
        test_frame.to_parquet(processed_dir / "test_features.parquet", index=False)

    return _write
