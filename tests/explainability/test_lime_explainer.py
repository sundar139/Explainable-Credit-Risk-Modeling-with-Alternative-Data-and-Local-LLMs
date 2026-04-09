"""Tests for robust LIME preprocessing and failure handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from pytest import MonkeyPatch

from credit_risk_altdata.explainability import lime_explainer
from credit_risk_altdata.explainability.lime_explainer import (
    _prepare_lime_explainer_matrix,
    generate_lime_local_artifacts,
)


class _DummyModel:
    def predict_proba(self, x_frame: DataFrame) -> np.ndarray:
        positive = np.full(len(x_frame), 0.6, dtype=np.float64)
        negative = 1.0 - positive
        return np.vstack([negative, positive]).T


def _selected_examples(row_indices: list[int]) -> DataFrame:
    records: list[dict[str, Any]] = []
    for offset, row_index in enumerate(row_indices):
        records.append(
            {
                "row_index": row_index,
                "applicant_id": 700001 + offset,
                "cohort_name": "true_positive",
                "split_name": "train_oof",
                "predicted_probability": 0.82,
                "predicted_label": 1,
                "actual_label": 1,
            }
        )
    return pd.DataFrame(records)


def test_prepare_lime_explainer_matrix_filters_unstable_and_imputes() -> None:
    x_train = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "TARGET": [0, 1, 0, 1],
            "stable_signal": [0.1, 0.3, np.nan, 0.8],
            "zero_variance": [5.0, 5.0, 5.0, 5.0],
            "near_zero_variance": [1.0, 1.0 + 1e-9, 1.0 - 1e-9, 1.0],
            "all_null": [np.nan, np.nan, np.nan, np.nan],
            "contains_inf": [1.0, np.inf, 3.0, -np.inf],
            "non_numeric": ["A", "B", "C", "D"],
        }
    )

    preparation = _prepare_lime_explainer_matrix(
        x_train,
        near_zero_variance_threshold=1e-8,
    )

    assert preparation.frame.columns.tolist() == ["stable_signal", "contains_inf"]
    assert preparation.non_numeric_removed_count == 1
    assert preparation.identifier_target_removed_count == 2
    assert preparation.instability_removed_count >= 3
    assert preparation.final_feature_count == 2
    assert not preparation.frame.isna().any().any()
    assert np.isfinite(np.asarray(preparation.frame, dtype=np.float64)).all()


def test_generate_lime_local_artifacts_degrades_gracefully_on_degenerate_matrix(
    tmp_path: Path,
) -> None:
    x_train = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "TARGET": [0, 1, 0],
            "all_null": [np.nan, np.nan, np.nan],
            "constant": [7.0, 7.0, 7.0],
        }
    )

    payloads, output_path, case_paths = generate_lime_local_artifacts(
        model=_DummyModel(),
        x_train=x_train,
        selected_examples=_selected_examples([0, 1]),
        top_k=5,
        threshold=0.5,
        random_seed=42,
        output_dir=tmp_path,
        model_metadata={"model_family": "lightgbm"},
        categorical_columns=[],
    )

    assert output_path.exists()
    assert len(payloads) == 2
    assert len(case_paths) == 2
    assert all(path.exists() for path in case_paths)
    assert all(payload["explanation_generated"] is False for payload in payloads)
    assert all(
        "No stable numeric features remain" in str(payload["failure_reason"])
        for payload in payloads
    )
    assert all(int(payload["failed_feature_count"]) >= 1 for payload in payloads)


def test_generate_lime_local_artifacts_partial_success_when_single_case_fails(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    class _FakeExplanation:
        local_pred = np.asarray([0.62], dtype=np.float64)
        intercept = np.asarray([0.41, 0.59], dtype=np.float64)
        score = 0.9

        def as_list(self, label: int = 1) -> list[tuple[str, float]]:
            assert label == 1
            return [("signal_a <= 0.25", 0.17), ("signal_b > 1.5", -0.08)]

    class _FailSecondCaseExplainer:
        def __init__(self, *_: Any, **__: Any) -> None:
            self._calls = 0

        def explain_instance(self, *_: Any, **__: Any) -> _FakeExplanation:
            self._calls += 1
            if self._calls == 2:
                raise ValueError("synthetic lime case failure")
            return _FakeExplanation()

    monkeypatch.setattr(lime_explainer, "LimeTabularExplainer", _FailSecondCaseExplainer)

    x_train = pd.DataFrame(
        {
            "signal_a": [0.1, 0.2, 0.3],
            "signal_b": [1.0, 2.0, 3.0],
        }
    )

    payloads, output_path, case_paths = generate_lime_local_artifacts(
        model=_DummyModel(),
        x_train=x_train,
        selected_examples=_selected_examples([0, 1]),
        top_k=5,
        threshold=0.5,
        random_seed=42,
        output_dir=tmp_path,
        model_metadata={"model_family": "lightgbm"},
        categorical_columns=[],
    )

    assert output_path.exists()
    assert len(case_paths) == 2
    assert len(payloads) == 2
    assert payloads[0]["explanation_generated"] is True
    assert payloads[0]["failure_reason"] is None
    assert payloads[1]["explanation_generated"] is False
    assert "synthetic lime case failure" in str(payloads[1]["failure_reason"])

    jsonl_lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(jsonl_lines) == 2
