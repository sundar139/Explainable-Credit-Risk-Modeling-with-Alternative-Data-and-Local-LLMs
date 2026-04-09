"""Tests for final production candidate selection logic."""

from __future__ import annotations

import pandas as pd

from credit_risk_altdata.modeling.final_selection import select_final_candidate


def test_select_final_candidate_prefers_primary_metric_then_tiebreakers() -> None:
    comparison = pd.DataFrame(
        [
            {
                "candidate_name": "lightgbm_tuned_none",
                "model_family": "lightgbm",
                "is_tuned": True,
                "is_calibrated": False,
                "calibration_method": "none",
                "roc_auc": 0.78,
                "pr_auc": 0.40,
                "f1": 0.33,
                "threshold": 0.5,
                "artifact_path": "artifacts/modeling/models/tuned/lightgbm_tuned_none.joblib",
            },
            {
                "candidate_name": "catboost_tuned_sigmoid",
                "model_family": "catboost",
                "is_tuned": True,
                "is_calibrated": True,
                "calibration_method": "sigmoid",
                "roc_auc": 0.78,
                "pr_auc": 0.42,
                "f1": 0.35,
                "threshold": 0.5,
                "artifact_path": "artifacts/modeling/models/tuned/catboost_tuned_sigmoid.joblib",
            },
        ]
    )

    summary = select_final_candidate(
        comparison,
        primary_metric="roc_auc",
        source_comparison_artifact="artifacts/modeling/metrics/tuned_model_comparison.csv",
    )

    assert summary["final_candidate_name"] == "catboost_tuned_sigmoid"
    assert summary["final_model_family"] == "catboost"
    assert summary["tuned"] is True
    assert summary["calibrated"] is True
    assert summary["justification"]
    assert summary["source_comparison_artifact"] == (
        "artifacts/modeling/metrics/tuned_model_comparison.csv"
    )
