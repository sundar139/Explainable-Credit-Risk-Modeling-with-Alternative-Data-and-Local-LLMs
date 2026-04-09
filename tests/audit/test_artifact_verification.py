"""Tests for Phase 9 artifact contract verification."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from credit_risk_altdata.audit.artifacts import verify_artifact_contracts
from credit_risk_altdata.config import Settings
from credit_risk_altdata.modeling.constants import FINAL_PRODUCTION_CANDIDATE_FILE


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_minimum_demo_artifacts(settings: Settings) -> None:
    settings.home_credit_processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"SK_ID_CURR": 700001, "TARGET": 0, "feature_a": 0.3},
            {"SK_ID_CURR": 700002, "TARGET": 1, "feature_a": 0.8},
        ]
    ).to_parquet(settings.home_credit_processed_dir / "train_features.parquet", index=False)
    pd.DataFrame(
        [
            {"SK_ID_CURR": 900001, "feature_a": 0.4},
            {"SK_ID_CURR": 900002, "feature_a": 0.7},
        ]
    ).to_parquet(settings.home_credit_processed_dir / "test_features.parquet", index=False)

    settings.feature_metadata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "feature_name": "SK_ID_CURR",
                "source_module": "identifier",
                "dtype": "int64",
                "null_fraction": 0.0,
                "is_target": False,
                "is_identifier": True,
            },
            {
                "feature_name": "TARGET",
                "source_module": "target",
                "dtype": "int8",
                "null_fraction": 0.0,
                "is_target": True,
                "is_identifier": False,
            },
            {
                "feature_name": "feature_a",
                "source_module": "synthetic",
                "dtype": "float64",
                "null_fraction": 0.0,
                "is_target": False,
                "is_identifier": False,
            },
        ]
    ).to_csv(settings.feature_metadata_dir / "feature_manifest.csv", index=False)

    settings.modeling_reports_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        settings.modeling_reports_dir / "best_model_summary.json",
        {
            "best_model_name": "lightgbm",
            "primary_metric": "roc_auc",
            "primary_metric_value": 0.73,
        },
    )

    settings.modeling_tuning_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_family": "lightgbm",
                "trial_number": 0,
                "state": "COMPLETE",
                "value": 0.78,
            }
        ]
    ).to_csv(settings.modeling_tuning_dir / "tuning_results.csv", index=False)

    settings.modeling_metrics_dir.mkdir(parents=True, exist_ok=True)
    tuned_comparison_path = settings.modeling_metrics_dir / "tuned_model_comparison.csv"
    pd.DataFrame(
        [
            {
                "candidate_name": "lightgbm_tuned_none",
                "model_family": "lightgbm",
                "source": "tuned",
                "is_tuned": True,
                "is_calibrated": False,
                "calibration_method": "none",
                "roc_auc": 0.78,
                "pr_auc": 0.42,
                "f1": 0.36,
                "threshold": 0.5,
                "artifact_path": "artifacts/modeling/models/tuned/lightgbm_tuned_none.joblib",
            }
        ]
    ).to_csv(tuned_comparison_path, index=False)

    selected_artifact = settings.modeling_models_dir / "tuned" / "lightgbm_tuned_none.joblib"
    selected_artifact.parent.mkdir(parents=True, exist_ok=True)
    selected_artifact.write_bytes(b"fake-model")

    final_model_output = settings.modeling_final_model_output_path
    final_model_output.parent.mkdir(parents=True, exist_ok=True)
    final_model_output.write_bytes(b"final-model")

    _write_json(
        settings.modeling_reports_dir / FINAL_PRODUCTION_CANDIDATE_FILE,
        {
            "final_model_family": "lightgbm",
            "final_candidate_name": "lightgbm_tuned_none",
            "tuned": True,
            "calibrated": False,
            "calibration_method": "none",
            "primary_metric": "roc_auc",
            "primary_metric_value": 0.78,
            "threshold": 0.5,
            "selected_artifact_path": str(
                selected_artifact.relative_to(settings.project_root)
            ),
            "final_model_output_path": str(
                final_model_output.relative_to(settings.project_root)
            ),
            "source_comparison_artifact": str(
                tuned_comparison_path.relative_to(settings.project_root)
            ),
        },
    )

    settings.explainability_selected_examples_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"applicant_id": 700001, "cohort_name": "true_positive", "row_index": 0}]
    ).to_csv(settings.explainability_selected_examples_dir / "selected_examples.csv", index=False)

    settings.explainability_reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.explainability_reports_dir / "explainability_summary.md").write_text(
        "# Explainability Summary\n",
        encoding="utf-8",
    )

    settings.explainability_shap_local_dir.mkdir(parents=True, exist_ok=True)
    (settings.explainability_shap_local_dir / "shap_local_explanations.jsonl").write_text(
        '{"applicant_id":700001}\n',
        encoding="utf-8",
    )

    settings.llm_reports_combined_dir.mkdir(parents=True, exist_ok=True)
    (settings.llm_reports_combined_dir / "llm_reports.jsonl").write_text(
        '{"report_id":"rr_1","applicant_id":700001}\n',
        encoding="utf-8",
    )
    settings.llm_reports_reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.llm_reports_reports_dir / "llm_reporting_summary.md").write_text(
        "# LLM Reporting Summary\n",
        encoding="utf-8",
    )


def test_verify_artifact_contracts_reports_missing_required_artifacts(
    synthetic_settings: Settings,
) -> None:
    report = verify_artifact_contracts(synthetic_settings)

    assert report.is_valid is False
    failed_required_names = {
        check.name for check in report.checks if check.required and not check.passed
    }
    assert "final_candidate_summary" in failed_required_names
    assert "processed_train_features" in failed_required_names
    assert report.required_failed_count > 0


def test_verify_artifact_contracts_passes_with_minimum_demo_artifacts(
    synthetic_settings: Settings,
) -> None:
    _write_minimum_demo_artifacts(synthetic_settings)

    report = verify_artifact_contracts(synthetic_settings)

    assert report.is_valid is True
    assert report.required_failed_count == 0


def test_verify_artifact_contracts_detects_final_candidate_mismatch(
    synthetic_settings: Settings,
) -> None:
    _write_minimum_demo_artifacts(synthetic_settings)

    tuned_comparison_path = (
        synthetic_settings.modeling_metrics_dir / "tuned_model_comparison.csv"
    )
    pd.DataFrame(
        [
            {
                "candidate_name": "catboost_tuned_sigmoid",
                "model_family": "catboost",
            }
        ]
    ).to_csv(tuned_comparison_path, index=False)

    report = verify_artifact_contracts(synthetic_settings)
    checks_by_name = {check.name: check for check in report.checks}

    assert report.is_valid is False
    assert checks_by_name["final_candidate_in_tuned_comparison"].passed is False
