"""Tests for Phase 6 explainability workflow orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest import MonkeyPatch

from credit_risk_altdata.config import Settings
from credit_risk_altdata.explainability import workflow
from credit_risk_altdata.modeling.constants import CALIBRATION_NONE, MODEL_LIGHTGBM
from credit_risk_altdata.modeling.tuning import run_tuned_modeling


def _write_phase6_inputs(
    settings: Settings,
    write_processed_features: Callable[..., None],
) -> None:
    write_processed_features(settings, n_train=60, n_test=20)

    train_frame = pd.read_parquet(settings.home_credit_processed_dir / "train_features.parquet")
    candidate_name = "lightgbm_tuned_none"

    settings.modeling_final_candidate_summary_path.parent.mkdir(parents=True, exist_ok=True)
    final_candidate_summary = {
        "final_candidate_name": candidate_name,
        "final_model_family": "lightgbm",
        "primary_metric": "roc_auc",
        "threshold": 0.5,
        "tuned": True,
        "calibrated": False,
        "selected_artifact_path": "artifacts/modeling/models/tuned/lightgbm_tuned_none.joblib",
        "justification": "Synthetic test candidate",
        "source_comparison_artifact": "artifacts/modeling/metrics/tuned_model_comparison.csv",
        "final_model_output_path": str(
            settings.modeling_final_model_output_path.relative_to(settings.project_root)
        ),
        "training_timestamp": "2026-01-01T00:00:00+00:00",
    }
    settings.modeling_final_candidate_summary_path.write_text(
        json.dumps(final_candidate_summary, indent=2),
        encoding="utf-8",
    )

    settings.modeling_final_model_output_path.parent.mkdir(parents=True, exist_ok=True)
    settings.modeling_final_model_output_path.write_bytes(b"stub-model")

    settings.explainability_input_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    oof_predictions = pd.DataFrame(
        {
            "SK_ID_CURR": train_frame["SK_ID_CURR"].astype(int),
            "TARGET": train_frame["TARGET"].astype(int),
            f"oof_pred_{candidate_name}": np.linspace(0.05, 0.95, len(train_frame)),
        }
    )
    oof_predictions.to_parquet(settings.explainability_input_predictions_path, index=False)


def test_run_explainability_workflow_fails_when_final_candidate_missing(
    synthetic_settings: Settings,
) -> None:
    with pytest.raises(FileNotFoundError):
        workflow.run_explainability_workflow(
            synthetic_settings,
            method_selection="shap",
            overwrite=True,
        )


def test_run_explainability_workflow_ignores_noncanonical_candidate_path(
    synthetic_settings: Settings,
) -> None:
    legacy_candidate_path = synthetic_settings.modeling_dir / "final_production_candidate.json"
    legacy_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_candidate_path.write_text(
        json.dumps(
            {
                "final_candidate_name": "legacy",
                "final_model_family": "lightgbm",
                "threshold": 0.5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        workflow.run_explainability_workflow(
            synthetic_settings,
            method_selection="shap",
            overwrite=True,
        )


def test_run_explainability_workflow_with_stubbed_explainers(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
    monkeypatch: MonkeyPatch,
) -> None:
    _write_phase6_inputs(synthetic_settings, write_processed_features)

    monkeypatch.setattr(workflow, "_load_model", lambda path: object())

    def fake_generate_shap_global_artifacts(
        **kwargs: object,
    ) -> tuple[dict[str, Path], dict[str, object]]:
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "shap_global_summary": output_dir / "shap_global_summary.json",
            "shap_feature_importance": output_dir / "shap_feature_importance.csv",
            "shap_summary_plot": output_dir / "shap_summary_plot.png",
            "shap_bar_plot": output_dir / "shap_bar_plot.png",
        }
        paths["shap_global_summary"].write_text("{}", encoding="utf-8")
        paths["shap_feature_importance"].write_text(
            "feature_name,mean_abs_shap\nfeature_1,0.1\n",
            encoding="utf-8",
        )
        paths["shap_summary_plot"].write_bytes(b"png")
        paths["shap_bar_plot"].write_bytes(b"png")
        return paths, {"row_count": 10}

    def fake_generate_shap_local_artifacts(
        **kwargs: object,
    ) -> tuple[list[dict[str, object]], Path]:
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "shap_local_explanations.jsonl"
        output_path.write_text('{"applicant_id": 1}\n', encoding="utf-8")
        return [{"applicant_id": 1}], output_path

    def fake_generate_lime_local_artifacts(
        **kwargs: object,
    ) -> tuple[list[dict[str, object]], Path, list[Path]]:
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "lime_explanations.jsonl"
        output_path.write_text('{"applicant_id": 1}\n', encoding="utf-8")
        case_path = output_dir / "lime_case_1.md"
        case_path.write_text("# LIME Local Explanation\n", encoding="utf-8")
        return [{"applicant_id": 1}], output_path, [case_path]

    monkeypatch.setattr(
        workflow,
        "generate_shap_global_artifacts",
        fake_generate_shap_global_artifacts,
    )
    monkeypatch.setattr(
        workflow,
        "generate_shap_local_artifacts",
        fake_generate_shap_local_artifacts,
    )
    monkeypatch.setattr(
        workflow,
        "generate_lime_local_artifacts",
        fake_generate_lime_local_artifacts,
    )

    result = workflow.run_explainability_workflow(
        synthetic_settings,
        method_selection="all",
        sample_size=25,
        top_k=5,
        overwrite=True,
    )

    assert result.selected_examples_path.exists()
    assert result.shap_global_summary_path is not None
    assert result.shap_global_summary_path.exists()
    assert result.shap_feature_importance_path is not None
    assert result.shap_feature_importance_path.exists()
    assert result.shap_local_explanations_path is not None
    assert result.shap_local_explanations_path.exists()
    assert result.lime_explanations_path is not None and result.lime_explanations_path.exists()
    assert result.explainability_summary_path.exists()

    selected_examples = pd.read_csv(result.selected_examples_path)
    assert not selected_examples.empty
    assert "row_index" in selected_examples.columns

    summary_markdown = result.explainability_summary_path.read_text(encoding="utf-8")
    assert "Methods: shap, lime" in summary_markdown
    assert "Selected examples" in summary_markdown


def test_run_explainability_workflow_fails_when_input_path_missing(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
    monkeypatch: MonkeyPatch,
) -> None:
    _write_phase6_inputs(synthetic_settings, write_processed_features)
    monkeypatch.setattr(workflow, "_load_model", lambda path: object())

    with pytest.raises(FileNotFoundError):
        workflow.run_explainability_workflow(
            synthetic_settings,
            method_selection="lime",
            input_path_override=Path("artifacts/modeling/predictions/missing_input.parquet"),
            overwrite=True,
        )


def test_run_explainability_workflow_reports_lime_partial_success(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
    monkeypatch: MonkeyPatch,
) -> None:
    _write_phase6_inputs(synthetic_settings, write_processed_features)

    monkeypatch.setattr(workflow, "_load_model", lambda path: object())

    def fake_generate_shap_global_artifacts(
        **kwargs: object,
    ) -> tuple[dict[str, Path], dict[str, object]]:
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "shap_global_summary": output_dir / "shap_global_summary.json",
            "shap_feature_importance": output_dir / "shap_feature_importance.csv",
            "shap_summary_plot": output_dir / "shap_summary_plot.png",
            "shap_bar_plot": output_dir / "shap_bar_plot.png",
        }
        paths["shap_global_summary"].write_text("{}", encoding="utf-8")
        paths["shap_feature_importance"].write_text(
            "feature_name,mean_abs_shap\nfeature_1,0.1\n",
            encoding="utf-8",
        )
        paths["shap_summary_plot"].write_bytes(b"png")
        paths["shap_bar_plot"].write_bytes(b"png")
        return paths, {"row_count": 10}

    def fake_generate_shap_local_artifacts(
        **kwargs: object,
    ) -> tuple[list[dict[str, object]], Path]:
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "shap_local_explanations.jsonl"
        output_path.write_text('{"applicant_id": 1}\n', encoding="utf-8")
        return [{"applicant_id": 1}], output_path

    def fake_generate_lime_local_artifacts(
        **kwargs: object,
    ) -> tuple[list[dict[str, object]], Path, list[Path]]:
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        output_dir.mkdir(parents=True, exist_ok=True)
        payloads: list[dict[str, object]] = [
            {
                "applicant_id": 700001,
                "explanation_generated": True,
                "failure_reason": None,
                "failed_feature_count": 0,
            },
            {
                "applicant_id": 700002,
                "explanation_generated": False,
                "failure_reason": "ValueError: synthetic lime failure",
                "failed_feature_count": 3,
            },
        ]
        output_path = output_dir / "lime_explanations.jsonl"
        output_path.write_text(
            "\n".join(json.dumps(payload) for payload in payloads) + "\n",
            encoding="utf-8",
        )
        return payloads, output_path, []

    monkeypatch.setattr(
        workflow,
        "generate_shap_global_artifacts",
        fake_generate_shap_global_artifacts,
    )
    monkeypatch.setattr(
        workflow,
        "generate_shap_local_artifacts",
        fake_generate_shap_local_artifacts,
    )
    monkeypatch.setattr(
        workflow,
        "generate_lime_local_artifacts",
        fake_generate_lime_local_artifacts,
    )

    result = workflow.run_explainability_workflow(
        synthetic_settings,
        method_selection="all",
        sample_size=25,
        top_k=5,
        overwrite=True,
    )

    assert result.shap_global_summary_path is not None
    assert result.shap_global_summary_path.exists()
    assert result.shap_local_explanations_path is not None
    assert result.shap_local_explanations_path.exists()
    assert result.lime_explanations_path is not None
    assert result.lime_explanations_path.exists()

    summary_markdown = result.explainability_summary_path.read_text(encoding="utf-8")
    assert "LIME status: generated=1 failed=1 total=2" in summary_markdown


def test_run_explainability_workflow_loads_phase5_final_candidate_artifact(
    synthetic_settings: Settings,
    write_processed_features: Callable[..., None],
    monkeypatch: MonkeyPatch,
) -> None:
    settings = synthetic_settings.model_copy(
        update={
            "modeling_folds": 3,
            "modeling_threshold_grid_min": 0.2,
            "modeling_threshold_grid_max": 0.8,
            "modeling_threshold_grid_step": 0.2,
        }
    )
    write_processed_features(settings, n_train=90, n_test=20)

    run_tuned_modeling(
        settings,
        model_selection=MODEL_LIGHTGBM,
        n_trials=1,
        calibration_selection=CALIBRATION_NONE,
        overwrite=True,
    )
    assert settings.modeling_final_candidate_summary_path.exists()

    monkeypatch.setattr(workflow, "_load_model", lambda path: object())

    def fake_generate_shap_global_artifacts(
        **kwargs: object,
    ) -> tuple[dict[str, Path], dict[str, object]]:
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "shap_global_summary": output_dir / "shap_global_summary.json",
            "shap_feature_importance": output_dir / "shap_feature_importance.csv",
            "shap_summary_plot": output_dir / "shap_summary_plot.png",
            "shap_bar_plot": output_dir / "shap_bar_plot.png",
        }
        paths["shap_global_summary"].write_text("{}", encoding="utf-8")
        paths["shap_feature_importance"].write_text(
            "feature_name,mean_abs_shap\nfeature_1,0.1\n",
            encoding="utf-8",
        )
        paths["shap_summary_plot"].write_bytes(b"png")
        paths["shap_bar_plot"].write_bytes(b"png")
        return paths, {"row_count": 5}

    def fake_generate_shap_local_artifacts(
        **kwargs: object,
    ) -> tuple[list[dict[str, object]], Path]:
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "shap_local_explanations.jsonl"
        output_path.write_text('{"applicant_id": 1}\n', encoding="utf-8")
        return [{"applicant_id": 1}], output_path

    def fake_generate_lime_local_artifacts(
        **kwargs: object,
    ) -> tuple[list[dict[str, object]], Path, list[Path]]:
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "lime_explanations.jsonl"
        output_path.write_text('{"applicant_id": 1}\n', encoding="utf-8")
        case_path = output_dir / "lime_case_1.md"
        case_path.write_text("# LIME Local Explanation\n", encoding="utf-8")
        return [{"applicant_id": 1}], output_path, [case_path]

    monkeypatch.setattr(
        workflow,
        "generate_shap_global_artifacts",
        fake_generate_shap_global_artifacts,
    )
    monkeypatch.setattr(
        workflow,
        "generate_shap_local_artifacts",
        fake_generate_shap_local_artifacts,
    )
    monkeypatch.setattr(
        workflow,
        "generate_lime_local_artifacts",
        fake_generate_lime_local_artifacts,
    )

    result = workflow.run_explainability_workflow(
        settings,
        method_selection="all",
        sample_size=20,
        top_k=5,
        overwrite=True,
    )

    assert result.selected_examples_path.exists()
    assert result.shap_global_summary_path is not None
    assert result.shap_global_summary_path.exists()
    assert result.shap_local_explanations_path is not None
    assert result.shap_local_explanations_path.exists()
    assert result.lime_explanations_path is not None
    assert result.lime_explanations_path.exists()
