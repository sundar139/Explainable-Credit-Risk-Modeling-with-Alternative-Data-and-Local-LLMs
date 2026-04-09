"""Optuna tuning and Phase 5 modeling orchestration."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

from credit_risk_altdata.config import Settings
from credit_risk_altdata.logging import get_logger
from credit_risk_altdata.modeling.calibration import (
    CalibratedCandidateResult,
    build_calibration_comparison,
    evaluate_tuned_candidates,
)
from credit_risk_altdata.modeling.constants import (
    BEST_PARAMS_CATBOOST_FILE,
    BEST_PARAMS_LIGHTGBM_FILE,
    CALIBRATION_ALL,
    CALIBRATION_COMPARISON_FILE,
    CALIBRATION_ISOTONIC,
    CALIBRATION_NONE,
    CALIBRATION_SIGMOID,
    EVALUATION_SUMMARY_FILE,
    MODEL_ALL,
    MODEL_CATBOOST,
    MODEL_LIGHTGBM,
    THRESHOLD_ANALYSIS_FILE,
    TUNED_MODEL_COMPARISON_FILE,
    TUNED_MODELING_SUMMARY_FILE,
    TUNING_RESULTS_FILE,
    CalibrationMethod,
    CalibrationSelection,
    ModelFamily,
    ModelSelection,
)
from credit_risk_altdata.modeling.data_prep import (
    ModelingDataset,
    build_stratified_folds,
    prepare_modeling_dataset,
)
from credit_risk_altdata.modeling.evaluation_reporting import (
    build_threshold_grid,
    generate_evaluation_artifacts,
)
from credit_risk_altdata.modeling.final_selection import select_final_candidate
from credit_risk_altdata.modeling.model_factories import (
    build_estimator,
    predict_positive_probability,
)
from credit_risk_altdata.modeling.reporting import (
    ModelingArtifactPaths,
    resolve_modeling_artifact_paths,
    write_json,
    write_markdown,
)
from credit_risk_altdata.utils.filesystem import ensure_directories

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ModelTuningResult:
    """Optuna tuning output for one model family."""

    model_family: ModelFamily
    best_params: dict[str, Any]
    best_score: float
    trial_results: DataFrame


@dataclass(frozen=True, slots=True)
class TunedModelingResult:
    """Paths and final selection produced by the Phase 5 workflow."""

    tuning_results_path: Path
    calibration_comparison_path: Path
    tuned_model_comparison_path: Path
    threshold_analysis_path: Path
    evaluation_summary_path: Path
    tuned_modeling_summary_path: Path
    final_candidate_summary_path: Path
    final_model_name: str


def _resolve_model_families(model_selection: ModelSelection) -> list[ModelFamily]:
    if model_selection == MODEL_ALL:
        return [MODEL_LIGHTGBM, MODEL_CATBOOST]
    if model_selection in (MODEL_LIGHTGBM, MODEL_CATBOOST):
        return [model_selection]
    raise ValueError(f"Unsupported model selection: {model_selection}")


def _resolve_calibration_methods(
    calibration_selection: CalibrationSelection,
) -> list[CalibrationMethod]:
    if calibration_selection == CALIBRATION_NONE:
        return [CALIBRATION_NONE]
    if calibration_selection == CALIBRATION_SIGMOID:
        return [CALIBRATION_NONE, CALIBRATION_SIGMOID]
    if calibration_selection == CALIBRATION_ISOTONIC:
        return [CALIBRATION_NONE, CALIBRATION_ISOTONIC]
    if calibration_selection == CALIBRATION_ALL:
        return [CALIBRATION_NONE, CALIBRATION_SIGMOID, CALIBRATION_ISOTONIC]
    raise ValueError(f"Unsupported calibration selection: {calibration_selection}")


def _sample_lightgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 700),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }


def _sample_catboost_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 150, 700),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 20.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }


def _sample_params(model_family: ModelFamily, trial: optuna.Trial) -> dict[str, Any]:
    if model_family == MODEL_LIGHTGBM:
        return _sample_lightgbm_params(trial)
    if model_family == MODEL_CATBOOST:
        return _sample_catboost_params(trial)
    raise ValueError(f"Unsupported model family for tuning: {model_family}")


def _objective(
    trial: optuna.Trial,
    *,
    model_family: ModelFamily,
    dataset: ModelingDataset,
    settings: Settings,
) -> float:
    params = _sample_params(model_family, trial)
    y_values = dataset.y.to_numpy(dtype=int)
    folds = build_stratified_folds(
        dataset.y,
        n_splits=settings.modeling_folds,
        random_seed=settings.random_seed,
    )

    fold_scores: list[float] = []
    for fold_index, (train_indices, valid_indices) in enumerate(folds, start=1):
        x_fold_train = dataset.x_train.iloc[train_indices]
        x_fold_valid = dataset.x_train.iloc[valid_indices]
        y_fold_train = y_values[train_indices]
        y_fold_valid = y_values[valid_indices]

        estimator = build_estimator(
            model_family=model_family,
            params=params,
            random_seed=settings.random_seed + (trial.number * 100) + fold_index,
            y_train=y_fold_train.astype(np.int_),
        )
        estimator.fit(x_fold_train, y_fold_train)
        valid_pred = predict_positive_probability(estimator, x_fold_valid)

        if np.unique(y_fold_valid).size < 2:
            fold_scores.append(0.5)
            continue

        score = float(roc_auc_score(y_fold_valid, valid_pred))
        fold_scores.append(score)

    return float(np.mean(fold_scores)) if fold_scores else 0.5


def tune_model_hyperparameters(
    *,
    settings: Settings,
    dataset: ModelingDataset,
    model_family: ModelFamily,
    n_trials: int,
) -> ModelTuningResult:
    """Run Optuna tuning for a single model family."""
    sampler = optuna.samplers.TPESampler(seed=settings.random_seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"{model_family}_tuning",
    )
    study.optimize(
        lambda trial: _objective(
            trial,
            model_family=model_family,
            dataset=dataset,
            settings=settings,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
        catch=(Exception,),
    )

    trial_records: list[dict[str, Any]] = []
    for trial in study.trials:
        record: dict[str, Any] = {
            "model_family": model_family,
            "trial_number": trial.number,
            "state": trial.state.name,
            "value": float(trial.value) if trial.value is not None else np.nan,
            "params_json": json.dumps(trial.params, sort_keys=True),
        }
        for key, value in trial.params.items():
            record[f"param_{key}"] = value
        trial_records.append(record)

    trial_results = pd.DataFrame(trial_records)
    best_params = dict(study.best_trial.params)
    best_score = float(study.best_value)
    return ModelTuningResult(
        model_family=model_family,
        best_params=best_params,
        best_score=best_score,
        trial_results=trial_results,
    )


def _check_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing_paths = [path for path in paths if path.exists()]
    if existing_paths:
        raise FileExistsError(
            "Tuning artifacts already exist. Use overwrite=True to replace them. "
            f"Existing: {[str(path) for path in existing_paths]}"
        )


def _write_best_params_artifacts(
    *,
    artifact_paths: ModelingArtifactPaths,
    tuning_results: dict[ModelFamily, ModelTuningResult],
    n_trials: int,
) -> tuple[Path, Path]:
    lightgbm_payload: dict[str, Any]
    catboost_payload: dict[str, Any]

    if MODEL_LIGHTGBM in tuning_results:
        lightgbm_output = tuning_results[MODEL_LIGHTGBM]
        lightgbm_payload = {
            "model_family": MODEL_LIGHTGBM,
            "tuned": True,
            "n_trials": n_trials,
            "best_score": lightgbm_output.best_score,
            "best_params": lightgbm_output.best_params,
        }
    else:
        lightgbm_payload = {
            "model_family": MODEL_LIGHTGBM,
            "tuned": False,
            "reason": "Model family not requested",
        }

    if MODEL_CATBOOST in tuning_results:
        catboost_output = tuning_results[MODEL_CATBOOST]
        catboost_payload = {
            "model_family": MODEL_CATBOOST,
            "tuned": True,
            "n_trials": n_trials,
            "best_score": catboost_output.best_score,
            "best_params": catboost_output.best_params,
        }
    else:
        catboost_payload = {
            "model_family": MODEL_CATBOOST,
            "tuned": False,
            "reason": "Model family not requested",
        }

    lightgbm_path = write_json(
        artifact_paths.tuning_dir / BEST_PARAMS_LIGHTGBM_FILE,
        lightgbm_payload,
    )
    catboost_path = write_json(
        artifact_paths.tuning_dir / BEST_PARAMS_CATBOOST_FILE,
        catboost_payload,
    )
    return lightgbm_path, catboost_path


def _build_tuned_comparison_rows(
    candidates: list[CalibratedCandidateResult],
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        overall_row = candidate.fold_metrics[candidate.fold_metrics["fold"] == "overall"].iloc[0]
        rows.append(
            {
                "candidate_name": candidate.candidate_name,
                "model_family": candidate.model_family,
                "source": "tuned",
                "is_tuned": True,
                "is_calibrated": candidate.is_calibrated,
                "calibration_method": candidate.calibration_method,
                "roc_auc": float(overall_row["roc_auc"]),
                "pr_auc": float(overall_row["pr_auc"]),
                "precision": float(overall_row["precision"]),
                "recall": float(overall_row["recall"]),
                "f1": float(overall_row["f1"]),
                "accuracy": float(overall_row["accuracy"]),
                "threshold": float(overall_row["threshold"]),
                "artifact_path": str(candidate.model_artifact_path),
            }
        )
    tuned_rows = pd.DataFrame(rows)
    if tuned_rows.empty:
        return tuned_rows
    return tuned_rows.sort_values(by=["roc_auc", "pr_auc", "f1"], ascending=False).reset_index(
        drop=True
    )


def _baseline_artifact_path(settings: Settings, model_family: ModelFamily) -> Path:
    if model_family == MODEL_LIGHTGBM:
        return settings.modeling_models_dir / MODEL_LIGHTGBM / "final_model.joblib"
    if model_family == MODEL_CATBOOST:
        return settings.modeling_models_dir / MODEL_CATBOOST / "final_model.cbm"
    raise ValueError(f"Unsupported model family: {model_family}")


def _load_baseline_rows(settings: Settings, requested_models: list[ModelFamily]) -> DataFrame:
    baseline_path = settings.modeling_metrics_dir / "model_comparison.csv"
    if not baseline_path.exists():
        return pd.DataFrame()

    baseline = pd.read_csv(baseline_path)
    if "model_name" not in baseline.columns:
        return pd.DataFrame()

    baseline = baseline[baseline["model_name"].isin(requested_models)].copy()
    if baseline.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in baseline.iterrows():
        family_raw = str(row["model_name"])
        if family_raw not in (MODEL_LIGHTGBM, MODEL_CATBOOST):
            continue
        family = family_raw
        rows.append(
            {
                "candidate_name": f"{family}_baseline_none",
                "model_family": family,
                "source": "baseline",
                "is_tuned": False,
                "is_calibrated": False,
                "calibration_method": CALIBRATION_NONE,
                "roc_auc": float(row["roc_auc"]),
                "pr_auc": float(row["pr_auc"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "accuracy": float(row["accuracy"]),
                "threshold": float(row["threshold"]),
                "artifact_path": str(_baseline_artifact_path(settings, family)),
            }
        )
    return pd.DataFrame(rows)


def _write_candidate_prediction_artifacts(
    *,
    dataset: ModelingDataset,
    candidates: list[CalibratedCandidateResult],
    predictions_dir: Path,
) -> tuple[Path, Path]:
    oof_predictions = pd.DataFrame(
        {
            "SK_ID_CURR": dataset.train_ids.to_numpy(),
            "TARGET": dataset.y.to_numpy(),
        }
    )
    test_predictions = pd.DataFrame({"SK_ID_CURR": dataset.test_ids.to_numpy()})

    for candidate in candidates:
        oof_predictions[f"oof_pred_{candidate.candidate_name}"] = candidate.oof_predictions
        test_predictions[f"pred_{candidate.candidate_name}"] = candidate.test_predictions

    oof_path = predictions_dir / "tuned_oof_predictions.parquet"
    test_path = predictions_dir / "tuned_test_predictions.parquet"
    oof_predictions.to_parquet(oof_path, index=False)
    test_predictions.to_parquet(test_path, index=False)
    return oof_path, test_path


def _materialize_final_model_copy(
    *,
    settings: Settings,
    selected_artifact_path: Path,
) -> Path:
    output_path = settings.modeling_final_model_output_path
    if output_path.suffix.lower() != selected_artifact_path.suffix.lower():
        output_path = output_path.with_suffix(selected_artifact_path.suffix)

    ensure_directories([output_path.parent])
    if selected_artifact_path.resolve() != output_path.resolve():
        shutil.copy2(selected_artifact_path, output_path)
    return output_path


def run_tuned_modeling(
    settings: Settings,
    *,
    model_selection: ModelSelection = MODEL_ALL,
    n_trials: int | None = None,
    calibration_selection: CalibrationSelection | None = None,
    input_path_override: Path | None = None,
    overwrite: bool = False,
) -> TunedModelingResult:
    """Execute Phase 5 tuning, calibration, and rich evaluation workflow."""
    artifact_paths = resolve_modeling_artifact_paths(settings)
    dataset = prepare_modeling_dataset(settings, input_path_override=input_path_override)

    requested_models = _resolve_model_families(model_selection)
    calibration_choice = calibration_selection or settings.modeling_calibration_strategy
    calibration_methods = _resolve_calibration_methods(calibration_choice)
    resolved_trials = int(n_trials if n_trials is not None else settings.modeling_tuning_trials)
    final_candidate_summary_path = settings.modeling_final_candidate_summary_path

    required_paths = [
        artifact_paths.tuning_dir / TUNING_RESULTS_FILE,
        artifact_paths.tuning_dir / BEST_PARAMS_LIGHTGBM_FILE,
        artifact_paths.tuning_dir / BEST_PARAMS_CATBOOST_FILE,
        artifact_paths.calibration_dir / CALIBRATION_COMPARISON_FILE,
        artifact_paths.evaluation_dir / THRESHOLD_ANALYSIS_FILE,
        artifact_paths.evaluation_dir / EVALUATION_SUMMARY_FILE,
        artifact_paths.metrics_dir / TUNED_MODEL_COMPARISON_FILE,
        artifact_paths.reports_dir / TUNED_MODELING_SUMMARY_FILE,
        final_candidate_summary_path,
        settings.modeling_final_model_output_path,
    ]
    _check_overwrite(required_paths, overwrite)

    LOGGER.info(
        "Starting model tuning: models=%s trials=%d calibration=%s",
        requested_models,
        resolved_trials,
        calibration_choice,
    )

    tuning_outputs: dict[ModelFamily, ModelTuningResult] = {}
    trial_frames: list[DataFrame] = []
    for model_family in requested_models:
        output = tune_model_hyperparameters(
            settings=settings,
            dataset=dataset,
            model_family=model_family,
            n_trials=resolved_trials,
        )
        tuning_outputs[model_family] = output
        trial_frames.append(output.trial_results)
        LOGGER.info(
            "Tuning complete for %s: best_roc_auc=%.6f", model_family, output.best_score
        )

    tuning_results = pd.concat(trial_frames, axis=0, ignore_index=True)
    tuning_results_path = artifact_paths.tuning_dir / TUNING_RESULTS_FILE
    tuning_results.to_csv(tuning_results_path, index=False)

    best_lightgbm_path, best_catboost_path = _write_best_params_artifacts(
        artifact_paths=artifact_paths,
        tuning_results=tuning_outputs,
        n_trials=resolved_trials,
    )

    candidate_results: list[CalibratedCandidateResult] = []
    for model_family, tuning_output in tuning_outputs.items():
        family_candidates = evaluate_tuned_candidates(
            settings=settings,
            dataset=dataset,
            model_family=model_family,
            tuned_params=tuning_output.best_params,
            calibration_methods=calibration_methods,
            artifact_paths=artifact_paths,
        )
        candidate_results.extend(family_candidates)

        for candidate in family_candidates:
            fold_metrics_path = (
                artifact_paths.calibration_dir
                / f"{candidate.candidate_name}_fold_metrics.csv"
            )
            candidate.fold_metrics.to_csv(fold_metrics_path, index=False)

    if not candidate_results:
        raise RuntimeError("No tuned candidates were evaluated")

    calibration_comparison = build_calibration_comparison(candidate_results)
    calibration_comparison_path = artifact_paths.calibration_dir / CALIBRATION_COMPARISON_FILE
    calibration_comparison.to_csv(calibration_comparison_path, index=False)

    tuned_rows = _build_tuned_comparison_rows(candidate_results)
    baseline_rows = _load_baseline_rows(settings, requested_models)
    tuned_model_comparison = pd.concat([baseline_rows, tuned_rows], axis=0, ignore_index=True)
    tuned_model_comparison = tuned_model_comparison.sort_values(
        by=["roc_auc", "pr_auc", "f1"],
        ascending=False,
    ).reset_index(drop=True)
    tuned_model_comparison_path = artifact_paths.metrics_dir / TUNED_MODEL_COMPARISON_FILE
    tuned_model_comparison.to_csv(tuned_model_comparison_path, index=False)

    tuned_only = tuned_model_comparison[tuned_model_comparison["is_tuned"]].reset_index(drop=True)
    final_candidate_summary = select_final_candidate(
        tuned_only,
        primary_metric=settings.modeling_primary_metric,
        source_comparison_artifact=str(tuned_model_comparison_path),
    )

    selected_candidate_name = str(final_candidate_summary["final_candidate_name"])
    selected_candidate = next(
        candidate
        for candidate in candidate_results
        if candidate.candidate_name == selected_candidate_name
    )

    selected_artifact = Path(str(final_candidate_summary["selected_artifact_path"]))
    final_model_output_path = _materialize_final_model_copy(
        settings=settings,
        selected_artifact_path=selected_artifact,
    )
    final_candidate_summary["final_model_output_path"] = str(final_model_output_path)

    thresholds = build_threshold_grid(settings)
    evaluation_summary, evaluation_artifacts = generate_evaluation_artifacts(
        y_true=dataset.y.to_numpy(dtype=int),
        y_prob=selected_candidate.oof_predictions,
        thresholds=thresholds,
        evaluation_dir=artifact_paths.evaluation_dir,
    )

    threshold_analysis_path = Path(str(evaluation_artifacts["threshold_analysis"]))
    evaluation_summary_payload: dict[str, Any] = {
        "final_candidate": final_candidate_summary,
        "evaluation_metrics": evaluation_summary,
        "artifacts": {key: str(path) for key, path in evaluation_artifacts.items()},
    }
    evaluation_summary_path = write_json(
        artifact_paths.evaluation_dir / EVALUATION_SUMMARY_FILE,
        evaluation_summary_payload,
    )

    final_candidate_summary_path = write_json(
        final_candidate_summary_path,
        final_candidate_summary,
    )
    if not final_candidate_summary_path.exists():
        raise RuntimeError(
            "Final candidate summary artifact was not created: "
            f"{final_candidate_summary_path}"
        )

    tuned_oof_path, tuned_test_path = _write_candidate_prediction_artifacts(
        dataset=dataset,
        candidates=candidate_results,
        predictions_dir=artifact_paths.predictions_dir,
    )

    summary_lines = [
        "## Configuration",
        f"- Model selection: {model_selection}",
        f"- Trials: {resolved_trials}",
        f"- Calibration selection: {calibration_choice}",
        f"- Threshold: {settings.modeling_threshold}",
        "",
        "## Best Trial Scores",
    ]
    for model_family in requested_models:
        best_score = tuning_outputs[model_family].best_score
        summary_lines.append(f"- {model_family}: best_cv_roc_auc={best_score:.6f}")

    summary_lines.extend(
        [
            "",
            "## Final Candidate",
            f"- Candidate: {final_candidate_summary['final_candidate_name']}",
            f"- Model family: {final_candidate_summary['final_model_family']}",
            f"- Tuned: {final_candidate_summary['tuned']}",
            f"- Calibrated: {final_candidate_summary['calibrated']}",
            f"- Calibration method: {final_candidate_summary['calibration_method']}",
            "- "
            f"{settings.modeling_primary_metric}: "
            f"{final_candidate_summary['primary_metric_value']:.6f}",
            f"- Threshold: {final_candidate_summary['threshold']}",
            "",
            "## Key Artifacts",
            f"- Tuning results: {tuning_results_path}",
            f"- Best params (LightGBM): {best_lightgbm_path}",
            f"- Best params (CatBoost): {best_catboost_path}",
            f"- Calibration comparison: {calibration_comparison_path}",
            f"- Tuned comparison: {tuned_model_comparison_path}",
            f"- Threshold analysis: {threshold_analysis_path}",
            f"- Evaluation summary: {evaluation_summary_path}",
            f"- Final candidate summary: {final_candidate_summary_path}",
            f"- Final model copy: {final_model_output_path}",
            f"- Tuned OOF predictions: {tuned_oof_path}",
            f"- Tuned test predictions: {tuned_test_path}",
        ]
    )
    tuned_modeling_summary_path = write_markdown(
        artifact_paths.reports_dir / TUNED_MODELING_SUMMARY_FILE,
        title="Tuned Modeling Summary",
        lines=summary_lines,
    )

    LOGGER.info(
        "Tuned modeling completed. final_candidate=%s comparison=%s",
        final_candidate_summary["final_candidate_name"],
        tuned_model_comparison_path,
    )

    return TunedModelingResult(
        tuning_results_path=tuning_results_path,
        calibration_comparison_path=calibration_comparison_path,
        tuned_model_comparison_path=tuned_model_comparison_path,
        threshold_analysis_path=threshold_analysis_path,
        evaluation_summary_path=evaluation_summary_path,
        tuned_modeling_summary_path=tuned_modeling_summary_path,
        final_candidate_summary_path=final_candidate_summary_path,
        final_model_name=str(final_candidate_summary["final_candidate_name"]),
    )
