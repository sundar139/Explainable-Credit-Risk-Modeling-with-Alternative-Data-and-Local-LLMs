"""Baseline model training orchestration for Home Credit features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from credit_risk_altdata.config import Settings
from credit_risk_altdata.modeling.catboost_model import (
    catboost_feature_importance,
    fit_catboost_classifier,
    predict_catboost_probabilities,
)
from credit_risk_altdata.modeling.constants import (
    BEST_MODEL_SUMMARY_FILE,
    FOLD_METRICS_FILE,
    IDENTIFIER_COLUMN,
    LABEL_COLUMN,
    MODEL_ALL,
    MODEL_CATBOOST,
    MODEL_COMPARISON_FILE,
    MODEL_LIGHTGBM,
    MODELING_SUMMARY_FILE,
    OOF_PREDICTIONS_FILE,
    PRIMARY_METRIC,
    SUPPORTED_MODEL_FAMILIES,
    TEST_PREDICTIONS_FILE,
    ModelFamily,
    ModelSelection,
)
from credit_risk_altdata.modeling.data_prep import (
    ModelingDataset,
    build_stratified_folds,
    prepare_modeling_dataset,
)
from credit_risk_altdata.modeling.lightgbm_model import (
    fit_lightgbm_classifier,
    lightgbm_feature_importance,
    predict_lightgbm_probabilities,
)
from credit_risk_altdata.modeling.metrics import (
    compute_classification_metrics,
    summarize_fold_metrics,
)
from credit_risk_altdata.modeling.registry import select_best_model
from credit_risk_altdata.modeling.reporting import (
    ModelingArtifactPaths,
    resolve_modeling_artifact_paths,
    write_json,
    write_markdown,
)
from credit_risk_altdata.utils.filesystem import ensure_directories


@dataclass(frozen=True, slots=True)
class ModelRunResult:
    """Cross-validated training outputs for one model family."""

    model_name: ModelFamily
    fold_metrics: DataFrame
    oof_predictions: NDArray[np.float64]
    test_predictions: NDArray[np.float64]
    feature_importance: DataFrame
    final_model_path: Path


@dataclass(frozen=True, slots=True)
class BaselineTrainingResult:
    """Output locations and high-level summary for baseline training run."""

    fold_metrics_path: Path
    model_comparison_path: Path
    oof_predictions_path: Path
    test_predictions_path: Path
    best_model_summary_path: Path
    summary_report_path: Path
    best_model_name: str


def _resolve_model_families(model_selection: ModelSelection) -> list[ModelFamily]:
    if model_selection == MODEL_ALL:
        return list(SUPPORTED_MODEL_FAMILIES)
    if model_selection in (MODEL_LIGHTGBM, MODEL_CATBOOST):
        return [model_selection]
    raise ValueError(f"Unsupported model selection: {model_selection}")


def _aggregate_feature_importance(fold_importance: list[DataFrame]) -> DataFrame:
    if not fold_importance:
        return DataFrame(columns=["feature_name", "importance_mean", "importance_std"])

    merged = pd.concat(fold_importance, axis=0, ignore_index=True)
    summary = merged.groupby("feature_name", as_index=False)["importance"].agg(["mean", "std"])
    summary = summary.reset_index().rename(
        columns={"mean": "importance_mean", "std": "importance_std"}
    )
    return summary.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _train_single_model(
    *,
    model_name: ModelFamily,
    dataset: ModelingDataset,
    settings: Settings,
    artifact_paths: ModelingArtifactPaths,
) -> ModelRunResult:
    x_train = dataset.x_train
    x_test = dataset.x_test
    y_series = dataset.y
    feature_columns = dataset.feature_columns

    y_values = y_series.to_numpy(dtype=int)
    folds = build_stratified_folds(
        y_series,
        n_splits=settings.modeling_folds,
        random_seed=settings.random_seed,
    )

    model_dir = artifact_paths.models_dir / model_name
    ensure_directories([model_dir])

    oof_predictions = np.zeros(shape=(len(y_values),), dtype=float)
    test_predictions_per_fold: list[NDArray[np.float64]] = []
    fold_metrics_records: list[dict[str, float | str]] = []
    fold_importances: list[DataFrame] = []

    for fold_index, (train_indices, valid_indices) in enumerate(folds, start=1):
        x_fold_train = x_train.iloc[train_indices]
        x_fold_valid = x_train.iloc[valid_indices]
        y_fold_train = y_values[train_indices]
        y_fold_valid = y_values[valid_indices]

        seed = settings.random_seed + fold_index
        if model_name == MODEL_LIGHTGBM:
            model = fit_lightgbm_classifier(
                x_train=x_fold_train,
                y_train=y_fold_train,
                random_seed=seed,
            )
            valid_pred = predict_lightgbm_probabilities(model, x_fold_valid)
            test_pred = predict_lightgbm_probabilities(model, x_test)
            importance = lightgbm_feature_importance(model=model, feature_columns=feature_columns)
            fold_model_path = model_dir / f"fold_{fold_index}.joblib"
            joblib.dump(model, fold_model_path)
        elif model_name == MODEL_CATBOOST:
            model = fit_catboost_classifier(
                x_train=x_fold_train,
                y_train=y_fold_train,
                random_seed=seed,
            )
            valid_pred = predict_catboost_probabilities(model, x_fold_valid)
            test_pred = predict_catboost_probabilities(model, x_test)
            importance = catboost_feature_importance(model=model, feature_columns=feature_columns)
            fold_model_path = model_dir / f"fold_{fold_index}.cbm"
            model.save_model(str(fold_model_path))
        else:
            raise ValueError(f"Unsupported model family: {model_name}")

        oof_predictions[np.asarray(valid_indices, dtype=int)] = valid_pred
        test_predictions_per_fold.append(np.asarray(test_pred, dtype=np.float64))
        fold_importances.append(importance.assign(fold=fold_index))

        fold_metrics = dict(
            compute_classification_metrics(
                y_true=y_fold_valid.astype(int),
                y_prob=valid_pred.astype(float),
                threshold=settings.modeling_threshold,
            )
        )
        fold_metrics_records.append(
            {
                **fold_metrics,
                "model_name": model_name,
                "fold": str(fold_index),
            }
        )

    if not test_predictions_per_fold:
        raise RuntimeError(f"No fold predictions produced for {model_name}")

    mean_test_predictions = np.mean(np.vstack(test_predictions_per_fold), axis=0)
    overall_metrics = dict(
        compute_classification_metrics(
            y_true=y_values.astype(int),
            y_prob=oof_predictions.astype(float),
            threshold=settings.modeling_threshold,
        )
    )
    fold_metrics_records.append(
        {
            **overall_metrics,
            "model_name": model_name,
            "fold": "overall",
        }
    )
    fold_metrics_frame = pd.DataFrame(fold_metrics_records)

    feature_importance_summary = _aggregate_feature_importance(fold_importances)

    final_seed = settings.random_seed + 10_000
    if model_name == MODEL_LIGHTGBM:
        final_model = fit_lightgbm_classifier(
            x_train=x_train,
            y_train=y_values,
            random_seed=final_seed,
        )
        final_model_path = model_dir / "final_model.joblib"
        joblib.dump(final_model, final_model_path)
    elif model_name == MODEL_CATBOOST:
        final_model = fit_catboost_classifier(
            x_train=x_train,
            y_train=y_values,
            random_seed=final_seed,
        )
        final_model_path = model_dir / "final_model.cbm"
        final_model.save_model(str(final_model_path))
    else:
        raise ValueError(f"Unsupported model family: {model_name}")

    return ModelRunResult(
        model_name=model_name,
        fold_metrics=fold_metrics_frame,
        oof_predictions=np.asarray(oof_predictions, dtype=np.float64),
        test_predictions=np.asarray(mean_test_predictions, dtype=np.float64),
        feature_importance=feature_importance_summary,
        final_model_path=final_model_path,
    )


def _check_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing_paths = [path for path in paths if path.exists()]
    if existing_paths:
        raise FileExistsError(
            "Modeling artifacts already exist. Use overwrite=True to replace them. "
            f"Existing: {[str(path) for path in existing_paths]}"
        )


def run_baseline_training(
    settings: Settings,
    *,
    model_selection: ModelSelection = MODEL_ALL,
    input_path_override: Path | None = None,
    overwrite: bool = False,
) -> BaselineTrainingResult:
    """Train LightGBM/CatBoost baselines with stratified CV and artifact reporting."""
    artifact_paths = resolve_modeling_artifact_paths(settings)
    dataset = prepare_modeling_dataset(settings, input_path_override=input_path_override)

    target_paths = [
        artifact_paths.metrics_dir / FOLD_METRICS_FILE,
        artifact_paths.metrics_dir / MODEL_COMPARISON_FILE,
        artifact_paths.predictions_dir / OOF_PREDICTIONS_FILE,
        artifact_paths.predictions_dir / TEST_PREDICTIONS_FILE,
        artifact_paths.reports_dir / BEST_MODEL_SUMMARY_FILE,
        artifact_paths.reports_dir / MODELING_SUMMARY_FILE,
    ]
    _check_overwrite(target_paths, overwrite)

    requested_models = _resolve_model_families(model_selection)
    run_results: list[ModelRunResult] = []
    for model_name in requested_models:
        run_results.append(
            _train_single_model(
                model_name=model_name,
                dataset=dataset,
                settings=settings,
                artifact_paths=artifact_paths,
            )
        )

    fold_metrics_frame = pd.concat(
        [result.fold_metrics for result in run_results],
        axis=0,
        ignore_index=True,
    )
    fold_metrics_path = artifact_paths.metrics_dir / FOLD_METRICS_FILE
    fold_metrics_frame.to_csv(fold_metrics_path, index=False)

    comparison_records: list[dict[str, float | str]] = []
    for result in run_results:
        per_model_metrics_path = (
            artifact_paths.metrics_dir / f"{result.model_name}_fold_metrics.csv"
        )
        result.fold_metrics.to_csv(per_model_metrics_path, index=False)

        overall_row = result.fold_metrics[result.fold_metrics["fold"] == "overall"].iloc[0]
        comparison_records.append(
            {
                "model_name": result.model_name,
                "roc_auc": float(overall_row["roc_auc"]),
                "pr_auc": float(overall_row["pr_auc"]),
                "precision": float(overall_row["precision"]),
                "recall": float(overall_row["recall"]),
                "f1": float(overall_row["f1"]),
                "accuracy": float(overall_row["accuracy"]),
                "threshold": float(overall_row["threshold"]),
            }
        )

        feature_importance_path = (
            artifact_paths.feature_importance_dir / f"{result.model_name}_feature_importance.csv"
        )
        result.feature_importance.to_csv(feature_importance_path, index=False)

    model_comparison = pd.DataFrame(comparison_records).sort_values(
        by=[PRIMARY_METRIC, "pr_auc", "f1"],
        ascending=False,
    )
    model_comparison_path = artifact_paths.metrics_dir / MODEL_COMPARISON_FILE
    model_comparison.to_csv(model_comparison_path, index=False)

    metrics_summary_path = artifact_paths.metrics_dir / "model_metric_summary.csv"
    summarize_fold_metrics(fold_metrics_frame).to_csv(metrics_summary_path, index=False)

    oof_predictions = pd.DataFrame(
        {
            IDENTIFIER_COLUMN: dataset.train_ids.to_numpy(),
            LABEL_COLUMN: dataset.y.to_numpy(),
        }
    )
    for result in run_results:
        oof_predictions[f"oof_pred_{result.model_name}"] = result.oof_predictions
    oof_predictions_path = artifact_paths.predictions_dir / OOF_PREDICTIONS_FILE
    oof_predictions.to_parquet(oof_predictions_path, index=False)

    test_predictions = pd.DataFrame({IDENTIFIER_COLUMN: dataset.test_ids.to_numpy()})
    for result in run_results:
        test_predictions[f"pred_{result.model_name}"] = result.test_predictions
    test_predictions_path = artifact_paths.predictions_dir / TEST_PREDICTIONS_FILE
    test_predictions.to_parquet(test_predictions_path, index=False)

    best_model_summary = select_best_model(
        model_comparison,
        primary_metric=settings.modeling_primary_metric,
        threshold=settings.modeling_threshold,
        folds=settings.modeling_folds,
        random_seed=settings.random_seed,
    )
    best_model_summary["model_selection"] = model_selection
    best_model_summary["input_train_path"] = str(dataset.train_input_path)
    best_model_summary["input_test_path"] = str(dataset.test_input_path)
    best_model_summary["trained_model_paths"] = {
        result.model_name: str(result.final_model_path) for result in run_results
    }
    best_model_summary_path = write_json(
        artifact_paths.reports_dir / BEST_MODEL_SUMMARY_FILE,
        best_model_summary,
    )

    summary_lines = [
        "## Modeling Configuration",
        f"- Model selection: {model_selection}",
        f"- CV folds: {settings.modeling_folds}",
        f"- Threshold: {settings.modeling_threshold}",
        f"- Primary metric: {settings.modeling_primary_metric}",
        "",
        "## Model Comparison",
    ]
    for _, row in model_comparison.iterrows():
        summary_lines.append(
            f"- {row['model_name']}: roc_auc={row['roc_auc']:.6f}, "
            f"pr_auc={row['pr_auc']:.6f}, f1={row['f1']:.6f}"
        )

    primary_metric_value = cast(float, best_model_summary["primary_metric_value"])
    summary_lines.extend(
        [
            "",
            "## Best Baseline",
            f"- Best model: {best_model_summary['best_model_name']}",
            f"- {settings.modeling_primary_metric}: {primary_metric_value:.6f}",
            f"- Threshold: {settings.modeling_threshold}",
            "",
            "## Artifact Paths",
            f"- Fold metrics: {fold_metrics_path}",
            f"- Model comparison: {model_comparison_path}",
            f"- OOF predictions: {oof_predictions_path}",
            f"- Test predictions: {test_predictions_path}",
            f"- Best model summary: {best_model_summary_path}",
        ]
    )
    summary_report_path = write_markdown(
        artifact_paths.reports_dir / MODELING_SUMMARY_FILE,
        title="Baseline Modeling Summary",
        lines=summary_lines,
    )

    return BaselineTrainingResult(
        fold_metrics_path=fold_metrics_path,
        model_comparison_path=model_comparison_path,
        oof_predictions_path=oof_predictions_path,
        test_predictions_path=test_predictions_path,
        best_model_summary_path=best_model_summary_path,
        summary_report_path=summary_report_path,
        best_model_name=str(best_model_summary["best_model_name"]),
    )
