"""Calibration-aware candidate evaluation for tuned models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier  # type: ignore[import-untyped]
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]
from sklearn.calibration import CalibratedClassifierCV  # type: ignore[import-untyped]
from sklearn.metrics import brier_score_loss, log_loss  # type: ignore[import-untyped]

from credit_risk_altdata.config import Settings
from credit_risk_altdata.modeling.constants import (
    CALIBRATION_ISOTONIC,
    CALIBRATION_NONE,
    CALIBRATION_SIGMOID,
    MODEL_CATBOOST,
    MODEL_LIGHTGBM,
    CalibrationMethod,
    ModelFamily,
)
from credit_risk_altdata.modeling.data_prep import ModelingDataset, build_stratified_folds
from credit_risk_altdata.modeling.metrics import compute_classification_metrics
from credit_risk_altdata.modeling.model_factories import (
    build_estimator,
    predict_positive_probability,
)
from credit_risk_altdata.modeling.reporting import ModelingArtifactPaths


@dataclass(frozen=True, slots=True)
class CalibratedCandidateResult:
    """Evaluation outputs for one tuned + calibration candidate."""

    candidate_name: str
    model_family: ModelFamily
    calibration_method: CalibrationMethod
    is_calibrated: bool
    is_tuned: bool
    fold_metrics: DataFrame
    oof_predictions: NDArray[np.float64]
    test_predictions: NDArray[np.float64]
    model_artifact_path: Path


def _resolve_calibration_cv_splits(y_train: NDArray[np.int_]) -> int:
    class_counts = np.bincount(y_train)
    min_count = int(class_counts.min()) if class_counts.size > 0 else 2
    if min_count >= 3:
        return 3
    return 2


def _fit_predict_candidate(
    *,
    model_family: ModelFamily,
    calibration_method: CalibrationMethod,
    params: dict[str, Any],
    x_train_fold: DataFrame,
    y_train_fold: NDArray[np.int_],
    x_valid_fold: DataFrame,
    x_test: DataFrame,
    random_seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    estimator = build_estimator(
        model_family=model_family,
        params=params,
        random_seed=random_seed,
        y_train=y_train_fold,
    )

    if calibration_method == CALIBRATION_NONE:
        estimator.fit(x_train_fold, y_train_fold)
        valid_pred = predict_positive_probability(estimator, x_valid_fold)
        test_pred = predict_positive_probability(estimator, x_test)
        return valid_pred, test_pred

    calibration_cv = _resolve_calibration_cv_splits(y_train_fold)
    calibrated_model = CalibratedClassifierCV(
        estimator=cast(BaseEstimator, estimator),
        method=calibration_method,
        cv=calibration_cv,
    )
    calibrated_model.fit(x_train_fold, y_train_fold)
    valid_pred = np.asarray(calibrated_model.predict_proba(x_valid_fold)[:, 1], dtype=np.float64)
    test_pred = np.asarray(calibrated_model.predict_proba(x_test)[:, 1], dtype=np.float64)
    return valid_pred, test_pred


def evaluate_tuned_candidates(
    *,
    settings: Settings,
    dataset: ModelingDataset,
    model_family: ModelFamily,
    tuned_params: dict[str, Any],
    calibration_methods: list[CalibrationMethod],
    artifact_paths: ModelingArtifactPaths,
) -> list[CalibratedCandidateResult]:
    """Evaluate tuned model variants for each calibration option."""
    x_train = dataset.x_train
    x_test = dataset.x_test
    y_values = dataset.y.to_numpy(dtype=int)

    folds = build_stratified_folds(
        dataset.y,
        n_splits=settings.modeling_folds,
        random_seed=settings.random_seed,
    )

    results: list[CalibratedCandidateResult] = []
    for calibration_method in calibration_methods:
        candidate_name = f"{model_family}_tuned_{calibration_method}"
        oof_predictions = np.zeros(shape=(len(y_values),), dtype=np.float64)
        test_predictions_per_fold: list[NDArray[np.float64]] = []
        fold_records: list[dict[str, float | str]] = []

        for fold_index, (train_indices, valid_indices) in enumerate(folds, start=1):
            x_fold_train = x_train.iloc[train_indices]
            x_fold_valid = x_train.iloc[valid_indices]
            y_fold_train = y_values[train_indices]
            y_fold_valid = y_values[valid_indices]

            valid_pred, test_pred = _fit_predict_candidate(
                model_family=model_family,
                calibration_method=calibration_method,
                params=tuned_params,
                x_train_fold=x_fold_train,
                y_train_fold=y_fold_train,
                x_valid_fold=x_fold_valid,
                x_test=x_test,
                random_seed=settings.random_seed + fold_index,
            )

            oof_predictions[np.asarray(valid_indices, dtype=int)] = valid_pred
            test_predictions_per_fold.append(test_pred)

            fold_metrics = dict(
                compute_classification_metrics(
                    y_true=y_fold_valid.astype(int),
                    y_prob=valid_pred.astype(np.float64),
                    threshold=settings.modeling_threshold,
                )
            )
            fold_metrics["brier_score"] = float(brier_score_loss(y_fold_valid, valid_pred))
            fold_metrics["log_loss"] = float(
                log_loss(y_fold_valid, np.clip(valid_pred, 1e-6, 1.0 - 1e-6), labels=[0, 1])
            )
            fold_records.append(
                {
                    **fold_metrics,
                    "fold": str(fold_index),
                    "model_family": model_family,
                    "calibration_method": calibration_method,
                    "candidate_name": candidate_name,
                }
            )

        mean_test_predictions = np.mean(np.vstack(test_predictions_per_fold), axis=0)
        overall_metrics = dict(
            compute_classification_metrics(
                y_true=y_values.astype(int),
                y_prob=oof_predictions,
                threshold=settings.modeling_threshold,
            )
        )
        overall_metrics["brier_score"] = float(brier_score_loss(y_values, oof_predictions))
        overall_metrics["log_loss"] = float(
            log_loss(y_values, np.clip(oof_predictions, 1e-6, 1.0 - 1e-6), labels=[0, 1])
        )
        fold_records.append(
            {
                **overall_metrics,
                "fold": "overall",
                "model_family": model_family,
                "calibration_method": calibration_method,
                "candidate_name": candidate_name,
            }
        )
        fold_metrics_frame = pd.DataFrame(fold_records)

        final_estimator = build_estimator(
            model_family=model_family,
            params=tuned_params,
            random_seed=settings.random_seed + 10_000,
            y_train=y_values.astype(np.int_),
        )
        if calibration_method == CALIBRATION_NONE:
            final_estimator.fit(x_train, y_values)
            if model_family == MODEL_LIGHTGBM:
                artifact_path = artifact_paths.tuned_models_dir / f"{candidate_name}.joblib"
                joblib.dump(final_estimator, artifact_path)
            elif model_family == MODEL_CATBOOST:
                artifact_path = artifact_paths.tuned_models_dir / f"{candidate_name}.cbm"
                catboost_estimator = cast(CatBoostClassifier, final_estimator)
                catboost_estimator.save_model(str(artifact_path))
            else:
                raise ValueError(f"Unsupported model family: {model_family}")
        elif calibration_method in (CALIBRATION_SIGMOID, CALIBRATION_ISOTONIC):
            calibration_cv = _resolve_calibration_cv_splits(y_values.astype(np.int_))
            calibrated_final = CalibratedClassifierCV(
                estimator=cast(BaseEstimator, final_estimator),
                method=calibration_method,
                cv=calibration_cv,
            )
            calibrated_final.fit(x_train, y_values)
            artifact_path = artifact_paths.tuned_models_dir / f"{candidate_name}.joblib"
            joblib.dump(calibrated_final, artifact_path)
        else:
            raise ValueError(f"Unsupported calibration method: {calibration_method}")

        results.append(
            CalibratedCandidateResult(
                candidate_name=candidate_name,
                model_family=model_family,
                calibration_method=calibration_method,
                is_calibrated=calibration_method != CALIBRATION_NONE,
                is_tuned=True,
                fold_metrics=fold_metrics_frame,
                oof_predictions=oof_predictions,
                test_predictions=np.asarray(mean_test_predictions, dtype=np.float64),
                model_artifact_path=artifact_path,
            )
        )

    return results


def build_calibration_comparison(candidates: list[CalibratedCandidateResult]) -> DataFrame:
    """Build calibration comparison table from candidate overall rows."""
    records: list[dict[str, float | str | bool]] = []
    for candidate in candidates:
        overall_row = candidate.fold_metrics[candidate.fold_metrics["fold"] == "overall"].iloc[0]
        records.append(
            {
                "candidate_name": candidate.candidate_name,
                "model_family": candidate.model_family,
                "calibration_method": candidate.calibration_method,
                "is_calibrated": candidate.is_calibrated,
                "roc_auc": float(overall_row["roc_auc"]),
                "pr_auc": float(overall_row["pr_auc"]),
                "precision": float(overall_row["precision"]),
                "recall": float(overall_row["recall"]),
                "f1": float(overall_row["f1"]),
                "accuracy": float(overall_row["accuracy"]),
                "brier_score": float(overall_row["brier_score"]),
                "log_loss": float(overall_row["log_loss"]),
                "threshold": float(overall_row["threshold"]),
                "artifact_path": str(candidate.model_artifact_path),
            }
        )

    comparison = pd.DataFrame(records)
    if comparison.empty:
        return comparison
    return comparison.sort_values(by=["roc_auc", "pr_auc", "f1"], ascending=False).reset_index(
        drop=True
    )
