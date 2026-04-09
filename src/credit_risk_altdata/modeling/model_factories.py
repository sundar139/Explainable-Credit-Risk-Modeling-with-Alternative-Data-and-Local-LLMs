"""Shared estimator factories for tuning and calibration workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
from catboost import CatBoostClassifier  # type: ignore[import-untyped]
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from pandas import DataFrame

from credit_risk_altdata.modeling.constants import MODEL_CATBOOST, MODEL_LIGHTGBM, ModelFamily


def _scale_pos_weight(y_train: NDArray[np.int_]) -> float:
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0 or negatives == 0:
        return 1.0
    return float(negatives / positives)


def _class_weights(y_train: NDArray[np.int_]) -> list[float]:
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0 or negatives == 0:
        return [1.0, 1.0]
    return [1.0, float(negatives / positives)]


def build_lightgbm_estimator(
    params: dict[str, Any],
    *,
    random_seed: int,
    y_train: NDArray[np.int_],
) -> LGBMClassifier:
    """Construct a LightGBM estimator with class-imbalance handling."""
    model_params: dict[str, Any] = {
        "objective": "binary",
        "random_state": random_seed,
        "n_jobs": -1,
        "verbosity": -1,
        "scale_pos_weight": _scale_pos_weight(y_train),
    }
    model_params.update(params)
    return LGBMClassifier(**model_params)


def build_catboost_estimator(
    params: dict[str, Any],
    *,
    random_seed: int,
    y_train: NDArray[np.int_],
) -> CatBoostClassifier:
    """Construct a CatBoost estimator with class-imbalance handling."""
    model_params: dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "allow_writing_files": False,
        "verbose": False,
        "random_seed": random_seed,
        "class_weights": _class_weights(y_train),
    }
    model_params.update(params)
    return CatBoostClassifier(**model_params)


def build_estimator(
    *,
    model_family: ModelFamily,
    params: dict[str, Any],
    random_seed: int,
    y_train: NDArray[np.int_],
) -> LGBMClassifier | CatBoostClassifier:
    """Construct model-family-specific estimator."""
    if model_family == MODEL_LIGHTGBM:
        return build_lightgbm_estimator(params, random_seed=random_seed, y_train=y_train)
    if model_family == MODEL_CATBOOST:
        return build_catboost_estimator(params, random_seed=random_seed, y_train=y_train)
    raise ValueError(f"Unsupported model family: {model_family}")


def predict_positive_probability(
    model: LGBMClassifier | CatBoostClassifier,
    x_frame: DataFrame,
) -> NDArray[np.float64]:
    """Predict positive-class probability as float64 numpy array."""
    probability_matrix = np.asarray(model.predict_proba(x_frame), dtype=np.float64)
    if probability_matrix.ndim != 2 or probability_matrix.shape[1] < 2:
        raise ValueError(
            "Estimator predict_proba output must include "
            "two class probability columns"
        )
    return np.asarray(probability_matrix[:, 1], dtype=np.float64)
