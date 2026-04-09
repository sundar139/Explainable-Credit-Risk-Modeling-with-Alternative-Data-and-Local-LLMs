"""CatBoost baseline model utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier  # type: ignore[import-untyped]
from numpy.typing import NDArray
from pandas import DataFrame


def _class_weights(y_train: NDArray[np.int_]) -> list[float]:
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0 or negatives == 0:
        return [1.0, 1.0]
    return [1.0, float(negatives / positives)]


def fit_catboost_classifier(
    *,
    x_train: DataFrame,
    y_train: NDArray[np.int_],
    random_seed: int,
) -> CatBoostClassifier:
    """Fit a CatBoost classifier with stable baseline settings."""
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=6,
        learning_rate=0.05,
        iterations=400,
        random_seed=random_seed,
        class_weights=_class_weights(y_train),
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x_train, y_train, verbose=False)
    return model


def predict_catboost_probabilities(
    model: CatBoostClassifier,
    x_frame: DataFrame,
) -> NDArray[np.float64]:
    """Predict positive-class probabilities using a fitted CatBoost model."""
    probabilities = model.predict_proba(x_frame)[:, 1]
    return np.asarray(probabilities, dtype=np.float64)


def catboost_feature_importance(
    *,
    model: CatBoostClassifier,
    feature_columns: Sequence[str],
) -> DataFrame:
    """Return feature importance dataframe for a fitted CatBoost model."""
    importances = model.get_feature_importance()
    return pd.DataFrame(
        {
            "feature_name": list(feature_columns),
            "importance": importances.astype(float),
        }
    )
