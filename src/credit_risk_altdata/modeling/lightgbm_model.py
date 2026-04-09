"""LightGBM baseline model utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from pandas import DataFrame


def _scale_pos_weight(y_train: NDArray[np.int_]) -> float:
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0 or negatives == 0:
        return 1.0
    return float(negatives / positives)


def fit_lightgbm_classifier(
    *,
    x_train: DataFrame,
    y_train: NDArray[np.int_],
    random_seed: int,
) -> LGBMClassifier:
    """Fit a LightGBM classifier with practical baseline settings."""
    model = LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=-1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=random_seed,
        n_jobs=-1,
        scale_pos_weight=_scale_pos_weight(y_train),
    )
    model.fit(x_train, y_train)
    return model


def predict_lightgbm_probabilities(
    model: LGBMClassifier,
    x_frame: DataFrame,
) -> NDArray[np.float64]:
    """Predict positive-class probabilities using a fitted LightGBM model."""
    probabilities = model.predict_proba(x_frame)[:, 1]
    return np.asarray(probabilities, dtype=np.float64)


def lightgbm_feature_importance(
    *,
    model: LGBMClassifier,
    feature_columns: Sequence[str],
) -> DataFrame:
    """Return feature importance dataframe for a fitted LightGBM model."""
    return pd.DataFrame(
        {
            "feature_name": list(feature_columns),
            "importance": model.feature_importances_.astype(float),
        }
    )
