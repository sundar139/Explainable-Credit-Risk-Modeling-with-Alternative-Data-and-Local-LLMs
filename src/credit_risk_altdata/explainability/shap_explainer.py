"""SHAP global and local explainability generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import matplotlib
import numpy as np
import pandas as pd
import shap  # type: ignore[import-untyped]
from catboost import CatBoostClassifier  # type: ignore[import-untyped]
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.calibration import CalibratedClassifierCV  # type: ignore[import-untyped]

from credit_risk_altdata.explainability.constants import (
    SHAP_BAR_PLOT_FILE,
    SHAP_FEATURE_IMPORTANCE_FILE,
    SHAP_GLOBAL_SUMMARY_FILE,
    SHAP_LOCAL_EXPLANATIONS_FILE,
    SHAP_SUMMARY_PLOT_FILE,
)
from credit_risk_altdata.explainability.payloads import build_local_explanation_payload
from credit_risk_altdata.explainability.reporting import write_json, write_jsonl

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _extract_positive_class_shap_values(shap_values: Any) -> NDArray[np.float64]:
    if isinstance(shap_values, list):
        selected = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        return np.asarray(selected, dtype=np.float64)

    values_array = np.asarray(shap_values, dtype=np.float64)
    if values_array.ndim == 3:
        if values_array.shape[2] < 2:
            raise ValueError("SHAP output has unexpected class dimension")
        return np.asarray(values_array[:, :, 1], dtype=np.float64)
    if values_array.ndim == 2:
        return np.asarray(values_array, dtype=np.float64)

    raise ValueError(f"Unsupported SHAP output shape: {values_array.shape}")


def _resolve_tree_shap_model(model: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(model, LGBMClassifier):
        return model, {"model_type": "lightgbm", "calibrated": False}
    if isinstance(model, CatBoostClassifier):
        return model, {"model_type": "catboost", "calibrated": False}

    if isinstance(model, CalibratedClassifierCV):
        if not model.calibrated_classifiers_:
            raise ValueError("Calibrated model has no fitted calibrated classifiers")
        first_calibrator = model.calibrated_classifiers_[0]
        base_estimator = getattr(first_calibrator, "estimator", None)
        if isinstance(base_estimator, (LGBMClassifier, CatBoostClassifier)):
            return base_estimator, {
                "model_type": base_estimator.__class__.__name__.lower(),
                "calibrated": True,
                "shap_base_estimator_used": True,
            }
        raise ValueError(
            "Calibrated model is not supported for SHAP unless its base estimator is "
            "LightGBM or CatBoost"
        )

    raise ValueError(
        f"Unsupported model type for SHAP explainability: {type(model).__name__}"
    )


def predict_positive_probability(model: Any, x_frame: DataFrame) -> NDArray[np.float64]:
    probability_matrix = np.asarray(model.predict_proba(x_frame), dtype=np.float64)
    if probability_matrix.ndim != 2 or probability_matrix.shape[1] < 2:
        raise ValueError("Model predict_proba output is not a two-class probability matrix")
    return np.asarray(probability_matrix[:, 1], dtype=np.float64)


def _sample_background(
    x_frame: DataFrame,
    *,
    sample_size: int,
    random_seed: int,
) -> DataFrame:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if len(x_frame) <= sample_size:
        return x_frame.copy()
    return x_frame.sample(n=sample_size, random_state=random_seed).sort_index()


def _build_feature_contribution_map(
    feature_names: list[str],
    contributions: NDArray[np.float64],
) -> dict[str, float]:
    return {
        feature_name: float(contributions[index])
        for index, feature_name in enumerate(feature_names)
    }


def generate_shap_global_artifacts(
    *,
    model: Any,
    x_frame: DataFrame,
    sample_size: int,
    top_k: int,
    random_seed: int,
    output_dir: Path,
    model_metadata: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Generate SHAP global artifacts and return paths and summary payload."""
    shap_model, shap_model_info = _resolve_tree_shap_model(model)
    background = _sample_background(
        x_frame,
        sample_size=sample_size,
        random_seed=random_seed,
    )

    explainer = shap.TreeExplainer(shap_model)
    shap_values_raw = explainer.shap_values(background)
    shap_values = _extract_positive_class_shap_values(shap_values_raw)

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    feature_importance = pd.DataFrame(
        {
            "feature_name": background.columns.tolist(),
            "mean_abs_shap": mean_abs.astype(float),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    feature_importance_path = output_dir / SHAP_FEATURE_IMPORTANCE_FILE
    feature_importance.to_csv(feature_importance_path, index=False)

    max_display = min(max(top_k, 5), background.shape[1])

    shap.summary_plot(
        shap_values,
        background,
        show=False,
        max_display=max_display,
    )
    summary_plot_path = output_dir / SHAP_SUMMARY_PLOT_FILE
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=140, bbox_inches="tight")
    plt.close("all")

    shap.summary_plot(
        shap_values,
        background,
        show=False,
        max_display=max_display,
        plot_type="bar",
    )
    bar_plot_path = output_dir / SHAP_BAR_PLOT_FILE
    plt.tight_layout()
    plt.savefig(bar_plot_path, dpi=140, bbox_inches="tight")
    plt.close("all")

    top_features = feature_importance.head(top_k).to_dict(orient="records")
    global_summary_payload = {
        "row_count": int(background.shape[0]),
        "feature_count": int(background.shape[1]),
        "sample_size_requested": int(sample_size),
        "sample_size_used": int(background.shape[0]),
        "top_k": int(top_k),
        "top_features": top_features,
        "model_metadata": {
            **model_metadata,
            **shap_model_info,
        },
    }
    global_summary_path = write_json(output_dir / SHAP_GLOBAL_SUMMARY_FILE, global_summary_payload)

    artifact_paths = {
        "shap_global_summary": global_summary_path,
        "shap_feature_importance": feature_importance_path,
        "shap_summary_plot": summary_plot_path,
        "shap_bar_plot": bar_plot_path,
    }
    return artifact_paths, global_summary_payload


def generate_shap_local_artifacts(
    *,
    model: Any,
    x_frame: DataFrame,
    selected_examples: DataFrame,
    top_k: int,
    threshold: float,
    output_dir: Path,
    model_metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], Path]:
    """Generate SHAP local explanation payloads for selected examples."""
    required_columns = {
        "row_index",
        "applicant_id",
        "cohort_name",
        "split_name",
        "predicted_probability",
        "predicted_label",
        "actual_label",
    }
    missing_columns = sorted(required_columns.difference(selected_examples.columns))
    if missing_columns:
        raise ValueError(f"Selected examples missing required columns: {missing_columns}")

    shap_model, shap_model_info = _resolve_tree_shap_model(model)
    row_indices = selected_examples["row_index"].astype(int).tolist()
    selected_frame = x_frame.iloc[row_indices].copy()

    explainer = shap.TreeExplainer(shap_model)
    shap_values_raw = explainer.shap_values(selected_frame)
    shap_values = _extract_positive_class_shap_values(shap_values_raw)

    payloads: list[dict[str, Any]] = []
    feature_names = selected_frame.columns.tolist()

    for position, (_, row) in enumerate(selected_examples.iterrows()):
        contribution_map = _build_feature_contribution_map(
            feature_names,
            np.asarray(shap_values[position], dtype=np.float64),
        )
        payload = build_local_explanation_payload(
            explanation_method="shap",
            applicant_id=int(row["applicant_id"]),
            cohort_name=cast(Any, row["cohort_name"]),
            split_name=str(row["split_name"]),
            predicted_probability=float(row["predicted_probability"]),
            predicted_label=int(row["predicted_label"]),
            actual_label=int(row["actual_label"]),
            threshold=float(threshold),
            feature_contributions=contribution_map,
            top_k=top_k,
            metadata={
                **model_metadata,
                **shap_model_info,
            },
        )
        payloads.append(payload)

    local_path = write_jsonl(output_dir / SHAP_LOCAL_EXPLANATIONS_FILE, payloads)
    return payloads, local_path


def load_joblib_model(path: Path) -> Any:
    """Load a joblib-serialized model from disk."""
    return joblib.load(path)
