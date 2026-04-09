"""Model and schema loading utilities with in-process caching for API scoring."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier  # type: ignore[import-untyped]
from pandas import DataFrame

from credit_risk_altdata.config import Settings

_IDENTIFIER_COLUMN = "SK_ID_CURR"
_TARGET_COLUMN = "TARGET"


class ModelStore:
    """Cache-aware loader for final candidate metadata, model artifact, and feature schema."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = RLock()
        self._candidate_summary_cache: dict[str, Any] | None = None
        self._model_cache: Any | None = None
        self._model_path_cache: Path | None = None
        self._feature_columns_cache: list[str] | None = None

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def feature_manifest_path(self) -> Path:
        return self._settings.feature_metadata_dir / "feature_manifest.csv"

    def _load_candidate_summary(self) -> dict[str, Any]:
        summary_path = self._settings.modeling_final_candidate_summary_path
        if not summary_path.exists():
            raise FileNotFoundError(
                "Final production candidate summary is missing. "
                f"Expected: {summary_path}"
            )

        payload_raw = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload_raw, dict):
            raise ValueError("Final candidate summary must be a JSON object")

        payload = dict(payload_raw)
        required_keys = {
            "final_candidate_name",
            "final_model_family",
            "threshold",
        }
        missing_keys = sorted(required_keys.difference(payload.keys()))
        if missing_keys:
            raise ValueError(f"Final candidate summary missing required keys: {missing_keys}")
        return payload

    def get_final_candidate_summary(self) -> dict[str, Any]:
        """Return cached final-candidate metadata."""
        with self._lock:
            if self._candidate_summary_cache is None:
                self._candidate_summary_cache = self._load_candidate_summary()
            return dict(self._candidate_summary_cache)

    def get_model_artifact_path(self) -> Path:
        """Resolve final model artifact path from candidate summary metadata."""
        summary = self.get_final_candidate_summary()
        preferred = summary.get("final_model_output_path")
        fallback = summary.get("selected_artifact_path")
        raw_path = preferred if preferred else fallback
        if not raw_path:
            raise ValueError(
                "Final candidate summary must include final_model_output_path "
                "or selected_artifact_path"
            )

        path = Path(str(raw_path))
        if not path.is_absolute():
            path = self._settings.project_root / path

        if not path.exists():
            raise FileNotFoundError(f"Final production model artifact not found: {path}")
        return path

    def _load_model(self, path: Path) -> Any:
        suffix = path.suffix.lower()
        if suffix == ".joblib":
            return joblib.load(path)
        if suffix == ".cbm":
            model = CatBoostClassifier()
            model.load_model(str(path))
            return model
        raise ValueError(
            "Unsupported final model artifact format for scoring: "
            f"{path.suffix}. Supported suffixes: .joblib, .cbm"
        )

    def get_model(self) -> Any:
        """Return cached production model instance."""
        with self._lock:
            model_path = self.get_model_artifact_path()
            if self._model_cache is None or self._model_path_cache != model_path:
                self._model_cache = self._load_model(model_path)
                self._model_path_cache = model_path
            return self._model_cache

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return bool(int(value))
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        return False

    def _load_expected_feature_columns(self) -> list[str]:
        manifest_path = self.feature_manifest_path
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            if "feature_name" not in manifest.columns:
                raise ValueError(
                    f"Feature manifest missing feature_name column: {manifest_path}"
                )

            if "is_target" in manifest.columns and "is_identifier" in manifest.columns:
                feature_rows = manifest[
                    (~manifest["is_target"].map(self._to_bool))
                    & (~manifest["is_identifier"].map(self._to_bool))
                ]
            else:
                feature_rows = manifest[
                    ~manifest["feature_name"].astype(str).isin(
                        [_IDENTIFIER_COLUMN, _TARGET_COLUMN]
                    )
                ]

            columns = feature_rows["feature_name"].astype(str).tolist()
            if columns:
                return columns

        model = self.get_model()
        feature_names_in = getattr(model, "feature_names_in_", None)
        if feature_names_in is not None:
            columns = [str(name) for name in feature_names_in]
            if columns:
                return columns

        feature_name_attr = getattr(model, "feature_name_", None)
        if isinstance(feature_name_attr, list):
            columns = [str(name) for name in feature_name_attr]
            if columns:
                return columns

        raise FileNotFoundError(
            "Expected feature schema is unavailable. Provide feature_manifest.csv under "
            f"{self._settings.feature_metadata_dir} or use a model with feature names metadata."
        )

    def get_expected_feature_columns(self) -> list[str]:
        """Return cached expected engineered-feature columns for scoring."""
        with self._lock:
            if self._feature_columns_cache is None:
                self._feature_columns_cache = self._load_expected_feature_columns()
            return list(self._feature_columns_cache)

    @staticmethod
    def _coerce_numeric_feature(value: float | int | bool | None, feature_name: str) -> float:
        if value is None:
            raise ValueError(
                f"Feature '{feature_name}' has null value. All required engineered features "
                "must be provided as numeric values."
            )

        coerced = float(int(value)) if isinstance(value, bool) else float(value)

        if not np.isfinite(coerced):
            raise ValueError(
                f"Feature '{feature_name}' has non-finite value: {value}"
            )
        return coerced

    def build_scoring_frame(
        self,
        engineered_features: dict[str, float | int | bool | None],
    ) -> tuple[DataFrame, list[str]]:
        """Validate and normalize one-row engineered features into model input frame."""
        expected_columns = self.get_expected_feature_columns()
        expected_set = set(expected_columns)
        supplied_set = set(engineered_features)

        missing_features = sorted(expected_set.difference(supplied_set))
        if missing_features:
            raise ValueError(
                "Missing required engineered features: "
                f"{missing_features}"
            )

        extra_features = sorted(supplied_set.difference(expected_set))
        warnings: list[str] = []
        if extra_features:
            warnings.append(
                "Ignored extra engineered features not used by model: "
                f"{extra_features}"
            )

        row = {
            feature_name: self._coerce_numeric_feature(
                engineered_features[feature_name],
                feature_name,
            )
            for feature_name in expected_columns
        }
        frame = pd.DataFrame([row], columns=expected_columns, dtype=np.float64)
        return frame, warnings

    @staticmethod
    def predict_positive_probability(model: Any, x_frame: DataFrame) -> float:
        """Predict positive-class probability for one-row input."""
        probability_matrix = np.asarray(model.predict_proba(x_frame), dtype=np.float64)
        if probability_matrix.ndim != 2 or probability_matrix.shape[1] < 2:
            raise ValueError(
                "Model predict_proba output is not a two-class probability matrix"
            )
        return float(probability_matrix[0, 1])
