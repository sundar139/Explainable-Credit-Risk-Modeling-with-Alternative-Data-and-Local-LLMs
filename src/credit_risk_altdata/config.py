"""Configuration loading and validation for the project."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AnyHttpUrl, Field, TypeAdapter, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from credit_risk_altdata.data.constants import (
    HOME_CREDIT_INTERIM_SUBDIR,
    HOME_CREDIT_RAW_SUBDIR,
)


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "credit-risk-altdata"
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_code_model: str = "qwen2.5-coder:7b"
    ollama_chat_model: str = "qwen2.5:7b"
    ollama_timeout_seconds: int = Field(default=30, ge=1, le=300)
    llm_reports_model_name: str = "qwen2.5:7b"
    llm_reports_report_type: Literal["all", "plain", "underwriter", "adverse-action"] = "all"
    llm_reports_method_source: Literal["auto", "shap", "lime"] = "auto"
    llm_reports_max_cases: int = Field(default=50, ge=1, le=10_000)
    llm_reports_timeout_seconds: int = Field(default=30, ge=1, le=600)
    llm_reports_retries: int = Field(default=2, ge=0, le=10)
    llm_reports_enable_fallback: bool = True
    llm_reports_output_dir: Path = Path("artifacts/llm_reports")

    kaggle_dataset: str = "home-credit-default-risk"
    kaggle_username: str | None = None
    kaggle_key: str | None = None

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    random_seed: int = 42
    modeling_folds: int = Field(default=5, ge=2, le=20)
    modeling_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    modeling_primary_metric: Literal["roc_auc"] = "roc_auc"
    modeling_tuning_trials: int = Field(default=20, ge=1, le=500)
    modeling_calibration_strategy: Literal["none", "sigmoid", "isotonic", "all"] = "all"
    modeling_threshold_grid_min: float = Field(default=0.1, ge=0.0, lt=1.0)
    modeling_threshold_grid_max: float = Field(default=0.9, gt=0.0, le=1.0)
    modeling_threshold_grid_step: float = Field(default=0.05, gt=0.0, lt=1.0)
    modeling_final_model_path: Path = Path(
        "artifacts/modeling/models/final_production_model.joblib"
    )
    explainability_input_path: Path = Path(
        "artifacts/modeling/predictions/tuned_oof_predictions.parquet"
    )
    explainability_sample_size: int = Field(default=1000, ge=10, le=1_000_000)
    explainability_top_k: int = Field(default=10, ge=1, le=100)
    explainability_random_seed: int = 42
    explainability_true_positive_examples: int = Field(default=3, ge=0, le=50)
    explainability_true_negative_examples: int = Field(default=3, ge=0, le=50)
    explainability_false_positive_examples: int = Field(default=3, ge=0, le=50)
    explainability_false_negative_examples: int = Field(default=3, ge=0, le=50)
    explainability_borderline_examples: int = Field(default=3, ge=0, le=50)
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])

    @field_validator("ollama_code_model", "ollama_chat_model", "llm_reports_model_name")
    @classmethod
    def validate_model_identifier(cls, value: str) -> str:
        """Validate model tags such as qwen2.5:7b."""
        model_name = value.strip()
        if not model_name or ":" not in model_name:
            raise ValueError("Model identifier must include a tag, for example qwen2.5:7b")
        return model_name

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, value: str) -> str:
        """Validate that OLLAMA_BASE_URL is a valid HTTP URL."""
        parsed = TypeAdapter(AnyHttpUrl).validate_python(value)
        return str(parsed).rstrip("/")

    @field_validator(
        "data_dir",
        "artifacts_dir",
        "modeling_final_model_path",
        "explainability_input_path",
        "llm_reports_output_dir",
    )
    @classmethod
    def validate_relative_paths(cls, value: Path) -> Path:
        """Force relative paths so project root controls all storage locations."""
        if value.is_absolute():
            raise ValueError("Directory settings must be relative to project root")
        return value

    @model_validator(mode="after")
    def validate_kaggle_credentials(self) -> Settings:
        """Require Kaggle username and key to be supplied together if used."""
        has_username = bool(self.kaggle_username)
        has_key = bool(self.kaggle_key)
        if has_username != has_key:
            raise ValueError("Set both KAGGLE_USERNAME and KAGGLE_KEY together, or neither")
        if self.modeling_threshold_grid_min >= self.modeling_threshold_grid_max:
            raise ValueError(
                "MODELING_THRESHOLD_GRID_MIN must be less than "
                "MODELING_THRESHOLD_GRID_MAX"
            )
        if (
            self.explainability_true_positive_examples
            + self.explainability_true_negative_examples
            + self.explainability_false_positive_examples
            + self.explainability_false_negative_examples
            + self.explainability_borderline_examples
            <= 0
        ):
            raise ValueError(
                "At least one explainability selection count must be greater than zero"
            )
        return self

    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / self.data_dir / "raw"

    @property
    def interim_data_dir(self) -> Path:
        return self.project_root / self.data_dir / "interim"

    @property
    def processed_data_dir(self) -> Path:
        return self.project_root / self.data_dir / "processed"

    @property
    def resolved_artifacts_dir(self) -> Path:
        return self.project_root / self.artifacts_dir

    @property
    def home_credit_raw_dir(self) -> Path:
        return self.raw_data_dir / HOME_CREDIT_RAW_SUBDIR

    @property
    def home_credit_interim_dir(self) -> Path:
        return self.interim_data_dir / HOME_CREDIT_INTERIM_SUBDIR

    @property
    def data_validation_dir(self) -> Path:
        return self.resolved_artifacts_dir / "data_validation"

    @property
    def home_credit_processed_dir(self) -> Path:
        return self.processed_data_dir / "home_credit"

    @property
    def feature_metadata_dir(self) -> Path:
        return self.resolved_artifacts_dir / "feature_metadata"

    @property
    def modeling_dir(self) -> Path:
        return self.resolved_artifacts_dir / "modeling"

    @property
    def modeling_metrics_dir(self) -> Path:
        return self.modeling_dir / "metrics"

    @property
    def modeling_predictions_dir(self) -> Path:
        return self.modeling_dir / "predictions"

    @property
    def modeling_feature_importance_dir(self) -> Path:
        return self.modeling_dir / "feature_importance"

    @property
    def modeling_models_dir(self) -> Path:
        return self.modeling_dir / "models"

    @property
    def modeling_reports_dir(self) -> Path:
        return self.modeling_dir / "reports"

    @property
    def modeling_tuning_dir(self) -> Path:
        return self.modeling_dir / "tuning"

    @property
    def modeling_calibration_dir(self) -> Path:
        return self.modeling_dir / "calibration"

    @property
    def modeling_evaluation_dir(self) -> Path:
        return self.modeling_dir / "evaluation"

    @property
    def modeling_final_model_output_path(self) -> Path:
        return self.project_root / self.modeling_final_model_path

    @property
    def modeling_final_candidate_summary_path(self) -> Path:
        from credit_risk_altdata.modeling.constants import FINAL_PRODUCTION_CANDIDATE_FILE

        return self.modeling_reports_dir / FINAL_PRODUCTION_CANDIDATE_FILE

    @property
    def explainability_input_predictions_path(self) -> Path:
        return self.project_root / self.explainability_input_path

    @property
    def explainability_root_dir(self) -> Path:
        return self.resolved_artifacts_dir / "explainability"

    @property
    def explainability_dir(self) -> Path:
        return self.explainability_root_dir

    @property
    def explainability_shap_dir(self) -> Path:
        return self.explainability_root_dir / "shap"

    @property
    def explainability_selected_examples_dir(self) -> Path:
        return self.explainability_dir / "selected_examples"

    @property
    def explainability_reports_dir(self) -> Path:
        return self.explainability_dir / "reports"

    @property
    def explainability_shap_global_dir(self) -> Path:
        return self.explainability_shap_dir / "global"

    @property
    def explainability_shap_local_dir(self) -> Path:
        return self.explainability_shap_dir / "local"

    @property
    def explainability_lime_dir(self) -> Path:
        return self.explainability_dir / "lime"

    @property
    def llm_reports_root_dir(self) -> Path:
        return self.project_root / self.llm_reports_output_dir

    @property
    def llm_reports_plain_language_dir(self) -> Path:
        return self.llm_reports_root_dir / "plain_language"

    @property
    def llm_reports_underwriter_dir(self) -> Path:
        return self.llm_reports_root_dir / "underwriter"

    @property
    def llm_reports_adverse_action_dir(self) -> Path:
        return self.llm_reports_root_dir / "adverse_action_drafts"

    @property
    def llm_reports_combined_dir(self) -> Path:
        return self.llm_reports_root_dir / "combined"

    @property
    def llm_reports_reports_dir(self) -> Path:
        return self.llm_reports_root_dir / "reports"

    def safe_dump(self) -> dict[str, Any]:
        """Return settings for logs and diagnostics without exposing secrets."""
        data = self.model_dump(mode="json")
        if data.get("kaggle_key"):
            data["kaggle_key"] = "***"
        data["raw_data_dir"] = str(self.raw_data_dir)
        data["interim_data_dir"] = str(self.interim_data_dir)
        data["processed_data_dir"] = str(self.processed_data_dir)
        data["resolved_artifacts_dir"] = str(self.resolved_artifacts_dir)
        data["home_credit_raw_dir"] = str(self.home_credit_raw_dir)
        data["home_credit_interim_dir"] = str(self.home_credit_interim_dir)
        data["home_credit_processed_dir"] = str(self.home_credit_processed_dir)
        data["data_validation_dir"] = str(self.data_validation_dir)
        data["feature_metadata_dir"] = str(self.feature_metadata_dir)
        data["modeling_dir"] = str(self.modeling_dir)
        data["modeling_metrics_dir"] = str(self.modeling_metrics_dir)
        data["modeling_predictions_dir"] = str(self.modeling_predictions_dir)
        data["modeling_feature_importance_dir"] = str(self.modeling_feature_importance_dir)
        data["modeling_models_dir"] = str(self.modeling_models_dir)
        data["modeling_reports_dir"] = str(self.modeling_reports_dir)
        data["modeling_tuning_dir"] = str(self.modeling_tuning_dir)
        data["modeling_calibration_dir"] = str(self.modeling_calibration_dir)
        data["modeling_evaluation_dir"] = str(self.modeling_evaluation_dir)
        data["modeling_final_model_output_path"] = str(self.modeling_final_model_output_path)
        data["modeling_final_candidate_summary_path"] = str(
            self.modeling_final_candidate_summary_path
        )
        data["explainability_input_predictions_path"] = str(
            self.explainability_input_predictions_path
        )
        data["explainability_root_dir"] = str(self.explainability_root_dir)
        data["explainability_dir"] = str(self.explainability_dir)
        data["explainability_shap_dir"] = str(self.explainability_shap_dir)
        data["explainability_selected_examples_dir"] = str(
            self.explainability_selected_examples_dir
        )
        data["explainability_reports_dir"] = str(self.explainability_reports_dir)
        data["explainability_shap_global_dir"] = str(self.explainability_shap_global_dir)
        data["explainability_shap_local_dir"] = str(self.explainability_shap_local_dir)
        data["explainability_lime_dir"] = str(self.explainability_lime_dir)
        data["llm_reports_root_dir"] = str(self.llm_reports_root_dir)
        data["llm_reports_plain_language_dir"] = str(self.llm_reports_plain_language_dir)
        data["llm_reports_underwriter_dir"] = str(self.llm_reports_underwriter_dir)
        data["llm_reports_adverse_action_dir"] = str(self.llm_reports_adverse_action_dir)
        data["llm_reports_combined_dir"] = str(self.llm_reports_combined_dir)
        data["llm_reports_reports_dir"] = str(self.llm_reports_reports_dir)
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Create a cached settings instance with validated values."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear settings cache for testing and controlled reloads."""
    get_settings.cache_clear()
