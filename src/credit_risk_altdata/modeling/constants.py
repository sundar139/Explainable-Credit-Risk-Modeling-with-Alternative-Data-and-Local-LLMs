"""Constants used by baseline modeling workflows."""

from __future__ import annotations

from typing import Final, Literal

from credit_risk_altdata.features.constants import ENTITY_ID_COLUMN, TARGET_COLUMN

MODEL_LIGHTGBM: Final[Literal["lightgbm"]] = "lightgbm"
MODEL_CATBOOST: Final[Literal["catboost"]] = "catboost"
MODEL_ALL: Final[Literal["all"]] = "all"

CALIBRATION_NONE: Final[Literal["none"]] = "none"
CALIBRATION_SIGMOID: Final[Literal["sigmoid"]] = "sigmoid"
CALIBRATION_ISOTONIC: Final[Literal["isotonic"]] = "isotonic"
CALIBRATION_ALL: Final[Literal["all"]] = "all"

ModelFamily = Literal["lightgbm", "catboost"]
ModelSelection = Literal["all", "lightgbm", "catboost"]
CalibrationMethod = Literal["none", "sigmoid", "isotonic"]
CalibrationSelection = Literal["none", "sigmoid", "isotonic", "all"]

SUPPORTED_MODEL_FAMILIES: Final[tuple[ModelFamily, ...]] = (MODEL_LIGHTGBM, MODEL_CATBOOST)
SUPPORTED_MODEL_SELECTIONS: Final[tuple[ModelSelection, ...]] = (
    MODEL_ALL,
    MODEL_LIGHTGBM,
    MODEL_CATBOOST,
)
SUPPORTED_CALIBRATION_METHODS: Final[tuple[CalibrationMethod, ...]] = (
    CALIBRATION_NONE,
    CALIBRATION_SIGMOID,
    CALIBRATION_ISOTONIC,
)
SUPPORTED_CALIBRATION_SELECTIONS: Final[tuple[CalibrationSelection, ...]] = (
    CALIBRATION_NONE,
    CALIBRATION_SIGMOID,
    CALIBRATION_ISOTONIC,
    CALIBRATION_ALL,
)

PRIMARY_METRIC = "roc_auc"

FOLD_METRICS_FILE = "fold_metrics.csv"
MODEL_COMPARISON_FILE = "model_comparison.csv"
OOF_PREDICTIONS_FILE = "oof_predictions.parquet"
TEST_PREDICTIONS_FILE = "test_predictions.parquet"
BEST_MODEL_SUMMARY_FILE = "best_model_summary.json"
MODELING_SUMMARY_FILE = "baseline_modeling_summary.md"
TUNING_RESULTS_FILE = "tuning_results.csv"
TUNED_MODEL_COMPARISON_FILE = "tuned_model_comparison.csv"
CALIBRATION_COMPARISON_FILE = "calibration_comparison.csv"
THRESHOLD_ANALYSIS_FILE = "threshold_analysis.csv"
EVALUATION_SUMMARY_FILE = "evaluation_summary.json"
TUNED_MODELING_SUMMARY_FILE = "tuned_modeling_summary.md"
FINAL_PRODUCTION_CANDIDATE_FILE = "final_production_candidate.json"
BEST_PARAMS_LIGHTGBM_FILE = "best_params_lightgbm.json"
BEST_PARAMS_CATBOOST_FILE = "best_params_catboost.json"

IDENTIFIER_COLUMN = ENTITY_ID_COLUMN
LABEL_COLUMN = TARGET_COLUMN
