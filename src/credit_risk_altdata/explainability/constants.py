"""Constants for explainability workflows."""

from __future__ import annotations

from typing import Final, Literal

METHOD_ALL: Final[Literal["all"]] = "all"
METHOD_SHAP: Final[Literal["shap"]] = "shap"
METHOD_LIME: Final[Literal["lime"]] = "lime"

ExplainabilityMethodSelection = Literal["all", "shap", "lime"]
ExplanationMethod = Literal["shap", "lime"]

COHORT_TRUE_POSITIVE: Final[Literal["true_positive"]] = "true_positive"
COHORT_TRUE_NEGATIVE: Final[Literal["true_negative"]] = "true_negative"
COHORT_FALSE_POSITIVE: Final[Literal["false_positive"]] = "false_positive"
COHORT_FALSE_NEGATIVE: Final[Literal["false_negative"]] = "false_negative"
COHORT_BORDERLINE: Final[Literal["borderline_threshold"]] = "borderline_threshold"

CohortName = Literal[
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
    "borderline_threshold",
]

SUPPORTED_METHOD_SELECTIONS: Final[tuple[ExplainabilityMethodSelection, ...]] = (
    METHOD_ALL,
    METHOD_SHAP,
    METHOD_LIME,
)
SUPPORTED_EXPLANATION_METHODS: Final[tuple[ExplanationMethod, ...]] = (
    METHOD_SHAP,
    METHOD_LIME,
)
SUPPORTED_COHORTS: Final[tuple[CohortName, ...]] = (
    COHORT_TRUE_POSITIVE,
    COHORT_TRUE_NEGATIVE,
    COHORT_FALSE_POSITIVE,
    COHORT_FALSE_NEGATIVE,
    COHORT_BORDERLINE,
)

SELECTED_EXAMPLES_FILE = "selected_examples.csv"
SHAP_GLOBAL_SUMMARY_FILE = "shap_global_summary.json"
SHAP_FEATURE_IMPORTANCE_FILE = "shap_feature_importance.csv"
SHAP_SUMMARY_PLOT_FILE = "shap_summary_plot.png"
SHAP_BAR_PLOT_FILE = "shap_bar_plot.png"
SHAP_LOCAL_EXPLANATIONS_FILE = "shap_local_explanations.jsonl"
LIME_LOCAL_EXPLANATIONS_FILE = "lime_explanations.jsonl"
EXPLAINABILITY_SUMMARY_FILE = "explainability_summary.md"
