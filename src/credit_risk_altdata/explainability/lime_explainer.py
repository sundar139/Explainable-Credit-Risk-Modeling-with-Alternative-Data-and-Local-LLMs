"""LIME local explainability generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lime.lime_tabular import LimeTabularExplainer  # type: ignore[import-untyped]
from pandas import DataFrame

from credit_risk_altdata.explainability.constants import LIME_LOCAL_EXPLANATIONS_FILE
from credit_risk_altdata.explainability.payloads import build_local_explanation_payload
from credit_risk_altdata.explainability.reporting import write_jsonl, write_markdown
from credit_risk_altdata.explainability.shap_explainer import predict_positive_probability
from credit_risk_altdata.logging import get_logger

LOGGER = get_logger(__name__)

_IDENTIFIER_OR_TARGET_COLUMNS = frozenset({"SK_ID_CURR", "TARGET"})
_DEFAULT_NEAR_ZERO_VARIANCE_THRESHOLD = 1e-8


@dataclass(frozen=True, slots=True)
class LimeMatrixPreparation:
    """Prepared and sanitized matrix plus preprocessing diagnostics."""

    frame: DataFrame
    model_input_frame: DataFrame
    original_feature_count: int
    non_numeric_removed_count: int
    identifier_target_removed_count: int
    instability_removed_count: int
    final_feature_count: int
    instability_removed_columns: list[str]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "lime_original_feature_count": self.original_feature_count,
            "lime_non_numeric_removed_count": self.non_numeric_removed_count,
            "lime_identifier_target_removed_count": self.identifier_target_removed_count,
            "lime_instability_removed_count": self.instability_removed_count,
            "lime_final_feature_count": self.final_feature_count,
            "lime_model_input_feature_count": int(len(self.model_input_frame.columns)),
        }


def _safe_file_token(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_\-]+", "_", value).strip("_") or "case"


def _coerce_actual_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_applicant_id(value: Any) -> int | str:
    value_str = str(value)
    return int(value_str) if value_str.isdigit() else value_str


def _prepare_lime_explainer_matrix(
    x_train: DataFrame,
    *,
    near_zero_variance_threshold: float = _DEFAULT_NEAR_ZERO_VARIANCE_THRESHOLD,
) -> LimeMatrixPreparation:
    """Build a stable numeric matrix for LIME perturbation."""
    if near_zero_variance_threshold < 0.0:
        raise ValueError("near_zero_variance_threshold must be non-negative")

    original_feature_count = int(len(x_train.columns))
    numeric_frame = x_train.select_dtypes(include=["number", "bool"]).copy()
    non_numeric_removed_count = original_feature_count - int(len(numeric_frame.columns))

    identifier_target_columns = [
        column
        for column in numeric_frame.columns
        if column.upper() in _IDENTIFIER_OR_TARGET_COLUMNS
    ]
    if identifier_target_columns:
        numeric_frame = numeric_frame.drop(columns=identifier_target_columns)

    model_input_frame = numeric_frame.replace([np.inf, -np.inf], np.nan)

    all_null_columns = [
        column
        for column in model_input_frame.columns
        if int(model_input_frame[column].notna().sum()) == 0
    ]

    if not model_input_frame.empty:
        medians = model_input_frame.median(axis=0, skipna=True)
        model_input_frame = model_input_frame.fillna(medians)
        model_input_frame = model_input_frame.fillna(0.0)
        matrix = np.asarray(model_input_frame, dtype=np.float64)
        matrix[~np.isfinite(matrix)] = 0.0
        model_input_frame = DataFrame(
            matrix,
            index=model_input_frame.index,
            columns=model_input_frame.columns,
            dtype=np.float64,
        )

    low_variance_columns: list[str] = []
    if not model_input_frame.empty:
        variance = model_input_frame.var(axis=0, skipna=True, ddof=0)
        low_variance_columns = variance[
            variance <= near_zero_variance_threshold
        ].index.tolist()

    instability_removed_columns = sorted(set(all_null_columns).union(low_variance_columns))
    stable_columns = [
        column
        for column in model_input_frame.columns.tolist()
        if column not in instability_removed_columns
    ]
    stable_frame = model_input_frame[stable_columns].copy()

    preparation = LimeMatrixPreparation(
        frame=stable_frame,
        model_input_frame=model_input_frame,
        original_feature_count=original_feature_count,
        non_numeric_removed_count=non_numeric_removed_count,
        identifier_target_removed_count=len(identifier_target_columns),
        instability_removed_count=len(instability_removed_columns),
        final_feature_count=int(len(stable_columns)),
        instability_removed_columns=instability_removed_columns,
    )

    LOGGER.info(
        "Prepared LIME matrix: instability_removed=%d final_feature_count=%d",
        preparation.instability_removed_count,
        preparation.final_feature_count,
    )
    return preparation


def _resolve_lime_intercept(explanation: Any) -> float:
    intercept = getattr(explanation, "intercept", None)
    if isinstance(intercept, dict):
        if 1 in intercept:
            return float(intercept[1])
        if intercept:
            return float(next(iter(intercept.values())))

    intercept_array = np.asarray(intercept, dtype=np.float64).reshape(-1)
    if intercept_array.size == 0:
        return 0.0
    if intercept_array.size > 1:
        return float(intercept_array[1])
    return float(intercept_array[0])


def _build_failure_payload(
    *,
    row: Any,
    threshold: float,
    top_k: int,
    model_metadata: dict[str, Any],
    preparation: LimeMatrixPreparation,
    failure_reason: str,
) -> dict[str, Any]:
    return build_local_explanation_payload(
        explanation_method="lime",
        applicant_id=_coerce_applicant_id(row["applicant_id"]),
        cohort_name=row["cohort_name"],
        split_name=str(row["split_name"]),
        predicted_probability=float(row["predicted_probability"]),
        predicted_label=int(row["predicted_label"]),
        actual_label=_coerce_actual_label(row["actual_label"]),
        threshold=float(threshold),
        feature_contributions={},
        top_k=top_k,
        metadata={
            **model_metadata,
            **preparation.as_metadata(),
        },
        explanation_generated=False,
        failure_reason=failure_reason,
        failed_feature_count=preparation.instability_removed_count,
    )


def generate_lime_local_artifacts(
    *,
    model: Any,
    x_train: DataFrame,
    selected_examples: DataFrame,
    top_k: int,
    threshold: float,
    random_seed: int,
    output_dir: Path,
    model_metadata: dict[str, Any],
    categorical_columns: list[str],
) -> tuple[list[dict[str, Any]], Path, list[Path]]:
    """Generate LIME local explanation payloads and case summaries."""
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

    preparation = _prepare_lime_explainer_matrix(x_train)
    feature_names = preparation.frame.columns.tolist()
    categorical_indices = [
        feature_names.index(column)
        for column in categorical_columns
        if column in feature_names
    ]

    payloads: list[dict[str, Any]] = []
    case_paths: list[Path] = []

    if selected_examples.empty:
        explanations_path = write_jsonl(output_dir / LIME_LOCAL_EXPLANATIONS_FILE, payloads)
        return payloads, explanations_path, case_paths

    if not feature_names:
        reason = "No stable numeric features remain after LIME preprocessing"
        LOGGER.warning(reason)
        for _, row in selected_examples.iterrows():
            payload = _build_failure_payload(
                row=row,
                threshold=threshold,
                top_k=top_k,
                model_metadata=model_metadata,
                preparation=preparation,
                failure_reason=reason,
            )
            payloads.append(payload)
            case_file = output_dir / (
                f"lime_{_safe_file_token(str(row['cohort_name']))}"
                f"_{_safe_file_token(str(row['applicant_id']))}.md"
            )
            case_paths.append(
                write_markdown(
                    case_file,
                    title="LIME Local Explanation",
                    lines=[
                        f"- Applicant ID: {row['applicant_id']}",
                        f"- Cohort: {row['cohort_name']}",
                        "- Explanation generated: False",
                        f"- Failure reason: {reason}",
                    ],
                )
            )

        explanations_path = write_jsonl(output_dir / LIME_LOCAL_EXPLANATIONS_FILE, payloads)
        return payloads, explanations_path, case_paths

    training_matrix = np.asarray(preparation.frame, dtype=np.float64)
    try:
        explainer = LimeTabularExplainer(
            training_data=training_matrix,
            feature_names=feature_names,
            class_names=["non_default", "default"],
            categorical_features=categorical_indices,
            mode="classification",
            discretize_continuous=True,
            random_state=random_seed,
        )
    except Exception as exc:
        reason = f"LIME explainer initialization failed: {type(exc).__name__}: {exc}"
        LOGGER.warning(reason)
        for _, row in selected_examples.iterrows():
            payload = _build_failure_payload(
                row=row,
                threshold=threshold,
                top_k=top_k,
                model_metadata=model_metadata,
                preparation=preparation,
                failure_reason=reason,
            )
            payloads.append(payload)
        explanations_path = write_jsonl(output_dir / LIME_LOCAL_EXPLANATIONS_FILE, payloads)
        return payloads, explanations_path, case_paths

    model_feature_names = preparation.model_input_frame.columns.tolist()
    model_feature_lookup = {
        feature_name: index
        for index, feature_name in enumerate(model_feature_names)
    }
    stable_feature_positions = np.asarray(
        [model_feature_lookup[feature_name] for feature_name in feature_names],
        dtype=np.int64,
    )

    for _, row in selected_examples.iterrows():
        case_file = output_dir / (
            f"lime_{_safe_file_token(str(row['cohort_name']))}_"
            f"{_safe_file_token(str(row['applicant_id']))}.md"
        )
        try:
            row_index = int(row["row_index"])
            instance = np.asarray(preparation.frame.iloc[row_index], dtype=np.float64)
            reference_row = np.asarray(
                preparation.model_input_frame.iloc[row_index],
                dtype=np.float64,
            )

            # Keep model input dimensionality stable while perturbing only robust features.
            def lime_predict_fn(
                rows: np.ndarray,
                reference_row_bound: np.ndarray = reference_row,
            ) -> np.ndarray:
                rows_matrix = np.asarray(rows, dtype=np.float64)
                if rows_matrix.ndim == 1:
                    rows_matrix = rows_matrix.reshape(1, -1)

                full_matrix = np.repeat(
                    reference_row_bound[np.newaxis, :],
                    repeats=rows_matrix.shape[0],
                    axis=0,
                )
                full_matrix[:, stable_feature_positions] = rows_matrix

                frame = DataFrame(full_matrix, columns=model_feature_names)
                positive = predict_positive_probability(model, x_frame=frame)
                negative = 1.0 - positive
                return np.vstack([negative, positive]).T

            explanation = explainer.explain_instance(
                data_row=instance,
                predict_fn=lime_predict_fn,
                num_features=top_k,
                labels=(1,),
            )
            explanation_items = explanation.as_list(label=1)
            contribution_map = {
                str(feature_rule): float(weight)
                for feature_rule, weight in explanation_items
            }

            payload = build_local_explanation_payload(
                explanation_method="lime",
                applicant_id=_coerce_applicant_id(row["applicant_id"]),
                cohort_name=row["cohort_name"],
                split_name=str(row["split_name"]),
                predicted_probability=float(row["predicted_probability"]),
                predicted_label=int(row["predicted_label"]),
                actual_label=_coerce_actual_label(row["actual_label"]),
                threshold=float(threshold),
                feature_contributions=contribution_map,
                top_k=top_k,
                metadata={
                    **model_metadata,
                    **preparation.as_metadata(),
                    "lime_local_prediction": float(np.asarray(explanation.local_pred)[0]),
                    "lime_intercept": _resolve_lime_intercept(explanation),
                    "lime_score": float(explanation.score),
                },
                explanation_generated=True,
                failure_reason=None,
                failed_feature_count=0,
            )
            payloads.append(payload)

            actual_label = _coerce_actual_label(row["actual_label"])
            lines = [
                f"- Applicant ID: {row['applicant_id']}",
                f"- Cohort: {row['cohort_name']}",
                f"- Predicted probability: {float(row['predicted_probability']):.6f}",
                f"- Predicted label: {int(row['predicted_label'])}",
                f"- Actual label: {actual_label}",
                f"- Threshold: {float(threshold):.6f}",
                "- Explanation generated: True",
                "",
                "## Top LIME Feature Rules",
            ]
            for feature_rule, weight in explanation_items:
                lines.append(f"- {feature_rule}: {float(weight):.6f}")
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            LOGGER.warning(
                "LIME explanation failed for applicant_id=%s cohort=%s: %s",
                row["applicant_id"],
                row["cohort_name"],
                reason,
            )
            payload = _build_failure_payload(
                row=row,
                threshold=threshold,
                top_k=top_k,
                model_metadata=model_metadata,
                preparation=preparation,
                failure_reason=reason,
            )
            payloads.append(payload)
            lines = [
                f"- Applicant ID: {row['applicant_id']}",
                f"- Cohort: {row['cohort_name']}",
                f"- Predicted probability: {float(row['predicted_probability']):.6f}",
                f"- Predicted label: {int(row['predicted_label'])}",
                f"- Actual label: {_coerce_actual_label(row['actual_label'])}",
                f"- Threshold: {float(threshold):.6f}",
                "- Explanation generated: False",
                f"- Failure reason: {reason}",
            ]

        case_paths.append(
            write_markdown(
                case_file,
                title="LIME Local Explanation",
                lines=lines,
            )
        )

    explanations_path = write_jsonl(output_dir / LIME_LOCAL_EXPLANATIONS_FILE, payloads)
    return payloads, explanations_path, case_paths
