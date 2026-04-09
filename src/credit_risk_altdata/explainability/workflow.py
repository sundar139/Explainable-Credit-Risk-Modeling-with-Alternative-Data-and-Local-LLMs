"""End-to-end explainability workflow orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import pandas as pd
from catboost import CatBoostClassifier  # type: ignore[import-untyped]

from credit_risk_altdata.config import Settings
from credit_risk_altdata.explainability.constants import (
    EXPLAINABILITY_SUMMARY_FILE,
    LIME_LOCAL_EXPLANATIONS_FILE,
    METHOD_ALL,
    METHOD_LIME,
    METHOD_SHAP,
    SELECTED_EXAMPLES_FILE,
    SHAP_BAR_PLOT_FILE,
    SHAP_FEATURE_IMPORTANCE_FILE,
    SHAP_GLOBAL_SUMMARY_FILE,
    SHAP_LOCAL_EXPLANATIONS_FILE,
    SHAP_SUMMARY_PLOT_FILE,
    ExplainabilityMethodSelection,
)
from credit_risk_altdata.explainability.lime_explainer import generate_lime_local_artifacts
from credit_risk_altdata.explainability.reporting import (
    ExplainabilityArtifactPaths,
    resolve_explainability_artifact_paths,
    write_dataframe_csv,
    write_jsonl,
    write_markdown,
)
from credit_risk_altdata.explainability.selection import (
    build_prediction_frame,
    select_representative_examples,
)
from credit_risk_altdata.explainability.shap_explainer import (
    generate_shap_global_artifacts,
    generate_shap_local_artifacts,
)
from credit_risk_altdata.logging import get_logger
from credit_risk_altdata.modeling.data_prep import ModelingDataset, prepare_modeling_dataset

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExplainabilityWorkflowResult:
    """Output artifact locations from explainability workflow."""

    selected_examples_path: Path
    shap_global_summary_path: Path | None
    shap_feature_importance_path: Path | None
    shap_summary_plot_path: Path | None
    shap_bar_plot_path: Path | None
    shap_local_explanations_path: Path | None
    lime_explanations_path: Path | None
    explainability_summary_path: Path


def _resolve_methods(method_selection: ExplainabilityMethodSelection) -> list[str]:
    if method_selection == METHOD_ALL:
        return [METHOD_SHAP, METHOD_LIME]
    if method_selection == METHOD_SHAP:
        return [METHOD_SHAP]
    if method_selection == METHOD_LIME:
        return [METHOD_LIME]
    raise ValueError(f"Unsupported explainability method: {method_selection}")


def _check_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing_paths = [path for path in paths if path.exists()]
    if existing_paths:
        raise FileExistsError(
            "Explainability artifacts already exist. Use overwrite=True to replace them. "
            f"Existing: {[str(path) for path in existing_paths]}"
        )


def _load_final_candidate_summary(settings: Settings) -> dict[str, Any]:
    summary_path = settings.modeling_final_candidate_summary_path
    if not summary_path.exists():
        raise FileNotFoundError(
            "Final production candidate summary is missing. "
            f"Expected: {summary_path}. Run tune-models first."
        )

    payload_raw = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, dict):
        raise ValueError("Final candidate summary must be a JSON object")
    payload = cast(dict[str, Any], payload_raw)
    required_keys = {
        "final_candidate_name",
        "final_model_family",
        "threshold",
    }
    missing_keys = sorted(required_keys.difference(payload.keys()))
    if missing_keys:
        raise ValueError(f"Final candidate summary missing required keys: {missing_keys}")
    return payload


def _resolve_model_artifact_path(settings: Settings, candidate_summary: dict[str, Any]) -> Path:
    preferred = candidate_summary.get("final_model_output_path")
    fallback = candidate_summary.get("selected_artifact_path")
    raw_path = preferred if preferred else fallback
    if not raw_path:
        raise ValueError(
            "Final candidate summary must include final_model_output_path or selected_artifact_path"
        )

    path = Path(str(raw_path))
    if not path.is_absolute():
        path = settings.project_root / path

    if not path.exists():
        raise FileNotFoundError(f"Final production model artifact not found: {path}")
    return path


def _load_model(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".joblib":
        return joblib.load(path)
    if suffix == ".cbm":
        model = CatBoostClassifier()
        model.load_model(str(path))
        return model
    raise ValueError(
        "Unsupported final model artifact format for explainability: "
        f"{path.suffix}. Supported suffixes: .joblib, .cbm"
    )


def _resolve_oof_prediction_column(candidate_name: str) -> str:
    return f"oof_pred_{candidate_name}"


def _build_selected_examples(
    *,
    settings: Settings,
    candidate_summary: dict[str, Any],
    paths: ExplainabilityArtifactPaths,
    input_predictions_path: Path,
) -> tuple[pd.DataFrame, Path, ModelingDataset]:
    dataset = prepare_modeling_dataset(settings)

    if not input_predictions_path.exists():
        raise FileNotFoundError(
            "Prediction input for explainability is missing. "
            f"Expected: {input_predictions_path}."
        )

    oof_predictions = pd.read_parquet(input_predictions_path)
    required_columns = {"SK_ID_CURR", "TARGET"}
    missing_required = sorted(required_columns.difference(oof_predictions.columns))
    if missing_required:
        raise ValueError(
            f"Tuned OOF predictions missing required columns: {missing_required}"
        )

    candidate_name = str(candidate_summary["final_candidate_name"])
    prediction_column = _resolve_oof_prediction_column(candidate_name)
    if prediction_column not in oof_predictions.columns:
        raise ValueError(
            "Tuned OOF predictions missing final-candidate column: "
            f"{prediction_column}"
        )

    threshold = float(candidate_summary["threshold"])
    prediction_frame = build_prediction_frame(
        applicant_ids=oof_predictions["SK_ID_CURR"].astype(int).tolist(),
        actual_labels=oof_predictions["TARGET"].astype(int).tolist(),
        predicted_probabilities=oof_predictions[prediction_column].astype(float).tolist(),
        threshold=threshold,
        split_name="train_oof",
    )

    selected_examples = select_representative_examples(
        prediction_frame=prediction_frame,
        threshold=threshold,
        true_positive_count=settings.explainability_true_positive_examples,
        true_negative_count=settings.explainability_true_negative_examples,
        false_positive_count=settings.explainability_false_positive_examples,
        false_negative_count=settings.explainability_false_negative_examples,
        borderline_count=settings.explainability_borderline_examples,
    )

    id_to_row_index = {
        int(applicant_id): int(index)
        for index, applicant_id in enumerate(dataset.train_ids.astype(int).tolist())
    }
    selected_examples["row_index"] = selected_examples["applicant_id"].map(id_to_row_index)

    if selected_examples["row_index"].isna().any():
        raise ValueError("Selected examples include applicant IDs missing from training matrix")
    selected_examples["row_index"] = selected_examples["row_index"].astype(int)

    selected_examples_path = write_dataframe_csv(
        paths.selected_examples_dir / SELECTED_EXAMPLES_FILE,
        selected_examples,
    )
    return selected_examples, selected_examples_path, dataset


def run_explainability_workflow(
    settings: Settings,
    *,
    method_selection: ExplainabilityMethodSelection = METHOD_ALL,
    sample_size: int | None = None,
    top_k: int | None = None,
    input_path_override: Path | None = None,
    overwrite: bool = False,
) -> ExplainabilityWorkflowResult:
    """Generate representative examples, SHAP artifacts, and LIME artifacts."""
    paths = resolve_explainability_artifact_paths(settings)
    methods = _resolve_methods(method_selection)
    input_predictions_path = (
        input_path_override
        if input_path_override is not None
        else settings.explainability_input_predictions_path
    )
    if not input_predictions_path.is_absolute():
        input_predictions_path = settings.project_root / input_predictions_path

    resolved_sample_size = (
        int(sample_size) if sample_size is not None else settings.explainability_sample_size
    )
    resolved_top_k = int(top_k) if top_k is not None else settings.explainability_top_k
    resolved_seed = settings.explainability_random_seed

    if resolved_sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if resolved_top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    candidate_summary = _load_final_candidate_summary(settings)
    model_path = _resolve_model_artifact_path(settings, candidate_summary)

    check_paths = [
        paths.selected_examples_dir / SELECTED_EXAMPLES_FILE,
        paths.reports_dir / EXPLAINABILITY_SUMMARY_FILE,
    ]
    if METHOD_SHAP in methods:
        check_paths.extend(
            [
                paths.shap_global_dir / SHAP_GLOBAL_SUMMARY_FILE,
                paths.shap_global_dir / SHAP_FEATURE_IMPORTANCE_FILE,
                paths.shap_global_dir / SHAP_SUMMARY_PLOT_FILE,
                paths.shap_global_dir / SHAP_BAR_PLOT_FILE,
                paths.shap_local_dir / SHAP_LOCAL_EXPLANATIONS_FILE,
            ]
        )
    if METHOD_LIME in methods:
        check_paths.append(paths.lime_dir / LIME_LOCAL_EXPLANATIONS_FILE)
    _check_overwrite(check_paths, overwrite)

    LOGGER.info(
        "Starting explainability workflow: methods=%s sample_size=%d top_k=%d input=%s",
        methods,
        resolved_sample_size,
        resolved_top_k,
        input_predictions_path,
    )

    model = _load_model(model_path)
    selected_examples, selected_examples_path, dataset = _build_selected_examples(
        settings=settings,
        candidate_summary=candidate_summary,
        paths=paths,
        input_predictions_path=input_predictions_path,
    )

    model_metadata = {
        "model_family": str(candidate_summary["final_model_family"]),
        "candidate_name": str(candidate_summary["final_candidate_name"]),
        "model_artifact_path": str(model_path),
        "training_timestamp": candidate_summary.get("training_timestamp"),
    }

    shap_global_summary_path: Path | None = None
    shap_feature_importance_path: Path | None = None
    shap_summary_plot_path: Path | None = None
    shap_bar_plot_path: Path | None = None
    shap_local_explanations_path: Path | None = None
    lime_explanations_path: Path | None = None
    lime_generated_count = 0
    lime_failed_count = 0
    lime_total_count = 0

    if METHOD_SHAP in methods:
        shap_global_artifacts, _ = generate_shap_global_artifacts(
            model=model,
            x_frame=dataset.x_train,
            sample_size=resolved_sample_size,
            top_k=resolved_top_k,
            random_seed=resolved_seed,
            output_dir=paths.shap_global_dir,
            model_metadata=model_metadata,
        )
        shap_global_summary_path = shap_global_artifacts["shap_global_summary"]
        shap_feature_importance_path = shap_global_artifacts["shap_feature_importance"]
        shap_summary_plot_path = shap_global_artifacts["shap_summary_plot"]
        shap_bar_plot_path = shap_global_artifacts["shap_bar_plot"]

        if selected_examples.empty:
            shap_local_explanations_path = write_jsonl(
                paths.shap_local_dir / SHAP_LOCAL_EXPLANATIONS_FILE,
                [],
            )
        else:
            _, shap_local_explanations_path = generate_shap_local_artifacts(
                model=model,
                x_frame=dataset.x_train,
                selected_examples=selected_examples,
                top_k=resolved_top_k,
                threshold=float(candidate_summary["threshold"]),
                output_dir=paths.shap_local_dir,
                model_metadata=model_metadata,
            )

    if METHOD_LIME in methods:
        lime_payloads, lime_explanations_path, _ = generate_lime_local_artifacts(
            model=model,
            x_train=dataset.x_train,
            selected_examples=selected_examples,
            top_k=resolved_top_k,
            threshold=float(candidate_summary["threshold"]),
            random_seed=resolved_seed,
            output_dir=paths.lime_dir,
            model_metadata=model_metadata,
            categorical_columns=dataset.categorical_columns,
        )
        lime_total_count = len(lime_payloads)
        lime_generated_count = sum(
            1 for payload in lime_payloads if bool(payload.get("explanation_generated", True))
        )
        lime_failed_count = lime_total_count - lime_generated_count
        LOGGER.info(
            "LIME local explainability status: generated=%d failed=%d total=%d",
            lime_generated_count,
            lime_failed_count,
            lime_total_count,
        )

    summary_lines = [
        "## Configuration",
        f"- Methods: {', '.join(methods)}",
        f"- Sample size: {resolved_sample_size}",
        f"- Top-k features: {resolved_top_k}",
        f"- Selection counts: tp={settings.explainability_true_positive_examples}, "
        f"tn={settings.explainability_true_negative_examples}, "
        f"fp={settings.explainability_false_positive_examples}, "
        f"fn={settings.explainability_false_negative_examples}, "
        f"borderline={settings.explainability_borderline_examples}",
        "",
        "## Inputs",
        f"- Final candidate: {candidate_summary['final_candidate_name']}",
        f"- Model family: {candidate_summary['final_model_family']}",
        f"- Model artifact: {model_path}",
        f"- Threshold: {candidate_summary['threshold']}",
        f"- Prediction input path: {input_predictions_path}",
        "",
        "## Artifacts",
        f"- Selected examples: {selected_examples_path}",
    ]
    if shap_global_summary_path is not None:
        summary_lines.append(f"- SHAP global summary: {shap_global_summary_path}")
    if shap_feature_importance_path is not None:
        summary_lines.append(f"- SHAP feature importance: {shap_feature_importance_path}")
    if shap_summary_plot_path is not None:
        summary_lines.append(f"- SHAP summary plot: {shap_summary_plot_path}")
    if shap_bar_plot_path is not None:
        summary_lines.append(f"- SHAP bar plot: {shap_bar_plot_path}")
    if shap_local_explanations_path is not None:
        summary_lines.append(f"- SHAP local explanations: {shap_local_explanations_path}")
    if lime_explanations_path is not None:
        summary_lines.append(f"- LIME local explanations: {lime_explanations_path}")
        summary_lines.append(
            "- LIME status: "
            f"generated={lime_generated_count} failed={lime_failed_count} "
            f"total={lime_total_count}"
        )

    explainability_summary_path = write_markdown(
        paths.reports_dir / EXPLAINABILITY_SUMMARY_FILE,
        title="Explainability Summary",
        lines=summary_lines,
    )

    LOGGER.info(
        "Explainability workflow completed. selected_examples=%d summary=%s",
        len(selected_examples),
        explainability_summary_path,
    )

    return ExplainabilityWorkflowResult(
        selected_examples_path=selected_examples_path,
        shap_global_summary_path=shap_global_summary_path,
        shap_feature_importance_path=shap_feature_importance_path,
        shap_summary_plot_path=shap_summary_plot_path,
        shap_bar_plot_path=shap_bar_plot_path,
        shap_local_explanations_path=shap_local_explanations_path,
        lime_explanations_path=lime_explanations_path,
        explainability_summary_path=explainability_summary_path,
    )
