"""Command-line interface for project operations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import requests
from requests import RequestException

from credit_risk_altdata.audit import verify_artifact_contracts
from credit_risk_altdata.config import Settings, get_settings
from credit_risk_altdata.data.download import DataDownloadError, download_home_credit_dataset
from credit_risk_altdata.data.loaders import build_interim_parquet
from credit_risk_altdata.data.raw_validation import validate_raw_data
from credit_risk_altdata.explainability.constants import (
    METHOD_ALL,
    METHOD_LIME,
    METHOD_SHAP,
    ExplainabilityMethodSelection,
)
from credit_risk_altdata.explainability.workflow import run_explainability_workflow
from credit_risk_altdata.features.pipeline import FeaturePipelineError, build_feature_matrices
from credit_risk_altdata.llm.constants import (
    METHOD_SOURCE_AUTO,
    METHOD_SOURCE_LIME,
    METHOD_SOURCE_SHAP,
    REPORT_TYPE_ADVERSE_ACTION,
    REPORT_TYPE_ALL,
    REPORT_TYPE_PLAIN,
    REPORT_TYPE_UNDERWRITER,
    ExplanationMethodSourceSelection,
    ReportTypeSelection,
)
from credit_risk_altdata.llm.workflow import run_llm_reporting_workflow
from credit_risk_altdata.logging import configure_logging, get_logger
from credit_risk_altdata.modeling.constants import (
    CALIBRATION_ALL,
    CALIBRATION_ISOTONIC,
    CALIBRATION_NONE,
    CALIBRATION_SIGMOID,
    MODEL_ALL,
    MODEL_CATBOOST,
    MODEL_LIGHTGBM,
    CalibrationSelection,
    ModelSelection,
)
from credit_risk_altdata.modeling.training import run_baseline_training
from credit_risk_altdata.modeling.tuning import run_tuned_modeling
from credit_risk_altdata.utils.filesystem import ensure_directories

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser and subcommands."""
    parser = argparse.ArgumentParser(
        prog="credit-risk",
        description="Utilities for the credit risk alternative data project",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "show-config",
        help="Print validated runtime configuration as JSON",
    )

    subparsers.add_parser(
        "prepare-dirs",
        help="Create expected data and artifact directories",
    )

    verify_parser = subparsers.add_parser(
        "verify-artifacts",
        help="Validate core artifact contracts required for demo/API readiness",
    )
    verify_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable verification results",
    )

    healthcheck_parser = subparsers.add_parser(
        "healthcheck",
        help="Check that Ollama is reachable",
    )
    healthcheck_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Request timeout in seconds; defaults to OLLAMA_TIMEOUT_SECONDS",
    )

    download_parser = subparsers.add_parser(
        "download-data",
        help="Download and extract Home Credit raw files from Kaggle",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even when required files already exist",
    )

    validate_parser = subparsers.add_parser(
        "validate-raw-data",
        help="Run raw data validation and generate quality reports",
    )
    validate_parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Optional custom report output directory",
    )

    parquet_parser = subparsers.add_parser(
        "build-interim-parquet",
        help="Convert core raw CSV files into parquet under data/interim/home_credit",
    )
    parquet_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing parquet files",
    )

    features_parser = subparsers.add_parser(
        "build-features",
        help="Build model-ready train/test feature matrices from Home Credit tables",
    )
    features_parser.add_argument(
        "--input-source",
        choices=["raw", "interim"],
        default="raw",
        help="Input source tables: raw CSV or interim parquet",
    )
    features_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed feature outputs",
    )

    baseline_parser = subparsers.add_parser(
        "train-baselines",
        help="Train baseline LightGBM and CatBoost models with stratified CV",
    )
    baseline_parser.add_argument(
        "--model",
        choices=[MODEL_ALL, MODEL_LIGHTGBM, MODEL_CATBOOST],
        default=MODEL_ALL,
        help="Baseline model family selection",
    )
    baseline_parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional train feature parquet path or directory containing train/test feature files",
    )
    baseline_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing modeling artifacts",
    )

    tuning_parser = subparsers.add_parser(
        "tune-models",
        help="Run Optuna tuning, calibration comparison, and rich evaluation reporting",
    )
    tuning_parser.add_argument(
        "--model",
        choices=[MODEL_ALL, MODEL_LIGHTGBM, MODEL_CATBOOST],
        default=MODEL_ALL,
        help="Model family selection for tuning",
    )
    tuning_parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna trials per selected model family",
    )
    tuning_parser.add_argument(
        "--calibration",
        choices=[CALIBRATION_NONE, CALIBRATION_SIGMOID, CALIBRATION_ISOTONIC, CALIBRATION_ALL],
        default=None,
        help="Calibration strategy selection",
    )
    tuning_parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional train feature parquet path or directory containing train/test feature files",
    )
    tuning_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tuning/calibration/evaluation artifacts",
    )

    explainability_parser = subparsers.add_parser(
        "generate-explanations",
        help="Generate Phase 6 SHAP/LIME explainability artifacts",
    )
    explainability_parser.add_argument(
        "--method",
        choices=[METHOD_ALL, METHOD_SHAP, METHOD_LIME],
        default=METHOD_ALL,
        help="Explanation method selection",
    )
    explainability_parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional SHAP global sampling size override",
    )
    explainability_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k feature contributions override",
    )
    explainability_parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional OOF prediction parquet override",
    )
    explainability_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing explainability artifacts",
    )

    llm_reports_parser = subparsers.add_parser(
        "generate-risk-reports",
        help="Generate Phase 7 local narrative reports from explainability payloads",
    )
    llm_reports_parser.add_argument(
        "--report-type",
        choices=[
            REPORT_TYPE_ALL,
            REPORT_TYPE_PLAIN,
            REPORT_TYPE_UNDERWRITER,
            REPORT_TYPE_ADVERSE_ACTION,
        ],
        default=None,
        help="Narrative report style selection",
    )
    llm_reports_parser.add_argument(
        "--method-source",
        choices=[METHOD_SOURCE_AUTO, METHOD_SOURCE_SHAP, METHOD_SOURCE_LIME],
        default=None,
        help="Explanation payload source selection",
    )
    llm_reports_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional Ollama model override",
    )
    llm_reports_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of explanation cases to narrate",
    )
    llm_reports_parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional explanation payload JSONL override",
    )
    llm_reports_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing LLM reporting artifacts",
    )

    return parser


def command_show_config(settings: Settings) -> int:
    """Print non-sensitive configuration values."""
    print(json.dumps(settings.safe_dump(), indent=2))
    return 0


def command_prepare_dirs(settings: Settings) -> int:
    """Create standard folder structure for data and artifacts."""
    created = ensure_directories(
        [
            settings.raw_data_dir,
            settings.interim_data_dir,
            settings.processed_data_dir,
            settings.resolved_artifacts_dir,
            settings.home_credit_raw_dir,
            settings.home_credit_interim_dir,
            settings.home_credit_processed_dir,
            settings.data_validation_dir,
            settings.feature_metadata_dir,
            settings.modeling_dir,
            settings.modeling_metrics_dir,
            settings.modeling_predictions_dir,
            settings.modeling_feature_importance_dir,
            settings.modeling_models_dir,
            settings.modeling_reports_dir,
            settings.modeling_tuning_dir,
            settings.modeling_calibration_dir,
            settings.modeling_evaluation_dir,
            settings.modeling_final_model_output_path.parent,
            settings.explainability_dir,
            settings.explainability_selected_examples_dir,
            settings.explainability_reports_dir,
            settings.explainability_shap_global_dir,
            settings.explainability_shap_local_dir,
            settings.explainability_lime_dir,
            settings.llm_reports_root_dir,
            settings.llm_reports_plain_language_dir,
            settings.llm_reports_underwriter_dir,
            settings.llm_reports_adverse_action_dir,
            settings.llm_reports_combined_dir,
            settings.llm_reports_reports_dir,
        ]
    )
    LOGGER.info("Validated %d project directories", len(created))
    return 0


def command_verify_artifacts(settings: Settings, *, output_json: bool) -> int:
    """Validate canonical artifact contracts across Phases 3-8."""
    report = verify_artifact_contracts(settings)

    if output_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        for check in report.checks:
            status = "PASS" if check.passed else ("FAIL" if check.required else "WARN")
            path_suffix = f" [{check.path}]" if check.path is not None else ""
            LOGGER.info("%s %-42s%s :: %s", status, check.name, path_suffix, check.message)

        if report.is_valid:
            LOGGER.info("Artifact verification passed with %d checks", len(report.checks))
        else:
            LOGGER.error(
                "Artifact verification failed. required_failures=%d optional_failures=%d",
                report.required_failed_count,
                report.optional_failed_count,
            )

    return 0 if report.is_valid else 1


def command_healthcheck(settings: Settings, timeout: float | None) -> int:
    """Verify local Ollama endpoint availability."""
    resolved_timeout = timeout if timeout is not None else float(settings.ollama_timeout_seconds)
    url = f"{str(settings.ollama_base_url).rstrip('/')}/api/tags"

    try:
        response = requests.get(url, timeout=resolved_timeout)
        response.raise_for_status()
    except RequestException as exc:
        LOGGER.error("Ollama healthcheck failed for %s: %s", url, exc)
        return 1

    LOGGER.info("Ollama reachable at %s", url)
    return 0


def command_download_data(settings: Settings, force: bool) -> int:
    """Download and extract Home Credit raw data from Kaggle."""
    try:
        result = download_home_credit_dataset(settings=settings, force=force)
    except DataDownloadError as exc:
        LOGGER.error("Dataset download failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.error("Unexpected error while downloading data: %s", exc)
        return 1

    if result.skipped:
        LOGGER.info("Raw files already exist in %s. Use --force to redownload.", result.destination)
        return 0

    LOGGER.info(
        "Downloaded and extracted dataset into %s (%d files extracted)",
        result.destination,
        len(result.extracted_files),
    )
    return 0


def command_validate_raw_data(settings: Settings, report_dir: Path | None) -> int:
    """Run raw data validation and report generation."""
    try:
        result = validate_raw_data(settings=settings, report_dir=report_dir)
    except Exception as exc:
        LOGGER.error("Raw data validation failed: %s", exc)
        return 1

    LOGGER.info("Validation reports written to %s", result.report_paths["data_quality_json"].parent)
    if not result.is_valid:
        LOGGER.error("Raw data validation found %d error(s)", len(result.errors))
        for message in result.errors:
            LOGGER.error("%s", message)
        return 1

    LOGGER.info("Raw data validation passed with %d warning(s)", len(result.warnings))
    return 0


def command_build_interim_parquet(settings: Settings, force: bool) -> int:
    """Build parquet copies from Home Credit raw CSV files."""
    try:
        records = build_interim_parquet(settings=settings, force=force)
    except Exception as exc:
        LOGGER.error("Interim parquet build failed: %s", exc)
        return 1

    written_count = sum(1 for record in records if record.written)
    skipped_count = len(records) - written_count
    LOGGER.info(
        "Interim parquet build complete. written=%d skipped=%d output_dir=%s",
        written_count,
        skipped_count,
        settings.home_credit_interim_dir,
    )
    return 0


def command_build_features(
    settings: Settings,
    *,
    input_source: Literal["raw", "interim"],
    overwrite: bool,
) -> int:
    """Build processed train/test feature matrices and metadata artifacts."""
    LOGGER.info("Starting feature build pipeline using input source: %s", input_source)
    try:
        result = build_feature_matrices(
            settings=settings,
            input_source=input_source,
            overwrite=overwrite,
        )
    except (FeaturePipelineError, FileNotFoundError, FileExistsError, ValueError) as exc:
        LOGGER.error("Feature build failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.error("Unexpected error while building features: %s", exc)
        return 1

    LOGGER.info(
        "Feature build complete. train=%s test=%s manifest=%s",
        result.train_output_path,
        result.test_output_path,
        result.manifest_path,
    )
    return 0


def command_train_baselines(
    settings: Settings,
    *,
    model_selection: ModelSelection,
    input_path: Path | None,
    overwrite: bool,
) -> int:
    """Train baseline models and save evaluation artifacts."""
    LOGGER.info("Starting baseline training with model selection: %s", model_selection)
    try:
        result = run_baseline_training(
            settings=settings,
            model_selection=model_selection,
            input_path_override=input_path,
            overwrite=overwrite,
        )
    except Exception as exc:
        LOGGER.error("Baseline training failed: %s", exc)
        return 1

    LOGGER.info(
        "Baseline training complete. best_model=%s comparison=%s",
        result.best_model_name,
        result.model_comparison_path,
    )
    return 0


def command_tune_models(
    settings: Settings,
    *,
    model_selection: ModelSelection,
    n_trials: int | None,
    calibration_selection: CalibrationSelection | None,
    input_path: Path | None,
    overwrite: bool,
) -> int:
    """Run tuning, calibration comparison, and Phase 5 evaluation outputs."""
    LOGGER.info(
        "Starting tune-models with model selection=%s n_trials=%s calibration=%s",
        model_selection,
        n_trials,
        calibration_selection,
    )
    try:
        result = run_tuned_modeling(
            settings,
            model_selection=model_selection,
            n_trials=n_trials,
            calibration_selection=calibration_selection,
            input_path_override=input_path,
            overwrite=overwrite,
        )
    except Exception as exc:
        LOGGER.error("Tuned modeling failed: %s", exc)
        return 1

    LOGGER.info(
        "Tuned modeling complete. final_candidate=%s comparison=%s",
        result.final_model_name,
        result.tuned_model_comparison_path,
    )
    return 0


def command_generate_explanations(
    settings: Settings,
    *,
    method_selection: ExplainabilityMethodSelection,
    sample_size: int | None,
    top_k: int | None,
    input_path: Path | None,
    overwrite: bool,
) -> int:
    """Generate explainability artifacts from final tuned model outputs."""
    LOGGER.info(
        "Starting generate-explanations with method=%s sample_size=%s top_k=%s",
        method_selection,
        sample_size,
        top_k,
    )
    try:
        result = run_explainability_workflow(
            settings,
            method_selection=method_selection,
            sample_size=sample_size,
            top_k=top_k,
            input_path_override=input_path,
            overwrite=overwrite,
        )
    except Exception as exc:
        LOGGER.error("Explainability generation failed: %s", exc)
        return 1

    LOGGER.info(
        "Explainability generation complete. selected_examples=%s summary=%s",
        result.selected_examples_path,
        result.explainability_summary_path,
    )
    return 0


def command_generate_risk_reports(
    settings: Settings,
    *,
    report_type_selection: ReportTypeSelection | None,
    method_source_selection: ExplanationMethodSourceSelection | None,
    model_name: str | None,
    limit: int | None,
    input_path: Path | None,
    overwrite: bool,
) -> int:
    """Generate narrative risk reports from explainability payload artifacts."""
    resolved_report_type = (
        report_type_selection
        if report_type_selection is not None
        else settings.llm_reports_report_type
    )
    resolved_method_source = (
        method_source_selection
        if method_source_selection is not None
        else settings.llm_reports_method_source
    )

    LOGGER.info(
        "Starting generate-risk-reports with report_type=%s method_source=%s limit=%s model=%s",
        resolved_report_type,
        resolved_method_source,
        limit,
        model_name if model_name is not None else settings.llm_reports_model_name,
    )
    try:
        result = run_llm_reporting_workflow(
            settings,
            report_type_selection=resolved_report_type,
            method_source_selection=resolved_method_source,
            model_name_override=model_name,
            limit=limit,
            input_path_override=input_path,
            overwrite=overwrite,
        )
    except Exception as exc:
        LOGGER.error("Risk report generation failed: %s", exc)
        return 1

    LOGGER.info(
        "Risk report generation complete. total_reports=%d llm_generated=%d fallback=%d summary=%s",
        result.total_reports,
        result.llm_generated_reports,
        result.fallback_generated_reports,
        result.reporting_summary_path,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    configure_logging(level="INFO")
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
    except Exception as exc:
        LOGGER.error("Configuration validation failed: %s", exc)
        return 2

    configure_logging(level=settings.log_level)

    if args.command == "show-config":
        return command_show_config(settings)
    if args.command == "prepare-dirs":
        return command_prepare_dirs(settings)
    if args.command == "verify-artifacts":
        return command_verify_artifacts(settings=settings, output_json=args.json)
    if args.command == "healthcheck":
        return command_healthcheck(settings=settings, timeout=args.timeout)
    if args.command == "download-data":
        return command_download_data(settings=settings, force=args.force)
    if args.command == "validate-raw-data":
        return command_validate_raw_data(settings=settings, report_dir=args.report_dir)
    if args.command == "build-interim-parquet":
        return command_build_interim_parquet(settings=settings, force=args.force)
    if args.command == "build-features":
        return command_build_features(
            settings=settings,
            input_source=cast(Literal["raw", "interim"], args.input_source),
            overwrite=args.overwrite,
        )
    if args.command == "train-baselines":
        return command_train_baselines(
            settings=settings,
            model_selection=cast(ModelSelection, args.model),
            input_path=args.input_path,
            overwrite=args.overwrite,
        )
    if args.command == "tune-models":
        return command_tune_models(
            settings=settings,
            model_selection=cast(ModelSelection, args.model),
            n_trials=args.n_trials,
            calibration_selection=cast(CalibrationSelection | None, args.calibration),
            input_path=args.input_path,
            overwrite=args.overwrite,
        )
    if args.command == "generate-explanations":
        return command_generate_explanations(
            settings=settings,
            method_selection=cast(ExplainabilityMethodSelection, args.method),
            sample_size=args.sample_size,
            top_k=args.top_k,
            input_path=args.input_path,
            overwrite=args.overwrite,
        )
    if args.command == "generate-risk-reports":
        return command_generate_risk_reports(
            settings=settings,
            report_type_selection=cast(ReportTypeSelection | None, args.report_type),
            method_source_selection=cast(
                ExplanationMethodSourceSelection | None,
                args.method_source,
            ),
            model_name=args.model,
            limit=args.limit,
            input_path=args.input_path,
            overwrite=args.overwrite,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2
