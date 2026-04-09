"""Reporting helpers for baseline modeling artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from credit_risk_altdata.config import Settings
from credit_risk_altdata.utils.filesystem import ensure_directories


@dataclass(frozen=True, slots=True)
class ModelingArtifactPaths:
    """Resolved artifact directories for baseline modeling outputs."""

    root_dir: Path
    metrics_dir: Path
    predictions_dir: Path
    feature_importance_dir: Path
    models_dir: Path
    tuned_models_dir: Path
    reports_dir: Path
    tuning_dir: Path
    calibration_dir: Path
    evaluation_dir: Path


def resolve_modeling_artifact_paths(settings: Settings) -> ModelingArtifactPaths:
    """Resolve and create artifact directories for modeling outputs."""
    paths = ModelingArtifactPaths(
        root_dir=settings.modeling_dir,
        metrics_dir=settings.modeling_metrics_dir,
        predictions_dir=settings.modeling_predictions_dir,
        feature_importance_dir=settings.modeling_feature_importance_dir,
        models_dir=settings.modeling_models_dir,
        tuned_models_dir=settings.modeling_models_dir / "tuned",
        reports_dir=settings.modeling_reports_dir,
        tuning_dir=settings.modeling_tuning_dir,
        calibration_dir=settings.modeling_calibration_dir,
        evaluation_dir=settings.modeling_evaluation_dir,
    )
    ensure_directories(
        [
            paths.root_dir,
            paths.metrics_dir,
            paths.predictions_dir,
            paths.feature_importance_dir,
            paths.models_dir,
            paths.tuned_models_dir,
            paths.reports_dir,
            paths.tuning_dir,
            paths.calibration_dir,
            paths.evaluation_dir,
        ]
    )
    return paths


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON artifact."""
    ensure_directories([path.parent])
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_markdown(path: Path, title: str, lines: list[str]) -> Path:
    """Write a markdown artifact."""
    ensure_directories([path.parent])
    markdown = [f"# {title}", ""] + lines + [""]
    path.write_text("\n".join(markdown), encoding="utf-8")
    return path
