"""Artifact path resolution and file-writing helpers for explainability."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pandas import DataFrame

from credit_risk_altdata.config import Settings
from credit_risk_altdata.utils.filesystem import ensure_directories


@dataclass(frozen=True, slots=True)
class ExplainabilityArtifactPaths:
    """Resolved explainability artifact directories."""

    root_dir: Path
    shap_dir: Path
    shap_global_dir: Path
    shap_local_dir: Path
    lime_dir: Path
    selected_examples_dir: Path
    reports_dir: Path


def resolve_explainability_artifact_paths(settings: Settings) -> ExplainabilityArtifactPaths:
    """Resolve and create directories for explainability outputs."""
    paths = ExplainabilityArtifactPaths(
        root_dir=settings.explainability_root_dir,
        shap_dir=settings.explainability_shap_dir,
        shap_global_dir=settings.explainability_shap_global_dir,
        shap_local_dir=settings.explainability_shap_local_dir,
        lime_dir=settings.explainability_lime_dir,
        selected_examples_dir=settings.explainability_selected_examples_dir,
        reports_dir=settings.explainability_reports_dir,
    )
    ensure_directories(
        [
            paths.root_dir,
            paths.shap_dir,
            paths.shap_global_dir,
            paths.shap_local_dir,
            paths.lime_dir,
            paths.selected_examples_dir,
            paths.reports_dir,
        ]
    )
    return paths


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write formatted JSON payload."""
    ensure_directories([path.parent])
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write rows in JSONL format."""
    ensure_directories([path.parent])
    lines = [json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def write_markdown(path: Path, title: str, lines: list[str]) -> Path:
    """Write markdown report."""
    ensure_directories([path.parent])
    markdown = [f"# {title}", ""] + lines + [""]
    path.write_text("\n".join(markdown), encoding="utf-8")
    return path


def write_dataframe_csv(path: Path, dataframe: DataFrame) -> Path:
    """Write dataframe to CSV."""
    ensure_directories([path.parent])
    dataframe.to_csv(path, index=False)
    return path
