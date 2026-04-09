"""Artifact path and writing helpers for LLM risk reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk_altdata.config import Settings
from credit_risk_altdata.utils.filesystem import ensure_directories


@dataclass(frozen=True, slots=True)
class LLMReportArtifactPaths:
    """Resolved directory layout for Phase 7 report artifacts."""

    root_dir: Path
    plain_language_dir: Path
    underwriter_dir: Path
    adverse_action_dir: Path
    combined_dir: Path
    reports_dir: Path


def resolve_llm_report_artifact_paths(settings: Settings) -> LLMReportArtifactPaths:
    """Resolve and create directories for LLM report outputs."""
    paths = LLMReportArtifactPaths(
        root_dir=settings.llm_reports_root_dir,
        plain_language_dir=settings.llm_reports_plain_language_dir,
        underwriter_dir=settings.llm_reports_underwriter_dir,
        adverse_action_dir=settings.llm_reports_adverse_action_dir,
        combined_dir=settings.llm_reports_combined_dir,
        reports_dir=settings.llm_reports_reports_dir,
    )
    ensure_directories(
        [
            paths.root_dir,
            paths.plain_language_dir,
            paths.underwriter_dir,
            paths.adverse_action_dir,
            paths.combined_dir,
            paths.reports_dir,
        ]
    )
    return paths


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write JSONL rows with UTF-8 encoding."""
    ensure_directories([path.parent])
    lines = [json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write rows to CSV while preserving nested fields as JSON strings."""
    ensure_directories([path.parent])
    serialized_rows: list[dict[str, Any]] = []
    for row in rows:
        serialized: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                serialized[key] = json.dumps(value)
            else:
                serialized[key] = value
        serialized_rows.append(serialized)

    pd.DataFrame(serialized_rows).to_csv(path, index=False)
    return path


def write_markdown(path: Path, title: str, lines: list[str]) -> Path:
    """Write markdown text report."""
    ensure_directories([path.parent])
    content = [f"# {title}", ""] + lines + [""]
    path.write_text("\n".join(content), encoding="utf-8")
    return path
