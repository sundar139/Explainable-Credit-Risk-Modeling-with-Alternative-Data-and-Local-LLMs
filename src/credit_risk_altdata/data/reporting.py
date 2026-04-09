"""Reporting helpers for data validation and data pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk_altdata.utils.filesystem import ensure_directories


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def ensure_report_directory(directory: Path) -> Path:
    """Create and return the report directory."""
    ensure_directories([directory])
    return directory


def write_json_report(path: Path, payload: dict[str, Any]) -> Path:
    """Write a dictionary payload as formatted JSON."""
    ensure_directories([path.parent])
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def write_csv_report(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write list-of-dict records into CSV format."""
    ensure_directories([path.parent])
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(path, index=False)
    return path


def write_markdown_report(path: Path, title: str, lines: list[str]) -> Path:
    """Write a simple markdown report for quick human review."""
    ensure_directories([path.parent])
    markdown_lines = [f"# {title}", ""]
    markdown_lines.extend(lines)
    markdown_lines.append("")
    path.write_text("\n".join(markdown_lines), encoding="utf-8")
    return path
