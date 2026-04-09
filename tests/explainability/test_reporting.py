"""Tests for explainability artifact reporting helpers."""

from __future__ import annotations

import json

import pandas as pd

from credit_risk_altdata.config import Settings
from credit_risk_altdata.explainability.reporting import (
    resolve_explainability_artifact_paths,
    write_dataframe_csv,
    write_json,
    write_jsonl,
    write_markdown,
)


def test_resolve_explainability_artifact_paths_creates_expected_directories(
    synthetic_settings: Settings,
) -> None:
    paths = resolve_explainability_artifact_paths(synthetic_settings)

    assert paths.root_dir.exists()
    assert paths.shap_dir.exists()
    assert paths.shap_global_dir.exists()
    assert paths.shap_local_dir.exists()
    assert paths.lime_dir.exists()
    assert paths.selected_examples_dir.exists()
    assert paths.reports_dir.exists()


def test_reporting_writers_emit_expected_formats(synthetic_settings: Settings) -> None:
    paths = resolve_explainability_artifact_paths(synthetic_settings)

    json_path = write_json(paths.reports_dir / "summary.json", {"a": 1, "b": "x"})
    assert json.loads(json_path.read_text(encoding="utf-8"))["a"] == 1

    jsonl_path = write_jsonl(
        paths.shap_local_dir / "local.jsonl",
        [{"applicant_id": 1}, {"applicant_id": 2}],
    )
    jsonl_lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == 2
    assert json.loads(jsonl_lines[0])["applicant_id"] == 1

    markdown_path = write_markdown(
        paths.reports_dir / "report.md",
        title="Explainability Summary",
        lines=["- line one", "- line two"],
    )
    markdown_content = markdown_path.read_text(encoding="utf-8")
    assert markdown_content.startswith("# Explainability Summary")

    dataframe_path = write_dataframe_csv(
        paths.selected_examples_dir / "selected_examples.csv",
        pd.DataFrame([{"applicant_id": 1, "cohort_name": "true_positive"}]),
    )
    dataframe = pd.read_csv(dataframe_path)
    assert dataframe.shape[0] == 1
    assert dataframe.loc[0, "cohort_name"] == "true_positive"
