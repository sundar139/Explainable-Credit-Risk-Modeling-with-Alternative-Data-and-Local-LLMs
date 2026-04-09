"""Tests for Phase 7 local LLM reporting workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from credit_risk_altdata.config import Settings
from credit_risk_altdata.llm import workflow as llm_workflow
from credit_risk_altdata.llm.ollama_client import OllamaHealthcheckResult


def _write_explanation_payloads(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _base_payload_row(applicant_id: int) -> dict[str, Any]:
    return {
        "explanation_method": "shap",
        "applicant_id": applicant_id,
        "cohort_name": "false_positive",
        "predicted_probability": 0.74,
        "predicted_label": 1,
        "actual_label": 0,
        "threshold": 0.5,
        "top_risk_increasing_features": [
            {"feature_name": "feature_a", "contribution": 0.22},
            {"feature_name": "feature_b", "contribution": 0.08},
        ],
        "top_risk_decreasing_features": [
            {"feature_name": "feature_c", "contribution": -0.11},
        ],
        "explanation_generated": True,
        "failure_reason": None,
    }


def test_run_llm_reporting_workflow_success_with_mocked_ollama(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    source_path = synthetic_settings.explainability_shap_local_dir / "shap_local_explanations.jsonl"
    _write_explanation_payloads(
        source_path,
        [_base_payload_row(700001), _base_payload_row(700002)],
    )

    class _FakeClient:
        def __init__(self, **_: Any) -> None:
            pass

        def healthcheck(self, *, required_model: str | None = None) -> OllamaHealthcheckResult:
            del required_model
            return OllamaHealthcheckResult(
                reachable=True,
                available_models=("qwen2.5:7b",),
                failure_reason=None,
            )

        def generate(self, *, prompt: str, model: str | None = None, **_: Any) -> str:
            assert "Guardrails" in prompt
            assert model == "qwen2.5:7b"
            return "Model-based narrative output."

    monkeypatch.setattr(llm_workflow, "OllamaClient", _FakeClient)

    result = llm_workflow.run_llm_reporting_workflow(
        synthetic_settings,
        report_type_selection="plain",
        method_source_selection="shap",
        model_name_override="qwen2.5:7b",
        limit=2,
        overwrite=True,
    )

    assert result.total_reports == 2
    assert result.llm_generated_reports == 2
    assert result.fallback_generated_reports == 0
    assert result.reports_jsonl_path.exists()
    assert result.reports_csv_path.exists()
    assert result.reporting_summary_path.exists()

    rows = [
        json.loads(line)
        for line in result.reports_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert all(row["fallback_generated"] is False for row in rows)
    required_fields = {
        "report_id",
        "applicant_id",
        "report_type",
        "explanation_method_source",
        "predicted_probability",
        "threshold",
        "supporting_features",
        "generated_text",
        "fallback_generated",
        "failure_reason",
        "model_name",
        "prompt_version",
        "artifact_version",
        "generation_timestamp",
        "disclaimer",
    }
    assert required_fields.issubset(rows[0].keys())


def test_run_llm_reporting_workflow_uses_fallback_when_ollama_unavailable(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    source_path = synthetic_settings.explainability_shap_local_dir / "shap_local_explanations.jsonl"
    _write_explanation_payloads(source_path, [_base_payload_row(700100)])

    class _UnavailableClient:
        def __init__(self, **_: Any) -> None:
            pass

        def healthcheck(self, *, required_model: str | None = None) -> OllamaHealthcheckResult:
            del required_model
            return OllamaHealthcheckResult(
                reachable=False,
                available_models=tuple(),
                failure_reason="connection refused",
            )

        def generate(self, *, prompt: str, model: str | None = None, **_: Any) -> str:
            raise AssertionError("generate should not be called when healthcheck is unreachable")

    monkeypatch.setattr(llm_workflow, "OllamaClient", _UnavailableClient)

    result = llm_workflow.run_llm_reporting_workflow(
        synthetic_settings,
        report_type_selection="all",
        method_source_selection="shap",
        overwrite=True,
    )

    assert result.total_reports == 3
    assert result.llm_generated_reports == 0
    assert result.fallback_generated_reports == 3

    rows = [
        json.loads(line)
        for line in result.reports_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(row["fallback_generated"] is True for row in rows)
    assert all("Ollama unavailable" in str(row["failure_reason"]) for row in rows)


def test_run_llm_reporting_workflow_partial_fallback_when_single_generation_fails(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    source_path = synthetic_settings.explainability_shap_local_dir / "shap_local_explanations.jsonl"
    _write_explanation_payloads(
        source_path,
        [_base_payload_row(700201), _base_payload_row(700202)],
    )

    class _PartiallyFailingClient:
        def __init__(self, **_: Any) -> None:
            self.call_count = 0

        def healthcheck(self, *, required_model: str | None = None) -> OllamaHealthcheckResult:
            del required_model
            return OllamaHealthcheckResult(
                reachable=True,
                available_models=("qwen2.5:7b",),
                failure_reason=None,
            )

        def generate(self, *, prompt: str, model: str | None = None, **_: Any) -> str:
            del prompt
            del model
            self.call_count += 1
            if self.call_count == 2:
                raise RuntimeError("synthetic generation failure")
            return "LLM narrative"

    monkeypatch.setattr(llm_workflow, "OllamaClient", _PartiallyFailingClient)

    result = llm_workflow.run_llm_reporting_workflow(
        synthetic_settings,
        report_type_selection="plain",
        method_source_selection="shap",
        overwrite=True,
    )

    assert result.total_reports == 2
    assert result.llm_generated_reports == 1
    assert result.fallback_generated_reports == 1

    rows = [
        json.loads(line)
        for line in result.reports_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fallback_flags = [bool(row["fallback_generated"]) for row in rows]
    assert fallback_flags.count(True) == 1
    assert fallback_flags.count(False) == 1


def test_run_llm_reporting_workflow_invalid_input_raises(
    synthetic_settings: Settings,
) -> None:
    source_path = synthetic_settings.explainability_shap_local_dir / "shap_local_explanations.jsonl"
    _write_explanation_payloads(
        source_path,
        [
            {
                "explanation_method": "shap",
                "applicant_id": 700301,
                "threshold": 0.5,
            }
        ],
    )

    with pytest.raises(ValueError):
        llm_workflow.run_llm_reporting_workflow(
            synthetic_settings,
            report_type_selection="plain",
            method_source_selection="shap",
            overwrite=True,
        )
