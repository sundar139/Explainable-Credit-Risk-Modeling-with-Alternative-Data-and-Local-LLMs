"""CLI tests for Phase 7 generate-risk-reports command."""

from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from credit_risk_altdata import cli
from credit_risk_altdata.config import Settings
from credit_risk_altdata.llm.workflow import LLMReportingWorkflowResult


def test_cli_generate_risk_reports_success(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: synthetic_settings)

    captured: dict[str, object] = {}

    def fake_run_llm_reporting_workflow(
        settings: Settings,
        *,
        report_type_selection: str,
        method_source_selection: str,
        model_name_override: str | None,
        limit: int | None,
        input_path_override: Path | None,
        overwrite: bool,
    ) -> LLMReportingWorkflowResult:
        captured["settings"] = settings
        captured["report_type_selection"] = report_type_selection
        captured["method_source_selection"] = method_source_selection
        captured["model_name_override"] = model_name_override
        captured["limit"] = limit
        captured["input_path_override"] = input_path_override
        captured["overwrite"] = overwrite

        reports_jsonl = synthetic_settings.llm_reports_combined_dir / "llm_reports.jsonl"
        reports_csv = synthetic_settings.llm_reports_combined_dir / "llm_reports.csv"
        summary = synthetic_settings.llm_reports_reports_dir / "llm_reporting_summary.md"
        reports_jsonl.parent.mkdir(parents=True, exist_ok=True)
        summary.parent.mkdir(parents=True, exist_ok=True)
        reports_jsonl.write_text('{"report_id":"x"}\n', encoding="utf-8")
        reports_csv.write_text("report_id\nx\n", encoding="utf-8")
        summary.write_text("# LLM Reporting Summary\n", encoding="utf-8")

        return LLMReportingWorkflowResult(
            source_artifact_path=Path("artifacts/explainability/shap/local/shap_local_explanations.jsonl"),
            method_source_used="shap",
            reports_jsonl_path=reports_jsonl,
            reports_csv_path=reports_csv,
            reporting_summary_path=summary,
            total_reports=3,
            llm_generated_reports=2,
            fallback_generated_reports=1,
        )

    monkeypatch.setattr(cli, "run_llm_reporting_workflow", fake_run_llm_reporting_workflow)

    exit_code = cli.main(
        [
            "generate-risk-reports",
            "--report-type",
            "underwriter",
            "--method-source",
            "lime",
            "--model",
            "qwen2.5:7b",
            "--limit",
            "7",
            "--input-path",
            "artifacts/explainability/lime/lime_explanations.jsonl",
            "--overwrite",
        ]
    )

    assert exit_code == 0
    assert captured["settings"] is synthetic_settings
    assert captured["report_type_selection"] == "underwriter"
    assert captured["method_source_selection"] == "lime"
    assert captured["model_name_override"] == "qwen2.5:7b"
    assert captured["limit"] == 7
    assert captured["input_path_override"] == Path(
        "artifacts/explainability/lime/lime_explanations.jsonl"
    )
    assert captured["overwrite"] is True


def test_cli_generate_risk_reports_uses_settings_defaults(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: synthetic_settings)

    captured: dict[str, object] = {}

    def fake_run_llm_reporting_workflow(
        settings: Settings,
        *,
        report_type_selection: str,
        method_source_selection: str,
        model_name_override: str | None,
        limit: int | None,
        input_path_override: Path | None,
        overwrite: bool,
    ) -> LLMReportingWorkflowResult:
        captured["settings"] = settings
        captured["report_type_selection"] = report_type_selection
        captured["method_source_selection"] = method_source_selection
        captured["model_name_override"] = model_name_override
        captured["limit"] = limit
        captured["input_path_override"] = input_path_override
        captured["overwrite"] = overwrite

        path = synthetic_settings.llm_reports_reports_dir / "llm_reporting_summary.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# summary\n", encoding="utf-8")
        return LLMReportingWorkflowResult(
            source_artifact_path=Path("artifacts/explainability/shap/local/shap_local_explanations.jsonl"),
            method_source_used="shap",
            reports_jsonl_path=synthetic_settings.llm_reports_combined_dir / "llm_reports.jsonl",
            reports_csv_path=synthetic_settings.llm_reports_combined_dir / "llm_reports.csv",
            reporting_summary_path=path,
            total_reports=1,
            llm_generated_reports=1,
            fallback_generated_reports=0,
        )

    monkeypatch.setattr(cli, "run_llm_reporting_workflow", fake_run_llm_reporting_workflow)

    exit_code = cli.main(["generate-risk-reports", "--overwrite"])

    assert exit_code == 0
    assert captured["settings"] is synthetic_settings
    assert captured["report_type_selection"] == synthetic_settings.llm_reports_report_type
    assert captured["method_source_selection"] == synthetic_settings.llm_reports_method_source
    assert captured["model_name_override"] is None
    assert captured["limit"] is None
    assert captured["input_path_override"] is None
    assert captured["overwrite"] is True


def test_cli_generate_risk_reports_failure_returns_exit_code_1(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: synthetic_settings)

    def fake_run_llm_reporting_workflow(*_: object, **__: object) -> LLMReportingWorkflowResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_llm_reporting_workflow", fake_run_llm_reporting_workflow)

    exit_code = cli.main(["generate-risk-reports", "--overwrite"])

    assert exit_code == 1
