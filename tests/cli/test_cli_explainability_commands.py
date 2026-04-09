"""CLI tests for Phase 6 generate-explanations command."""

from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from credit_risk_altdata import cli
from credit_risk_altdata.config import Settings
from credit_risk_altdata.explainability.workflow import ExplainabilityWorkflowResult


def test_cli_generate_explanations_success(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: synthetic_settings)

    captured: dict[str, object] = {}

    def fake_run_explainability_workflow(
        settings: Settings,
        *,
        method_selection: str,
        sample_size: int | None,
        top_k: int | None,
        input_path_override: Path | None,
        overwrite: bool,
    ) -> ExplainabilityWorkflowResult:
        captured["settings"] = settings
        captured["method_selection"] = method_selection
        captured["sample_size"] = sample_size
        captured["top_k"] = top_k
        captured["input_path_override"] = input_path_override
        captured["overwrite"] = overwrite

        summary_path = synthetic_settings.explainability_reports_dir / "explainability_summary.md"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("# Explainability Summary\n", encoding="utf-8")

        selected_examples_path = (
            synthetic_settings.explainability_selected_examples_dir / "selected_examples.csv"
        )
        selected_examples_path.parent.mkdir(parents=True, exist_ok=True)
        selected_examples_path.write_text("applicant_id\n123\n", encoding="utf-8")

        return ExplainabilityWorkflowResult(
            selected_examples_path=selected_examples_path,
            shap_global_summary_path=None,
            shap_feature_importance_path=None,
            shap_summary_plot_path=None,
            shap_bar_plot_path=None,
            shap_local_explanations_path=None,
            lime_explanations_path=None,
            explainability_summary_path=summary_path,
        )

    monkeypatch.setattr(cli, "run_explainability_workflow", fake_run_explainability_workflow)

    exit_code = cli.main(
        [
            "generate-explanations",
            "--method",
            "shap",
            "--sample-size",
            "128",
            "--top-k",
            "7",
            "--input-path",
            "artifacts/modeling/predictions/custom_oof.parquet",
            "--overwrite",
        ]
    )

    assert exit_code == 0
    assert captured["settings"] is synthetic_settings
    assert captured["method_selection"] == "shap"
    assert captured["sample_size"] == 128
    assert captured["top_k"] == 7
    assert captured["input_path_override"] == Path(
        "artifacts/modeling/predictions/custom_oof.parquet"
    )
    assert captured["overwrite"] is True


def test_cli_generate_explanations_failure_returns_exit_code_1(
    synthetic_settings: Settings,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: synthetic_settings)

    def fake_run_explainability_workflow(*_: object, **__: object) -> ExplainabilityWorkflowResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_explainability_workflow", fake_run_explainability_workflow)

    exit_code = cli.main(["generate-explanations", "--method", "all"])

    assert exit_code == 1
