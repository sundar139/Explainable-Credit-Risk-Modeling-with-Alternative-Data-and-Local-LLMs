"""Smoke tests for package import and config loading."""

from __future__ import annotations

from pytest import MonkeyPatch

from credit_risk_altdata import __version__
from credit_risk_altdata.api.app import create_app
from credit_risk_altdata.cli import build_parser
from credit_risk_altdata.config import get_settings, reset_settings_cache


def test_package_import() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_config_loading_defaults(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    reset_settings_cache()
    settings = get_settings()

    assert settings.app_env == "test"
    assert str(settings.ollama_base_url).startswith("http://127.0.0.1:11434")


def test_cli_parser_and_api_import_smoke(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "test")

    reset_settings_cache()
    settings = get_settings()

    parser = build_parser()
    app = create_app(settings)

    assert parser.prog == "credit-risk"
    assert app.title == settings.app_name
