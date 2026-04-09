"""Unit tests for local Ollama client behavior."""

from __future__ import annotations

from typing import Any

import pytest
from pytest import MonkeyPatch
from requests import RequestException

from credit_risk_altdata.llm.ollama_client import OllamaClient, OllamaClientError


class _DummyResponse:
    def __init__(
        self,
        *,
        status_code: int,
        payload: Any | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_ollama_client_success_paths(monkeypatch: MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_request(
        *,
        method: str,
        url: str,
        json: dict[str, Any] | None,
        timeout: float,
    ) -> _DummyResponse:
        del timeout
        calls.append((method, url))
        if url.endswith("/api/tags"):
            return _DummyResponse(
                status_code=200,
                payload={"models": [{"name": "qwen2.5:7b"}, {"name": "qwen2.5-coder:7b"}]},
            )
        if url.endswith("/api/generate"):
            assert json is not None
            assert json["model"] == "qwen2.5:7b"
            return _DummyResponse(status_code=200, payload={"response": "  model output  "})
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("requests.request", fake_request)

    client = OllamaClient(
        base_url="http://127.0.0.1:11434",
        model="qwen2.5:7b",
        timeout_seconds=10,
        max_retries=1,
    )

    health = client.healthcheck(required_model="qwen2.5:7b")
    assert health.reachable is True
    assert "qwen2.5:7b" in health.available_models
    assert health.failure_reason is None

    generated = client.generate(prompt="Summarize the case")
    assert generated == "model output"
    assert calls[0][1].endswith("/api/tags")
    assert calls[1][1].endswith("/api/generate")


def test_ollama_client_failure_paths(monkeypatch: MonkeyPatch) -> None:
    def failing_request(
        *,
        method: str,
        url: str,
        json: dict[str, Any] | None,
        timeout: float,
    ) -> _DummyResponse:
        del method
        del url
        del json
        del timeout
        raise RequestException("connection refused")

    monkeypatch.setattr("requests.request", failing_request)

    client = OllamaClient(
        base_url="http://127.0.0.1:11434",
        model="qwen2.5:7b",
        timeout_seconds=5,
        max_retries=0,
    )

    health = client.healthcheck(required_model="qwen2.5:7b")
    assert health.reachable is False
    assert health.failure_reason is not None

    with pytest.raises(OllamaClientError):
        client.generate(prompt="Generate text")


def test_ollama_client_healthcheck_model_missing(monkeypatch: MonkeyPatch) -> None:
    def fake_request(
        *,
        method: str,
        url: str,
        json: dict[str, Any] | None,
        timeout: float,
    ) -> _DummyResponse:
        del method
        del json
        del timeout
        assert url.endswith("/api/tags")
        return _DummyResponse(status_code=200, payload={"models": [{"name": "qwen2.5-coder:7b"}]})

    monkeypatch.setattr("requests.request", fake_request)

    client = OllamaClient(
        base_url="http://127.0.0.1:11434",
        model="qwen2.5:7b",
        timeout_seconds=5,
        max_retries=1,
    )

    health = client.healthcheck(required_model="qwen2.5:7b")
    assert health.reachable is True
    assert health.failure_reason is not None
    assert "not available" in health.failure_reason
