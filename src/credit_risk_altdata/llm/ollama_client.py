"""Local Ollama HTTP client with retry and availability checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from requests import RequestException, Response


class OllamaClientError(RuntimeError):
    """Raised when Ollama communication fails."""


@dataclass(frozen=True, slots=True)
class OllamaHealthcheckResult:
    """Healthcheck and model availability details."""

    reachable: bool
    available_models: tuple[str, ...]
    failure_reason: str | None = None


class OllamaClient:
    """Thin Ollama client for local, non-streaming text generation."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model).strip()
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)

        if not self.base_url:
            raise ValueError("base_url must not be empty")
        if not self.model:
            raise ValueError("model must not be empty")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

    def _build_url(self, path: str) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        return f"{self.base_url}{normalized_path}"

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        url = self._build_url(path)
        last_error: str | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response: Response = requests.request(
                    method=method,
                    url=url,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except RequestException as exc:
                last_error = f"HTTP request failed: {exc}"
                if attempt < self.max_retries:
                    continue
                raise OllamaClientError(last_error) from exc

            if response.status_code >= 500:
                last_error = (
                    f"Ollama server error ({response.status_code}) for {url}: "
                    f"{response.text[:300]}"
                )
                if attempt < self.max_retries:
                    continue
                raise OllamaClientError(last_error)

            if response.status_code >= 400:
                raise OllamaClientError(
                    f"Ollama request failed ({response.status_code}) for {url}: "
                    f"{response.text[:300]}"
                )

            try:
                data_raw = response.json()
            except ValueError as exc:
                raise OllamaClientError(
                    f"Ollama returned non-JSON response for {url}: {response.text[:300]}"
                ) from exc

            if not isinstance(data_raw, dict):
                raise OllamaClientError(
                    f"Ollama returned unexpected payload type for {url}: "
                    f"{type(data_raw).__name__}"
                )
            return data_raw

        raise OllamaClientError(last_error or "Unknown Ollama client failure")

    def list_models(self) -> tuple[str, ...]:
        """Return available model tags from local Ollama registry."""
        payload = self._request_json(method="GET", path="/api/tags", payload=None)
        models_raw = payload.get("models")

        if not isinstance(models_raw, list):
            return tuple()

        names: set[str] = set()
        for entry in models_raw:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())

        return tuple(sorted(names))

    def healthcheck(self, *, required_model: str | None = None) -> OllamaHealthcheckResult:
        """Return endpoint reachability and optional model availability status."""
        try:
            available_models = self.list_models()
        except OllamaClientError as exc:
            return OllamaHealthcheckResult(
                reachable=False,
                available_models=tuple(),
                failure_reason=str(exc),
            )

        if required_model is not None:
            model_name = required_model.strip()
            if model_name and model_name not in available_models:
                return OllamaHealthcheckResult(
                    reachable=True,
                    available_models=available_models,
                    failure_reason=(
                        f"Model '{model_name}' is not available in local Ollama tags"
                    ),
                )

        return OllamaHealthcheckResult(reachable=True, available_models=available_models)

    def generate(
        self,
        *,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Generate text using local Ollama `/api/generate` endpoint."""
        prompt_text = prompt.strip()
        if not prompt_text:
            raise ValueError("prompt must not be empty")

        resolved_model = (model or self.model).strip()
        if not resolved_model:
            raise ValueError("model must not be empty")

        payload: dict[str, Any] = {
            "model": resolved_model,
            "prompt": prompt_text,
            "stream": False,
        }
        if system_prompt is not None and system_prompt.strip():
            payload["system"] = system_prompt.strip()
        if options is not None:
            payload["options"] = options

        response_payload = self._request_json(
            method="POST",
            path="/api/generate",
            payload=payload,
        )

        error_text = response_payload.get("error")
        if isinstance(error_text, str) and error_text.strip():
            raise OllamaClientError(f"Ollama generation error: {error_text.strip()}")

        response_text = response_payload.get("response")
        if not isinstance(response_text, str) or not response_text.strip():
            raise OllamaClientError("Ollama returned an empty generation response")

        return response_text.strip()
