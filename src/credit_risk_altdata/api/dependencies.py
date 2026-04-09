"""Dependency providers for API runtime services."""

from __future__ import annotations

from functools import lru_cache

from credit_risk_altdata.api.model_store import ModelStore
from credit_risk_altdata.api.services import APIRuntimeService
from credit_risk_altdata.config import get_settings


@lru_cache(maxsize=1)
def get_model_store() -> ModelStore:
    """Return cached model store for API process lifetime."""
    return ModelStore(get_settings())


@lru_cache(maxsize=1)
def get_runtime_service() -> APIRuntimeService:
    """Return cached API runtime service for endpoint handlers."""
    return APIRuntimeService(get_settings(), model_store=get_model_store())


def reset_api_dependency_cache() -> None:
    """Clear dependency caches for testing and controlled reloads."""
    get_model_store.cache_clear()
    get_runtime_service.cache_clear()
