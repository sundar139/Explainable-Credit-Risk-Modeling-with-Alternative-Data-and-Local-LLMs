"""Shared logging setup for CLI, API, and future pipelines."""

from __future__ import annotations

import logging
from logging.config import dictConfig

from credit_risk_altdata.config import get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str | None = None) -> None:
    """Configure project-wide console logging."""
    resolved_level = (level or get_settings().log_level).upper()
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": LOG_FORMAT,
                    "datefmt": DATE_FORMAT,
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": resolved_level,
                }
            },
            "root": {
                "handlers": ["console"],
                "level": resolved_level,
            },
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger instance."""
    return logging.getLogger(name)
