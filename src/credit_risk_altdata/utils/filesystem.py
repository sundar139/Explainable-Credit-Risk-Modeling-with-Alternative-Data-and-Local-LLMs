"""File system utility helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def ensure_directories(paths: Iterable[Path]) -> list[Path]:
    """Ensure all given directories exist and return normalized paths."""
    ensured_paths: list[Path] = []
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        ensured_paths.append(path)
    return ensured_paths
