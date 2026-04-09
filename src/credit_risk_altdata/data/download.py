"""Kaggle download pipeline for Home Credit raw data."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from credit_risk_altdata.config import Settings
from credit_risk_altdata.data.constants import (
    CORE_RAW_FILES,
    HOME_CREDIT_COMPETITION,
)
from credit_risk_altdata.data.loaders import resolve_home_credit_paths
from credit_risk_altdata.utils.filesystem import ensure_directories


class DataDownloadError(RuntimeError):
    """Raised when dataset download or extraction fails."""


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Result of a download operation."""

    destination: Path
    downloaded: bool
    extracted_files: tuple[str, ...]
    skipped: bool


def _kaggle_credentials_file() -> Path:
    return Path.home() / ".kaggle" / "kaggle.json"


def _kaggle_credentials_available(settings: Settings) -> bool:
    has_env_credentials = bool(settings.kaggle_username and settings.kaggle_key)
    if has_env_credentials:
        return True
    return _kaggle_credentials_file().exists()


def _configure_kaggle_env(settings: Settings) -> None:
    if settings.kaggle_username and settings.kaggle_key:
        os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
        os.environ["KAGGLE_KEY"] = settings.kaggle_key


def _find_zip_file(destination: Path, dataset_slug: str) -> Path:
    preferred_name = destination / f"{dataset_slug}.zip"
    if preferred_name.exists():
        return preferred_name

    zip_candidates = sorted(
        destination.glob("*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not zip_candidates:
        raise DataDownloadError(
            f"Kaggle download did not create a zip file in {destination}"
        )
    return zip_candidates[0]


def _safe_extract(zip_path: Path, destination: Path) -> tuple[str, ...]:
    destination_resolved = destination.resolve()
    extracted_files: list[str] = []

    with ZipFile(zip_path, mode="r") as archive:
        for member in archive.infolist():
            member_target = (destination / member.filename).resolve()
            if (
                destination_resolved not in member_target.parents
                and member_target != destination_resolved
            ):
                raise DataDownloadError(
                    f"Unsafe zip member path detected: {member.filename}"
                )

            archive.extract(member, destination)
            if not member.is_dir():
                extracted_files.append(member.filename)

    return tuple(sorted(extracted_files))


def _cleanup_existing_files(destination: Path) -> None:
    for file_name in CORE_RAW_FILES:
        file_path = destination / file_name
        if file_path.exists():
            file_path.unlink()

    for zip_file in destination.glob("*.zip"):
        zip_file.unlink()


def _validate_required_core_files(destination: Path) -> list[str]:
    return [file_name for file_name in CORE_RAW_FILES if not (destination / file_name).exists()]


def download_home_credit_dataset(
    settings: Settings,
    *,
    force: bool = False,
) -> DownloadResult:
    """Download and extract Home Credit raw files from Kaggle competition data."""
    dataset_slug = settings.kaggle_dataset or HOME_CREDIT_COMPETITION
    paths = resolve_home_credit_paths(settings)
    ensure_directories([paths.raw_dir])

    existing_required = _validate_required_core_files(paths.raw_dir)
    if not existing_required and not force:
        return DownloadResult(
            destination=paths.raw_dir,
            downloaded=False,
            extracted_files=tuple(),
            skipped=True,
        )

    if force:
        _cleanup_existing_files(paths.raw_dir)

    if not _kaggle_credentials_available(settings):
        credentials_file = _kaggle_credentials_file()
        raise DataDownloadError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
            f"or create {credentials_file}."
        )

    _configure_kaggle_env(settings)

    command = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "download",
        "-c",
        dataset_slug,
        "-p",
        str(paths.raw_dir),
    ]
    if force:
        command.append("--force")

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        output = completed.stderr.strip() or completed.stdout.strip()
        raise DataDownloadError(
            f"Failed to download Kaggle dataset '{dataset_slug}': {output}"
        )

    zip_path = _find_zip_file(paths.raw_dir, dataset_slug)
    extracted_files = _safe_extract(zip_path=zip_path, destination=paths.raw_dir)
    missing_after_extract = _validate_required_core_files(paths.raw_dir)
    if missing_after_extract:
        missing = ", ".join(sorted(missing_after_extract))
        raise DataDownloadError(
            "Download completed but required files are still missing: " + missing
        )

    return DownloadResult(
        destination=paths.raw_dir,
        downloaded=True,
        extracted_files=extracted_files,
        skipped=False,
    )
