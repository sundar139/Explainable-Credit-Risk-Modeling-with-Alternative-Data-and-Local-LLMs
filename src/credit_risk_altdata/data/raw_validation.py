"""Validation logic for Home Credit raw tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from pandas import DataFrame

from credit_risk_altdata.config import Settings
from credit_risk_altdata.data.constants import (
    DUPLICATE_CHECK_RULES,
    REQUIRED_KEY_COLUMNS,
)
from credit_risk_altdata.data.loaders import (
    get_core_raw_file_paths,
    missing_core_raw_files,
    read_home_credit_table,
)
from credit_risk_altdata.data.reporting import (
    ensure_report_directory,
    write_csv_report,
    write_json_report,
    write_markdown_report,
)


@dataclass(frozen=True, slots=True)
class DuplicateCheckResult:
    """Result of duplicate analysis for a key set."""

    file_name: str
    keys: tuple[str, ...]
    duplicate_count: int
    duplicate_rate: float
    expect_unique: bool


@dataclass(frozen=True, slots=True)
class TableQualitySummary:
    """Quality summary for an individual table."""

    file_name: str
    row_count: int
    column_count: int
    required_key_columns: tuple[str, ...]
    missing_required_key_columns: tuple[str, ...]
    columns_with_missing: int
    total_missing_cells: int
    duplicate_checks: tuple[DuplicateCheckResult, ...]


@dataclass(frozen=True, slots=True)
class RawDataValidationResult:
    """Validation result payload for raw data checks."""

    is_valid: bool
    missing_files: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    table_summaries: tuple[TableQualitySummary, ...]
    report_paths: dict[str, Path]


def _utc_timestamp() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _build_schema_records(file_name: str, dataframe: DataFrame) -> list[dict[str, object]]:
    row_count = int(dataframe.shape[0])
    records: list[dict[str, object]] = []
    for column_name in dataframe.columns:
        series = dataframe[column_name]
        records.append(
            {
                "file_name": file_name,
                "column_name": str(column_name),
                "dtype": str(series.dtype),
                "row_count": row_count,
                "non_null_count": int(series.notna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
            }
        )
    return records


def _build_missingness_records(file_name: str, dataframe: DataFrame) -> list[dict[str, object]]:
    row_count = int(dataframe.shape[0])
    records: list[dict[str, object]] = []
    for column_name in dataframe.columns:
        series = dataframe[column_name]
        missing_count = int(series.isna().sum())
        missing_pct = float((missing_count / row_count) * 100) if row_count else 0.0
        records.append(
            {
                "file_name": file_name,
                "column_name": str(column_name),
                "missing_count": missing_count,
                "missing_pct": round(missing_pct, 6),
            }
        )
    return records


def validate_required_raw_files(settings: Settings) -> list[str]:
    """Return missing required Home Credit raw CSV files."""
    return missing_core_raw_files(settings)


def validate_raw_data(
    settings: Settings,
    *,
    report_dir: Path | None = None,
) -> RawDataValidationResult:
    """Validate raw Home Credit files and persist quality reports."""
    errors: list[str] = []
    warnings: list[str] = []
    schema_records: list[dict[str, object]] = []
    missingness_records: list[dict[str, object]] = []
    table_summaries: list[TableQualitySummary] = []

    required_file_paths = get_core_raw_file_paths(settings)
    missing_files = sorted(validate_required_raw_files(settings))
    if missing_files:
        errors.append(
            "Missing required raw files: " + ", ".join(missing_files)
        )

    for file_name, file_path in required_file_paths.items():
        if not file_path.exists():
            continue

        try:
            dataframe = read_home_credit_table(settings=settings, file_name=file_name)
        except Exception as exc:
            errors.append(f"Failed to load {file_name}: {exc}")
            continue

        row_count = int(dataframe.shape[0])
        column_count = int(dataframe.shape[1])
        if row_count == 0:
            errors.append(f"{file_name} has zero rows")
        if column_count == 0:
            errors.append(f"{file_name} has zero columns")

        required_columns = REQUIRED_KEY_COLUMNS.get(file_name, ())
        missing_required_columns = tuple(
            column for column in required_columns if column not in dataframe.columns
        )
        if missing_required_columns:
            errors.append(
                f"{file_name} is missing required key columns: "
                + ", ".join(missing_required_columns)
            )

        schema_records.extend(_build_schema_records(file_name=file_name, dataframe=dataframe))
        missingness_records.extend(
            _build_missingness_records(file_name=file_name, dataframe=dataframe)
        )

        duplicate_results: list[DuplicateCheckResult] = []
        for rule in DUPLICATE_CHECK_RULES.get(file_name, ()):  # pragma: no branch
            missing_duplicate_columns = [
                column for column in rule.keys if column not in dataframe.columns
            ]
            if missing_duplicate_columns:
                warnings.append(
                    f"Skipped duplicate check for {file_name} on {rule.keys}: "
                    f"missing columns {missing_duplicate_columns}"
                )
                continue

            duplicate_count = int(dataframe.duplicated(subset=list(rule.keys)).sum())
            duplicate_rate = float((duplicate_count / row_count) if row_count else 0.0)
            result = DuplicateCheckResult(
                file_name=file_name,
                keys=rule.keys,
                duplicate_count=duplicate_count,
                duplicate_rate=round(duplicate_rate, 8),
                expect_unique=rule.expect_unique,
            )
            duplicate_results.append(result)

            if rule.expect_unique and duplicate_count > 0:
                key_label = ", ".join(rule.keys)
                errors.append(
                    f"{file_name} has duplicate keys for [{key_label}]: "
                    f"{duplicate_count} duplicates"
                )

        columns_with_missing = int((dataframe.isna().sum() > 0).sum())
        total_missing_cells = int(dataframe.isna().sum().sum())
        table_summaries.append(
            TableQualitySummary(
                file_name=file_name,
                row_count=row_count,
                column_count=column_count,
                required_key_columns=tuple(required_columns),
                missing_required_key_columns=missing_required_columns,
                columns_with_missing=columns_with_missing,
                total_missing_cells=total_missing_cells,
                duplicate_checks=tuple(duplicate_results),
            )
        )

    final_report_dir = report_dir or settings.data_validation_dir
    ensure_report_directory(final_report_dir)

    schema_json_path = write_json_report(
        final_report_dir / "schema_summary.json",
        {
            "generated_at": _utc_timestamp(),
            "records": schema_records,
        },
    )

    missingness_json_path = write_json_report(
        final_report_dir / "missingness_summary.json",
        {
            "generated_at": _utc_timestamp(),
            "records": missingness_records,
        },
    )

    missingness_csv_path = write_csv_report(
        final_report_dir / "missingness_summary.csv",
        missingness_records,
    )

    data_quality_payload = {
        "generated_at": _utc_timestamp(),
        "is_valid": len(errors) == 0,
        "missing_files": missing_files,
        "errors": errors,
        "warnings": warnings,
        "table_summaries": [asdict(summary) for summary in table_summaries],
    }
    quality_json_path = write_json_report(
        final_report_dir / "data_quality_summary.json",
        data_quality_payload,
    )

    markdown_lines: list[str] = [
        f"Generated at: {_utc_timestamp()}",
        "",
        f"Validation status: {'PASS' if len(errors) == 0 else 'FAIL'}",
        f"Checked tables: {len(table_summaries)}",
        f"Missing required files: {len(missing_files)}",
        f"Error count: {len(errors)}",
        f"Warning count: {len(warnings)}",
        "",
        "## Errors",
    ]
    if errors:
        markdown_lines.extend([f"- {message}" for message in errors])
    else:
        markdown_lines.append("- None")

    markdown_lines.extend(["", "## Warnings"])
    if warnings:
        markdown_lines.extend([f"- {message}" for message in warnings])
    else:
        markdown_lines.append("- None")

    markdown_lines.extend(["", "## Table Summaries"])
    for summary in table_summaries:
        markdown_lines.append(
            "- "
            f"{summary.file_name}: rows={summary.row_count}, cols={summary.column_count}, "
            f"missing_cells={summary.total_missing_cells}"
        )

    summary_markdown_path = write_markdown_report(
        final_report_dir / "validation_summary.md",
        title="Raw Data Validation Summary",
        lines=markdown_lines,
    )

    report_paths = {
        "schema_json": schema_json_path,
        "missingness_json": missingness_json_path,
        "missingness_csv": missingness_csv_path,
        "data_quality_json": quality_json_path,
        "summary_markdown": summary_markdown_path,
    }

    return RawDataValidationResult(
        is_valid=len(errors) == 0,
        missing_files=tuple(missing_files),
        errors=tuple(errors),
        warnings=tuple(warnings),
        table_summaries=tuple(table_summaries),
        report_paths=report_paths,
    )
