param(
	[switch]$SkipTests,
	[switch]$IncludeArtifactAudit
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "Running Ruff lint checks..." -ForegroundColor Cyan
uv run ruff check .

Write-Host "Running mypy type checks..." -ForegroundColor Cyan
uv run mypy src tests

if (-not $SkipTests) {
	Write-Host "Running pytest suite..." -ForegroundColor Cyan
	uv run pytest
}

Write-Host "Running CLI/API import smoke check..." -ForegroundColor Cyan
uv run python -c "from credit_risk_altdata.api.app import app; from credit_risk_altdata.cli import build_parser; assert app is not None; build_parser(); print('import-smoke-ok')"

if ($IncludeArtifactAudit) {
	Write-Host "Running artifact contract verification..." -ForegroundColor Cyan
	uv run credit-risk verify-artifacts
}

Write-Host "All checks passed." -ForegroundColor Green
