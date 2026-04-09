param(
    [switch]$SkipPreCommit
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed or not available in PATH."
}

Write-Host "Syncing dependencies with uv..." -ForegroundColor Cyan
uv sync --python 3.13 --frozen

if (-not (Test-Path ".env")) {
    Write-Host "Creating .env from .env.example..." -ForegroundColor Cyan
    Copy-Item ".env.example" ".env"
}

Write-Host "Preparing standard data/artifact directories..." -ForegroundColor Cyan
uv run credit-risk prepare-dirs

if (-not $SkipPreCommit) {
    Write-Host "Installing pre-commit hooks..." -ForegroundColor Cyan
    uv run pre-commit install
}

Write-Host "Running import smoke check..." -ForegroundColor Cyan
uv run python -c "from credit_risk_altdata.api.app import app; from credit_risk_altdata.cli import build_parser; assert app is not None; build_parser(); print('import-smoke-ok')"

Write-Host "Bootstrap completed successfully." -ForegroundColor Green
