[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/env-uv-4B8BBE)](https://docs.astral.sh/uv/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

# Explainable Credit Risk Modeling with Alternative Data and Local LLMs

A production-style, end-to-end credit risk project built on the Kaggle Home Credit Default Risk dataset.  
The repository combines feature engineering, baseline and tuned modeling, explainability artifacts, local LLM-assisted narrative reporting, and a demo-ready FastAPI service.

## Why This Project

This project is designed to look and behave like an industry ML repository:

- Reproducible, config-driven workflows
- Clear artifact contracts across phases
- Explainability-first model interpretation
- Local-only LLM reporting (Ollama) with deterministic fallback
- API service layer for live scoring and demo workflows

## Architecture Overview

Pipeline flow:

1. Data acquisition and raw validation
2. Leakage-safe feature engineering
3. Baseline training
4. Hyperparameter tuning + calibration + evaluation
5. Explainability artifact generation (SHAP/LIME)
6. Local narrative report generation (Ollama + fallback)
7. FastAPI scoring and retrieval/generation endpoints
8. Final artifact verification gate for demo/API readiness

See [docs/architecture.md](docs/architecture.md) for a concise component map.

## Key Capabilities

- Baseline and tuned LightGBM/CatBoost workflows
- Final production candidate selection with explicit summary artifact
- SHAP global + local explanations and LIME local explanations
- Local LLM risk narratives (plain, underwriter, adverse-action-style draft)
- Artifact-first API with health/readiness/score/explain/report endpoints
- Final `verify-artifacts` command to prevent demo-time missing outputs

## Tech Stack

- Python 3.13
- `uv` for environment and dependency management
- `pandas`, `polars`, `scikit-learn`, `lightgbm`, `catboost`, `optuna`
- `shap`, `lime` for explainability
- FastAPI + Uvicorn for serving
- Ollama for local LLM inference (`qwen2.5:7b`, `qwen2.5-coder:7b`)

## Dataset and Problem Framing

- Dataset: Kaggle Home Credit Default Risk (`home-credit-default-risk`)
- Task: Predict default risk and produce transparent, reviewable risk evidence
- Objective: Build a realistic, local-first ML workflow suitable for portfolio/recruiter review and technical demos

## Repository Structure

```text
.
|-- .github/workflows/ci.yml
|-- docs/
|   |-- architecture.md
|   `-- demo_runbook.md
|-- scripts/
|   |-- bootstrap.ps1
|   `-- run_checks.ps1
|-- src/credit_risk_altdata/
|   |-- api/
|   |-- audit/
|   |-- data/
|   |-- explainability/
|   |-- features/
|   |-- llm/
|   |-- modeling/
|   |-- cli.py
|   `-- config.py
|-- tests/
|-- data/
|-- artifacts/
|-- pyproject.toml
`-- .env.example
```

## Setup (Windows 11 + PowerShell + uv)

Prerequisites:

- Python 3.13 installed
- `uv` installed and available in `PATH`
- Ollama installed locally

### Quick setup

```powershell
.\scripts\bootstrap.ps1
```

This script:

- syncs dependencies with `uv`
- creates `.env` from `.env.example` if missing
- creates expected data/artifact directories
- installs pre-commit hooks
- runs an import smoke check

### Manual setup (optional)

```powershell
uv sync --python 3.13 --frozen
Copy-Item .env.example .env
uv run credit-risk prepare-dirs
```

## Kaggle Credentials

You can provide Kaggle credentials in either way:

1. `.env`

```dotenv
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

2. `%USERPROFILE%\.kaggle\kaggle.json`

```json
{
  "username": "your_username",
  "key": "your_api_key"
}
```

## Ollama Setup

```powershell
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
ollama list
uv run credit-risk healthcheck
```

## Developer Workflow Commands

### Setup

```powershell
.\scripts\bootstrap.ps1
```

### Quality checks

```powershell
.\scripts\run_checks.ps1
```

### Quality checks + artifact contract gate

```powershell
.\scripts\run_checks.ps1 -IncludeArtifactAudit
```

### API startup

```powershell
uv run credit-risk-api
```

### Local LLM report generation only

```powershell
uv run credit-risk generate-risk-reports --report-type all --method-source auto --model qwen2.5:7b --overwrite
```

## End-to-End Workflow (Phase 2 -> Phase 8)

Run from repository root:

```powershell
uv run credit-risk download-data
uv run credit-risk validate-raw-data
uv run credit-risk build-interim-parquet
uv run credit-risk build-features --input-source raw --overwrite
uv run credit-risk train-baselines --model all --overwrite
uv run credit-risk tune-models --model all --n-trials 20 --calibration all --overwrite
uv run credit-risk generate-explanations --method all --overwrite
uv run credit-risk generate-risk-reports --report-type all --method-source auto --model qwen2.5:7b --overwrite
uv run credit-risk verify-artifacts
uv run credit-risk-api
```

## API Endpoints

- `GET /health`
- `GET /readiness`
- `POST /score`
- `POST /explain`
- `POST /risk-report`
- `GET /artifacts/summary`

### Example API calls (PowerShell)

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/readiness
Invoke-RestMethod http://127.0.0.1:8000/artifacts/summary
```

Score request:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/score `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"applicant_id":700001,"engineered_features":{"feature_a":0.5,"feature_b":0.4,"feature_c":0.3}}'
```

## Artifact Directory Overview

| Artifact | Path |
|---|---|
| Processed train matrix | `data/processed/home_credit/train_features.parquet` |
| Processed test matrix | `data/processed/home_credit/test_features.parquet` |
| Baseline summary | `artifacts/modeling/reports/best_model_summary.json` |
| Tuned comparison | `artifacts/modeling/metrics/tuned_model_comparison.csv` |
| Final candidate summary | `artifacts/modeling/reports/final_production_candidate.json` |
| Final production model | `artifacts/modeling/models/final_production_model.*` |
| SHAP local explanations | `artifacts/explainability/shap/local/shap_local_explanations.jsonl` |
| LIME local explanations | `artifacts/explainability/lime/lime_explanations.jsonl` |
| LLM reports JSONL | `artifacts/llm_reports/combined/llm_reports.jsonl` |
| LLM reporting summary | `artifacts/llm_reports/reports/llm_reporting_summary.md` |

The final Phase 9 gate validates these contracts:

```powershell
uv run credit-risk verify-artifacts
```

## Demo Runbook

A realistic local demo sequence is documented in [docs/demo_runbook.md](docs/demo_runbook.md).

## CI and Release Readiness

GitHub Actions workflow: [.github/workflows/ci.yml](.github/workflows/ci.yml)

CI runs on `push` and `pull_request` for `main` and `master`, and executes:

- `ruff`
- `mypy`
- `pytest`
- import smoke check for CLI parser and FastAPI app

CI does not require Kaggle download or live Ollama inference.

## Known Limitations

- CI validates code quality and tests but does not build heavy artifacts from scratch.
- Explainability/report endpoints are artifact-backed and applicant-id scoped.
- LLM outputs are model-dependent and non-deterministic unless fallback is used.
- Adverse-action-style narratives are internal draft text, not legal notices.

## Disclaimer

This repository is for educational and engineering demonstration purposes.  
Generated narratives (especially adverse-action-style drafts) are not legal advice and must not be treated as production compliance output without legal, policy, and model governance review.

## License

MIT. See [LICENSE](LICENSE).
