# Explainable Credit Risk Modeling with Alternative Data and Local LLMs

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/env-uv-4B8BBE)](https://docs.astral.sh/uv/)

An artifact-driven credit risk pipeline for the Kaggle Home Credit Default Risk dataset, built as a production-style repository rather than a notebook-only experiment.

The project covers the full lifecycle:

- data acquisition and quality validation
- feature engineering over relational tables
- baseline and tuned model workflows
- explainability artifacts (SHAP and LIME)
- local LLM narrative reporting with fallback behavior
- FastAPI service endpoints for demo scoring and retrieval
- final artifact contract verification for demo readiness

## Overview

This repository exists to answer a practical question:

How do you build a local, reproducible, explainable credit-risk system that can be validated, demoed, and operated with clear artifact contracts?

The emphasis is reliability and transparency:

- deterministic file outputs between phases
- explicit model and artifact selection metadata
- robust fallback behavior when local LLM generation is unavailable
- CI checks and test coverage that do not depend on Kaggle/Ollama runtime availability

## Why This Project

Credit risk projects often stop at model training notebooks. This one continues through explainability, report generation, and an API layer, because that is where many deployment failures happen.

The repository is designed to be credible for engineering review:

- modular source layout under src
- CLI-first orchestration
- typed code and strict quality checks
- explicit readiness checks and failure messages

## System Architecture

Pipeline layers:

1. Data Layer
- Download Home Credit data from Kaggle
- Validate required files and key schema assumptions
- Convert CSV to interim parquet for stable downstream I/O

2. Feature Layer
- Build train/test matrices with one row per SK_ID_CURR
- Emit feature manifest and join metadata

3. Modeling Layer
- Baseline LightGBM/CatBoost evaluation
- Hyperparameter tuning and calibration comparisons
- Final production-candidate summary and model materialization

4. Explainability Layer
- Representative cohort selection
- SHAP global/local and LIME local artifact generation

5. Local LLM Reporting Layer
- Narrative generation (plain, underwriter, adverse-action-style draft)
- Deterministic fallback path when Ollama/model/generation fails

6. Service Layer
- FastAPI endpoints for health/readiness/score/explain/report retrieval/generation

7. Audit Layer
- Artifact contract verification gate before demos

See [docs/architecture.md](docs/architecture.md) for a compact architecture reference.

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

## Tech Stack

- Python 3.13
- uv (dependency and environment management)
- pandas, polars, pyarrow
- scikit-learn, lightgbm, catboost, imbalanced-learn, optuna
- shap, lime, matplotlib, plotly
- FastAPI, Uvicorn
- Ollama (local runtime)

## Dataset and Problem Framing

- Dataset: Kaggle Home Credit Default Risk (home-credit-default-risk)
- Goal: predict default risk and surface interpretable, reviewable evidence
- Scope: local development and demo workflows, not a regulated production deployment

## Implementation Walkthrough (Phase by Phase)

Phase 1: Foundations
- Project config, CLI entrypoint, environment conventions

Phase 2: Data ingestion and validation
- Kaggle download flow
- raw-data validation reports and explicit failure messaging

Phase 3: Feature engineering
- modular feature builders per source table
- leakage-safe joins and metadata manifest

Phase 4: Baseline modeling
- stratified CV baselines for LightGBM and CatBoost
- baseline metrics and summary artifacts

Phase 5: Tuning, calibration, and final selection
- Optuna tuning
- calibration comparisons
- final production candidate summary + final model output

Phase 6: Explainability
- representative sample selection by cohort
- SHAP/LIME local explanation payloads

Phase 7: Local LLM reporting
- report generation from explainability payloads
- deterministic fallback with explicit failure reason capture

Phase 8: FastAPI service layer
- health/readiness
- scoring endpoint
- explanation and narrative retrieval/generation endpoints

Phase 9: Hardening and release readiness
- CI workflow
- artifact verification gate
- documentation and demo runbook polish

## Key Features and Why They Exist

- CLI orchestration
  Keeps runs reproducible and scriptable without notebook state drift.

- Artifact contracts
  Every phase writes explicit machine-readable outputs for the next phase.

- Local explainability and local LLM reporting
  Keeps evidence generation on local infrastructure with clear fallback behavior.

- API readiness endpoint
  Surfaces dependency/artifact health before live demos.

- verify-artifacts gate
  Prevents "tests passed but demo artifacts are missing" situations.

## Challenges Encountered and What Was Solved

- Artifact contract drift between phases
  Solved by adding a dedicated artifact verification command with required checks.

- API cache safety under repeated calls
  Solved by hardening model-store synchronization to avoid nested-cache lock issues.

- Explainability robustness on unstable feature matrices
  Solved by filtering unstable/non-numeric signals and recording per-case failures.

- Local LLM availability variability
  Solved with deterministic fallback generation and explicit fallback flags/reasons.

## Production-Style Characteristics (vs Notebook-Only)

- typed source modules with tests
- CI checks for lint, typing, tests, import smoke
- PowerShell automation scripts for setup and checks
- API endpoints with structured error handling
- artifact verification as a release/demo gate

## Tradeoffs

- CI intentionally skips heavyweight data/model runs
  Faster and stable CI, but artifact generation remains a local workflow.

- Final-candidate selection currently scoped to tuned candidates
  Clear Phase 5 contract, but baseline candidates are not auto-promoted in final selection.

- Explain/report endpoints are artifact-backed
  Reliable and fast for known IDs, but not a generic online explainability service.

## Setup (Windows 11 + PowerShell + uv)

Prerequisites:

- Python 3.13
- uv in PATH
- Ollama installed locally

Quick start:

```powershell
.\scripts\bootstrap.ps1
```

Manual setup:

```powershell
uv sync --python 3.13 --frozen
Copy-Item .env.example .env
uv run credit-risk prepare-dirs
```

## Kaggle Setup

Option A: .env

```dotenv
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

Option B: %USERPROFILE%\.kaggle\kaggle.json

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

## End-to-End Commands

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

Developer quality workflow:

```powershell
.\scripts\run_checks.ps1
.\scripts\run_checks.ps1 -IncludeArtifactAudit
```

## API Usage Overview

Endpoints:

- GET /health
- GET /readiness
- POST /score
- POST /explain
- POST /risk-report
- GET /artifacts/summary

Quick check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/readiness
Invoke-RestMethod http://127.0.0.1:8000/artifacts/summary
```

Score example:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/score `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"applicant_id":700001,"engineered_features":{"feature_a":0.5,"feature_b":0.4,"feature_c":0.3}}'
```

## Artifact Guide

Core contract artifacts checked by verify-artifacts:

- data/processed/home_credit/train_features.parquet
- data/processed/home_credit/test_features.parquet
- artifacts/feature_metadata/feature_manifest.csv
- artifacts/modeling/reports/best_model_summary.json
- artifacts/modeling/tuning/tuning_results.csv
- artifacts/modeling/metrics/tuned_model_comparison.csv
- artifacts/modeling/reports/final_production_candidate.json
- artifacts/modeling/models/final_production_model.*
- artifacts/explainability/selected_examples/selected_examples.csv
- artifacts/explainability/reports/explainability_summary.md
- artifacts/explainability/shap/local/shap_local_explanations.jsonl or artifacts/explainability/lime/lime_explanations.jsonl
- artifacts/llm_reports/combined/llm_reports.jsonl
- artifacts/llm_reports/reports/llm_reporting_summary.md

Demo runbook: [docs/demo_runbook.md](docs/demo_runbook.md)

## Results and Evaluation (Current Repository Artifacts)

The current checked artifacts show:

- Baseline best model summary (artifacts/modeling/reports/best_model_summary.json)
  - best_model_name: lightgbm
  - primary_metric (roc_auc): 0.7864

- Final production candidate summary (artifacts/modeling/reports/final_production_candidate.json)
  - final_candidate_name: lightgbm_tuned_none
  - primary_metric (roc_auc): 0.7604

- Evaluation summary (artifacts/modeling/evaluation/evaluation_summary.json)
  - roc_auc: 0.7604
  - pr_auc: 0.2494
  - row_count: 307511

Important interpretation:

- In the current repository state, the baseline LightGBM ROC AUC is higher than the selected tuned final candidate ROC AUC.
- This is visible in artifacts/modeling/metrics/tuned_model_comparison.csv.
- The final candidate selection flow is currently tuned-candidate scoped by design in Phase 5; it does not automatically promote baseline candidates.

This is documented intentionally rather than hidden.

## Limitations

- CI checks code quality and tests, not full data/model/LLM pipeline execution.
- API explain/report endpoints are artifact-backed and applicant-ID oriented.
- LLM outputs vary by local model/runtime state unless fallback is used.
- Adverse-action-style text is draft explanatory content only.

## Disclaimer

This project is for engineering demonstration and learning.  
Generated narratives, especially adverse-action-style drafts, are not legal advice and are not compliance-approved notices.

## Conclusion

This repository demonstrates a full, local-first ML system with explicit contracts from raw data through API serving.

The focus is not just model training; it is operational credibility:

- reproducible workflows
- explainability and reporting layers
- API readiness checks
- artifact verification before demos

For local demonstrations, run verify-artifacts before starting the API.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.