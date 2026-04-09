# Architecture Overview

This repository is organized as an artifact-driven ML system where each phase writes explicit outputs consumed by the next phase.

## Layered Flow

1. Data layer (`src/credit_risk_altdata/data`)
- Download Home Credit source files from Kaggle.
- Validate required files, schema assumptions, and core quality checks.
- Produce interim parquet tables.

2. Feature layer (`src/credit_risk_altdata/features`)
- Build one-row-per-`SK_ID_CURR` train/test matrices.
- Emit feature manifest and join summaries.

3. Modeling layer (`src/credit_risk_altdata/modeling`)
- Train baselines (LightGBM/CatBoost).
- Tune + calibrate + evaluate candidate models.
- Select and materialize final production candidate.

4. Explainability layer (`src/credit_risk_altdata/explainability`)
- Select representative applicants by cohort.
- Generate SHAP and/or LIME local artifacts.

5. Local LLM layer (`src/credit_risk_altdata/llm`)
- Convert explainability payloads into narratives via local Ollama.
- Fall back to deterministic templates if generation is unavailable.

6. Service layer (`src/credit_risk_altdata/api`)
- Expose health/readiness and artifact-backed score/explain/report endpoints.
- Keep runtime behavior deterministic and contract-focused.

7. Audit layer (`src/credit_risk_altdata/audit`)
- Verify that canonical artifacts and references are present and coherent.
- Provide a final gate before demos.

## Runtime Interfaces

- CLI entrypoint: `credit-risk`
- API entrypoint: `credit-risk-api`
- Main operator scripts:
  - `scripts/bootstrap.ps1`
  - `scripts/run_checks.ps1`

## Core Artifact Contracts

- Processed feature matrices
- Baseline summary and tuned comparison
- Final production candidate summary
- Final production model artifact
- Explainability outputs
- LLM report outputs

The canonical contract check is `credit-risk verify-artifacts`.
