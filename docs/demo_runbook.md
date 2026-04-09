# Demo Runbook

This runbook is for a local Windows 11 + PowerShell demo using `uv` and Ollama.

## 1) Validate environment

```powershell
uv --version
uv run python --version
ollama list
```

Optional quick bootstrap:

```powershell
.\scripts\bootstrap.ps1
```

## 2) Prepare core artifacts (if missing)

Run the full artifact chain:

```powershell
uv run credit-risk download-data
uv run credit-risk validate-raw-data
uv run credit-risk build-interim-parquet
uv run credit-risk build-features --input-source raw --overwrite
uv run credit-risk train-baselines --model all --overwrite
uv run credit-risk tune-models --model all --n-trials 20 --calibration all --overwrite
uv run credit-risk generate-explanations --method all --overwrite
uv run credit-risk generate-risk-reports --report-type all --method-source auto --model qwen2.5:7b --overwrite
```

## 3) Verify demo readiness contract

```powershell
uv run credit-risk verify-artifacts
```

If this command fails, fix missing artifacts before demoing the API.

## 4) Start API service

```powershell
uv run credit-risk-api
```

Keep this terminal open.

## 5) Validate health and readiness

In a new terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/readiness
Invoke-RestMethod http://127.0.0.1:8000/artifacts/summary
```

## 6) Demo scoring

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/score `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"applicant_id":700001,"engineered_features":{"feature_a":0.5,"feature_b":0.4,"feature_c":0.3}}'
```

## 7) Demo explanation retrieval

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/explain `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"applicant_id":700001,"explanation_method":"auto","allow_generate_if_missing":false}'
```

## 8) Demo risk report retrieval/generation

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/risk-report `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"applicant_id":700001,"explanation_method_source":"auto","report_type":"plain","allow_generate_if_missing":true,"allow_fallback":true}'
```

## 9) Optional: show fallback behavior

1. Stop Ollama or point to a non-running URL in `.env`.
2. Ensure `LLM_REPORTS_ENABLE_FALLBACK=true`.
3. Re-run report generation and show `fallback_generated=true` in output.

## 10) Suggested demo narrative

- Show `readiness` first to prove artifact-backed service assumptions.
- Show one score call and one explanation retrieval.
- Show one narrative report and mention fallback safety behavior.
- Close with `verify-artifacts` as the project hardening gate.
