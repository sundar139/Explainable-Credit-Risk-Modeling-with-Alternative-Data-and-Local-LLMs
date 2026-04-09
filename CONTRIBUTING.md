# Contributing

Thanks for your interest in improving this project.

## Local setup

```powershell
.\scripts\bootstrap.ps1
```

## Before opening a PR

```powershell
.\scripts\run_checks.ps1
```

If your change affects pipeline outputs or demo behavior, also run:

```powershell
uv run credit-risk verify-artifacts
```

## Scope expectations

- Keep changes modular and typed.
- Preserve existing CLI/API contracts unless a migration is documented.
- Avoid introducing online/runtime dependencies in CI.
- Do not commit secrets (`.env`, Kaggle keys, tokens).

## Pull request guidance

- Explain what changed and why.
- Link any affected commands or artifact paths.
- Include tests for meaningful behavior changes.
