# Contributing

## Development setup

Use Python 3.10 through 3.13 and uv 0.11.16. Create the exact locked
development environment with:

```bash
uv sync --locked --extra app --extra dev
```

Do not commit environment files, API caches, downloaded data, generated
outputs, credentials, or private portfolio information.

## Tests

Run the regression suite without pytest or bytecode caches:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync python -m pytest \
  -p no:cacheprovider -q
```

Changes to return conventions, horizons, VaR/CVaR, backtesting, simulations,
covariance estimation, optimization, or solver handling require focused tests
and an explicit explanation of their financial-behavior impact.

## Quality checks

Run the same local gates used by CI:

```bash
uv lock --check
uv run --locked --no-sync ruff check \
  app.py src tests scripts
uv run --locked --no-sync python -m scripts.check_public_boundary
uv run --locked --no-sync python -m scripts.check_git_history_boundary
uv run --locked --no-sync python -m pytest -p no:cacheprovider -q \
  --cov=var_cvar_crypto_risk --cov-fail-under=80
uv run --locked --no-sync python -m build --no-isolation
```

`pyproject.toml` is the dependency source of truth. Do not add independent
version lists to the compatibility requirements files. Dependency changes must
update `uv.lock` in the same focused pull request.

If an intentional formula or dependency change affects the numerical golden
test, print the candidate baseline first:

```bash
uv run --locked --no-sync python -m scripts.generate_numerical_baseline
```

Use `--write` only after reviewing and explaining every material numerical
change. Never refresh the golden file merely to make CI pass.

## Pull requests

Keep changes narrowly scoped. Describe the root cause, implementation, tests,
model-risk implications, and whether numerical outputs change. Never combine a
financial-methodology change with an unrelated architectural refactor.

Use the pull-request template. Public issues and pull requests must not contain
credentials, local paths, private datasets, real portfolio details, transaction
history, or proprietary parameters. Report suspected security vulnerabilities
through the private process in `SECURITY.md`.
