# Contributing

## Development setup

Use Python 3.10 or newer in an isolated virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,app]"
```

Do not commit environment files, API caches, downloaded data, generated
outputs, credentials, or private portfolio information.

## Tests

Run the regression suite without pytest or bytecode caches:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  -p no:cacheprovider --disable-warnings -q
```

Changes to return conventions, horizons, VaR/CVaR, backtesting, simulations,
covariance estimation, optimization, or solver handling require focused tests
and an explicit explanation of their financial-behavior impact.

## Quality checks

Run the same local gates used by CI:

```bash
ruff check app.py run_demo.py run_phase5_optimization_demo.py src tests
python -m pytest -p no:cacheprovider -q \
  --cov=var_cvar_crypto_risk --cov-fail-under=68
python -m build
```

`pyproject.toml` is the dependency source of truth. Do not add independent
version lists to the compatibility requirements files.

## Pull requests

Keep changes narrowly scoped. Describe the root cause, implementation, tests,
model-risk implications, and whether numerical outputs change. Never combine a
financial-methodology change with an unrelated architectural refactor.
