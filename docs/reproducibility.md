# Reproducibility

## Reproduction levels

The project distinguishes three levels:

1. **Code reproducibility** — the same version installs and its test suite
   passes.
2. **Numerical reproducibility** — the same pinned input, configuration, seed,
   dependency environment, and solver reproduce results within stated
   tolerances.
3. **Research reproducibility** — a reader can obtain legally usable input data
   and regenerate every table or figure used in an article.

Version 0.5.0 establishes a tested code baseline. A pinned article dataset and
artifact manifest are still required for full research reproducibility.

## Current workflow

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[app,dev]"

PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  -p no:cacheprovider --disable-warnings -q

python run_demo.py
python run_phase5_optimization_demo.py
```

Generated files are written beneath `outputs/` and intentionally ignored by
Git.

## Recording an experiment

For a result intended for GitHub, Medium, or LinkedIn, save an external manifest
containing:

```text
experiment_id
Git commit
package version
Python version
dependency lock or constraints hash
data source and extraction timestamp
input file hash
configuration hashes
asset universe and weights
return convention
horizon and confidence level
scenario source and count
random seed
expected-return estimator
covariance estimator and parameters
optimizer objective and constraints
output file hashes
```

## Live-data limitation

When `end_date` is unset, a later execution consumes a later sample. Vendors can
also revise observations. Screenshots from live data must therefore display the
actual data cutoff and should not be presented as exactly reproducible.

## Planned release controls

Before `v1.0.0`, add:

- a reviewed cross-platform lock strategy based on `pyproject.toml`;
- wheel and source-distribution installation checks;
- a synthetic or redistribution-approved article fixture;
- golden numerical outputs with explicit tolerances;
- a manifest generator for publication artifacts.

CI now performs clean installation and regression tests on Python 3.10 through
3.13, plus lint, coverage, build, and Streamlit startup checks. These gates test
compatibility against the declared dependency ranges; they do not replace a
pinned research environment.
