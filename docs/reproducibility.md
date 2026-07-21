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

Version 0.5.0 establishes a tested code baseline, a cross-platform dependency
lock, a deterministic synthetic numerical baseline, and a config-driven
publication workflow. The included experiment provides research
reproducibility for a synthetic methodology demonstration. A separately
licensed, pinned real-market dataset is still required before making empirical
claims about crypto-market behavior or performance.

## Current workflow

```bash
uv sync --locked --extra app --extra dev

PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync python -m pytest \
  -p no:cacheprovider -q

uv run --locked --no-sync python -m scripts.check_public_boundary

uv run --locked --no-sync python run_demo.py
uv run --locked --no-sync python run_phase5_optimization_demo.py
```

Generated files are written beneath `outputs/` and intentionally ignored by
Git.

## Numerical golden baseline

`tests/fixtures/synthetic_daily_prices.csv` is a deterministic, synthetic
three-asset price history. It is not vendor data, a calibrated market model, or
an investable backtest. Its only purpose is to detect unintended numerical
changes across the public analytics pipeline.

`tests/fixtures/golden_baseline.json` records:

- the synthetic input and `uv.lock` SHA-256 hashes;
- simple portfolio-return statistics;
- Historical, Gaussian, and Cornish-Fisher VaR;
- Historical and Gaussian CVaR;
- robust expected-return estimates;
- sample, EWMA, and constant-correlation shrinkage covariance estimates;
- seeded seven-day Normal Monte Carlo risk; and
- a one-day rolling Gaussian VaR/Kupiec snapshot.

Deterministic analytical outputs use `rtol=1e-10` and `atol=1e-12`. Seeded
Monte Carlo outputs use `rtol=1e-7` and `atol=1e-9` to allow immaterial
cross-platform linear-algebra variation without accepting economically
meaningful drift.

Print a candidate baseline with:

```bash
uv run --locked --no-sync python -m scripts.generate_numerical_baseline
```

Replacing the committed file requires the explicit `--write` option. A changed
golden file is never an automatic fix: the reviewer must identify whether the
cause is an intended formula change, a dependency update, a tolerance problem,
or a regression. The numerical baseline tests code stability; it does not
validate model accuracy or future risk forecasts.

## Reproducible publication experiment

The repository-local `methodology-demo-v1` experiment freezes a synthetic input
hash, cutoff, portfolio, return convention, estimator settings, backtest,
optimizer constraints, and article-to-app mapping. Generate it only from a
clean reviewed commit:

```bash
uv run --locked --no-sync python -m scripts.reproduce_publication \
  --config publication/configs/methodology_demo_v1.yaml \
  --output-dir publication/artifacts/methodology-demo-v1
```

Then verify it:

```bash
uv run --locked --no-sync python -m scripts.reproduce_publication \
  --verify publication/artifacts/methodology-demo-v1/manifest.json
```

The generated manifest records:

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
return handling mode and diagnostic convention
horizon and confidence level
scenario source and count
random seed
expected-return estimator
covariance estimator and parameters
covariance validation policy, repair flag, and adjustment diagnostics
optimizer objective and constraints
solver, raw solver status, residual tolerance, and maximum violation
output file hashes
```

Generation refuses a dirty tree by default. `--allow-dirty` is a preview-only
escape hatch and must not be used for article artifacts. Verification detects
changes to calculation sources, the config, dependency lock, dataset, cutoff,
Git commit, and generated artifacts. It verifies provenance and integrity, not
economic validity.

See [Reproducible publication experiments](../publication/README.md) for the
claims boundary and article workflow.

## Live-data limitation

When `end_date` is unset, a later execution consumes a later sample. Vendors can
also revise observations. Screenshots from live data must therefore display the
actual data cutoff and should not be presented as exactly reproducible.

## Remaining research-publication control

The synthetic workflow is complete. Before a real-market article, select a
dataset whose license explicitly permits the intended redistribution and use,
record its attribution and hash, choose a point-in-time asset universe, and
separate estimation from out-of-sample evaluation. Replacing synthetic prices
with vendor data without those controls would create licensing, survivorship,
and look-ahead risks.

The complete local and GitHub-hosted publication gates are listed in the
[public release checklist](public-release-checklist.md).

The committed `uv.lock` provides the exact dependency environment. CI verifies
it on Python 3.10 through 3.13 and also runs lint, coverage, build, numerical
baseline, and Streamlit startup checks. The synthetic fixture under
`tests/fixtures/` is safe to redistribute and guards deterministic calculations
within documented tolerances; it is not evidence of forecasting performance or
market validity.
