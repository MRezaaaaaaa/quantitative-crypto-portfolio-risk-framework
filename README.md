# Crypto Portfolio Risk Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.5.0-informational)](CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research-oriented Python and Streamlit platform for measuring, validating,
simulating, and optimizing multi-asset portfolio tail risk. The current
development version focuses on crypto assets while retaining a generic tabular
portfolio core.

The project is designed to make model assumptions visible. It is not a trading
system, a regulatory capital implementation, or evidence of future investment
performance.

## What it does

- Calculates Historical, Gaussian, and Cornish-Fisher VaR.
- Calculates Historical and Gaussian CVaR / Expected Shortfall.
- Produces horizon-aware portfolio and asset-level risk diagnostics.
- Backtests VaR with rolling forecasts, breach analysis, Kupiec POF, and
  Christoffersen independence and conditional-coverage tests.
- Simulates multivariate Normal and Student-t portfolio scenarios.
- Optimizes portfolios with scenario-based CVaR objectives and constraints.
- Compares historical mean, median, trimmed, winsorized, shrinkage, and manual
  expected-return assumptions.
- Compares sample, EWMA, and linearly shrunk covariance estimates.
- Exposes assumptions and diagnostics through an interactive Streamlit app.

## Important interpretation

VaR identifies a loss quantile; CVaR estimates the average loss in the tail
beyond that quantile. Neither is a worst-case loss. Results depend materially on
the sample period, horizon, return convention, scenario model, covariance
estimate, expected-return estimate, and portfolio constraints.

The statistical tests and traffic-light summaries are educational model-
validation tools. They should not be described as complete Basel or FRTB
compliance.

## Quick start

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[app,dev]"
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Run the command-line analysis pipeline:

```bash
python run_demo.py
```

Run the optimization demo:

```bash
python run_phase5_optimization_demo.py
```

Run the regression suite without writing bytecode or pytest cache files:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  -p no:cacheprovider --disable-warnings -q
```

The current baseline contains 244 passing tests. This is a snapshot, not a
permanent quality guarantee; the CI workflow will become the authoritative
status once it is added.

## Data

The application supports CoinGecko and yfinance. The package also exposes a CSV
price loader for offline workflows. CoinGecko responses can be cached locally;
cache and downloaded data directories are excluded from version control.

Default demonstration assets and weights are configured in
[`configs/assets.yaml`](configs/assets.yaml). Analysis settings are in
[`configs/config.yaml`](configs/config.yaml).

Do not treat free-vendor data as production-grade market data. Verify timestamps,
missing observations, quote currencies, corporate-action adjustments for
traditional assets, and redistribution rights before publishing results.

## Architecture

```text
Data providers / CSV
        ↓
Cleaning and return construction
        ↓
Portfolio aggregation
        ↓
Risk models ─ Backtesting ─ Scenario simulation
        ↓                         ↓
Diagnostics                CVaR optimization
        ↓                         ↓
CLI outputs and Streamlit presentation
```

Financial calculations live under `src/var_cvar_crypto_risk/`. Streamlit is a
presentation and orchestration layer; it should not become the authoritative
location for financial formulas.

See [Architecture](docs/architecture.md) for module boundaries and current
technical debt.

## Repository structure

```text
.
├── app.py
├── configs/
├── data/                         # local data and cache; ignored
├── docs/
├── notebooks/
├── openspec/                     # specifications and historical change records
├── outputs/                      # generated results; ignored
├── src/var_cvar_crypto_risk/
├── tests/
├── run_demo.py
├── run_phase5_optimization_demo.py
└── pyproject.toml
```

## Methodology map

| Area | Implemented methods | Primary module |
|---|---|---|
| Returns | Simple, log, realized horizon returns | `returns.py` |
| VaR | Historical, Gaussian, Cornish-Fisher | `var_models.py` |
| CVaR | Historical, Gaussian | `cvar_models.py` |
| Validation | Kupiec, Christoffersen independence and CC | `backtesting.py` |
| Scenarios | Historical, multivariate Normal, multivariate Student-t | `monte_carlo.py` |
| Assumptions | Robust location, volatility, and covariance estimates | `assumptions.py` |
| Optimization | Minimum CVaR, CVaR cap, target return, frontier, Sharpe search | `optimization.py` |
| Dependence | Static, rolling, weighted, stress-versus-normal correlation | `correlation.py` |

Cornish-Fisher is a moment-based approximation, not a guaranteed improvement
over Gaussian or Historical VaR. Likewise, Student-t scenarios introduce
heavier tails but do not eliminate parameter or regime uncertainty.

See [Methodology and horizon conventions](docs/methodology.md).

## Reproducing results

Generated outputs are intentionally not committed. To recreate them, install
the project, review the YAML configurations, then run `python run_demo.py` or
the relevant Streamlit workflow.

A live vendor run is not perfectly reproducible because upstream data can be
revised and the end date can move. Publication-quality results require a pinned,
license-compatible input dataset plus a recorded configuration, data cutoff,
package version, and random seed.

See [Reproducibility](docs/reproducibility.md).

## Known limitations

- Historical crypto returns are non-stationary and may not represent the next
  market regime.
- The configured asset universe is not point-in-time and may contain selection
  or survivorship bias.
- Multi-day square-root/time-linear scaling assumes independent and identically
  distributed returns.
- Overlapping horizon observations reduce the effective sample size and are not
  appropriate for independence claims without qualification.
- Expected-return estimates are noisy and can dominate optimized allocations.
- Optimization outputs are in-sample estimates, not demonstrated out-of-sample
  strategies.
- Transaction costs, market impact, liquidity, taxes, and execution constraints
  are not modeled.
- Solver status alone does not prove economic feasibility; stronger residual
  checks are planned before `v1.0.0`.
- No portfolio-monitoring or rebalancing engine is included in version 0.5.0.

See [Model risk](docs/model-risk.md) for the complete interpretation framework.

## Current release status

Version `0.5.0` is a pre-1.0 research and development version. Work required
before `v1.0.0` includes:

- correcting degenerate Christoffersen-test interpretation;
- formalizing the signed VaR/CVaR output contract;
- adding covariance-repair and solver-residual governance;
- adding clean-install, artifact-build, and Streamlit smoke checks;
- adding GitHub CI and security automation;
- protecting architectural refactors with numerical regression fixtures.

See [CHANGELOG](CHANGELOG.md) and the specifications under `openspec/`.

## Documentation

- [Architecture](docs/architecture.md)
- [Methodology and horizons](docs/methodology.md)
- [Model risk](docs/model-risk.md)
- [Data provenance](docs/data-provenance.md)
- [Reproducibility](docs/reproducibility.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## License and disclaimer

Released under the [MIT License](LICENSE).

This software is provided for research and educational purposes only. It does
not constitute investment advice, a recommendation, a solicitation, or a
guarantee of risk or performance. Users are responsible for validating data,
assumptions, numerical results, legal requirements, and suitability for their
own use.
