# Quantitative Crypto Portfolio Risk Framework

[![Python](https://img.shields.io/badge/Python-3.10--3.13-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-1.0.0-informational)](CHANGELOG.md)
[![CI](https://github.com/MRezaaaaaaa/quantitative-crypto-portfolio-risk-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/MRezaaaaaaa/quantitative-crypto-portfolio-risk-framework/actions/workflows/ci.yml)
[![Security](https://github.com/MRezaaaaaaa/quantitative-crypto-portfolio-risk-framework/actions/workflows/security.yml/badge.svg)](https://github.com/MRezaaaaaaa/quantitative-crypto-portfolio-risk-framework/actions/workflows/security.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research-oriented Python and Streamlit framework for measuring, validating,
simulating, and optimizing crypto-portfolio tail risk.

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
- Validates and, when required, deterministically repairs covariance inputs
  before parametric simulation.
- Independently validates optimizer budget, bound, return, CVaR, and auxiliary
  constraint residuals instead of trusting solver status alone.
- Enforces local and Git-history public/private publication boundaries and runs
  SHA-pinned CodeQL and dependency-review workflows on GitHub.
- Generates and verifies deterministic offline publication bundles with pinned
  inputs, assumptions, cutoffs, source revisions, and artifact hashes.
- Exposes assumptions and diagnostics through an interactive Streamlit app.
- Applies a centralized return policy: Simple for portfolio/NAV/scenario/
  optimization calculations, with Log available only for advanced distribution
  diagnostics.

## Important interpretation

VaR identifies a loss quantile; CVaR estimates the average loss in the tail
beyond that quantile. Neither is a worst-case loss. Results depend materially on
the sample period, horizon, return convention, scenario model, covariance
estimate, expected-return estimate, and portfolio constraints.

The statistical tests and traffic-light summaries are educational model-
validation tools. They should not be described as complete Basel or FRTB
compliance.

## Quick start

Python 3.10 through 3.13 and uv 0.11.16 are required for the exact locked
environment.

```bash
uv sync --locked --extra app --extra dev
```

Run the Streamlit application:

```bash
uv run --locked --no-sync streamlit run app.py
```

Run the regression suite without writing bytecode or pytest cache files:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync python -m pytest \
  -p no:cacheprovider -q
```

The regression suite includes a cross-module numerical golden test. Local test
counts are validation snapshots rather than permanent quality guarantees;
GitHub Actions becomes the authoritative gate after the repository is pushed
and its workflow completes successfully.

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
├── openspec/                     # specifications and historical change records
├── outputs/                      # generated results; ignored
├── publication/                  # reproducible article experiments
├── scripts/                      # explicit maintenance utilities
├── src/var_cvar_crypto_risk/
├── tests/
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
| Covariance governance | Symmetry/PSD diagnostics and deterministic repair | `covariance.py` |
| Optimization | Minimum CVaR, CVaR cap, target return, frontier, Sharpe search | `optimization.py` |
| Dependence | Static, rolling, weighted, stress-versus-normal correlation | `correlation.py` |

Cornish-Fisher is a moment-based approximation, not a guaranteed improvement
over Gaussian or Historical VaR. Likewise, Student-t scenarios introduce
heavier tails but do not eliminate parameter or regime uncertainty.

See [Methodology and horizon conventions](docs/methodology.md).
See [Return conventions and calculation boundaries](docs/return-conventions.md)
for the exact Simple/Log routing policy.
The exact sign and unit rules are defined in the
[VaR and CVaR output contract](docs/risk-measure-contract.md).
Numerical acceptance rules are defined in
[Covariance and solver governance](docs/covariance-and-solver-governance.md).

## Reproducing results

General demo outputs are intentionally not committed. For a reviewed,
deterministic article workflow, generate the synthetic methodology bundle:

```bash
uv run --locked --no-sync python -m scripts.reproduce_publication \
  --config publication/configs/methodology_demo_v1.yaml \
  --output-dir publication/artifacts/methodology-demo-v1
```

The runner requires a clean Git tree, runs offline, checks the input hash and
cutoff, records solver diagnostics, and writes a manifest containing hashes for
every table and figure. See the
[publication experiment guide](publication/README.md). The bundled fixture is
synthetic; its outputs illustrate methodology and must not be reported as
real-market performance, prediction accuracy, or an investable strategy.

A live vendor run is not perfectly reproducible because upstream data can be
revised and the end date can move. Publication-quality results require a pinned,
license-compatible input dataset plus a recorded configuration, data cutoff,
package version, and random seed.

See [Reproducibility](docs/reproducibility.md).

Dependency declarations and the committed lockfile workflow are documented in
[Dependency management](docs/dependency-management.md).

## Known limitations

- Historical crypto returns are non-stationary and may not represent the next
  market regime.
- The configured asset universe is not point-in-time and may contain selection
  or survivorship bias.
- Multi-day square-root/time-linear scaling assumes independent and identically
  distributed returns.
- Scenario paths and optimization intentionally require simple-return
  arithmetic. Log returns are limited to advanced distribution diagnostics and
  are converted exactly when portfolio-level diagnostic returns are required.
- Overlapping horizon observations reduce the effective sample size and are not
  appropriate for independence claims without qualification.
- Expected-return estimates are noisy and can dominate optimized allocations.
- Optimization outputs are in-sample estimates, not demonstrated out-of-sample
  strategies.
- Transaction costs, market impact, liquidity, taxes, and execution constraints
  are not modeled.
- Solver success is accepted only after independent numerical residual checks;
  passing those checks still does not prove economic optimality or future
  feasibility.
- No portfolio-monitoring or rebalancing engine is included in version 1.0.0.

See [Model risk](docs/model-risk.md) for the complete interpretation framework.

## Current release status

Version `1.0.0` is the first public research release. The public repository
requires the supported Python test matrix, coverage/build/app checks, OpenSpec,
CodeQL, and Dependency Review before protected-branch updates. Secret Scanning,
Push Protection, Dependabot security updates, and Private Vulnerability
Reporting are enabled.

The included publication workflow remains a synthetic methodology
demonstration. A separately licensed and pinned real-market dataset is required
before an article makes market-specific empirical or performance claims.
Portfolio monitoring and rebalancing remain planned post-1.0 capabilities.

The local and CI test suites enforce an 80% package coverage floor. Coverage is
a regression guard, not evidence that the financial models are correct.

See [CHANGELOG](CHANGELOG.md) and the specifications under `openspec/`.

## Documentation

- [Architecture](docs/architecture.md)
- [Methodology and horizons](docs/methodology.md)
- [Return conventions](docs/return-conventions.md)
- [VaR and CVaR output contract](docs/risk-measure-contract.md)
- [Covariance and solver governance](docs/covariance-and-solver-governance.md)
- [Model risk](docs/model-risk.md)
- [Data provenance](docs/data-provenance.md)
- [Reproducibility](docs/reproducibility.md)
- [Dependency management](docs/dependency-management.md)
- [Public release checklist](docs/public-release-checklist.md)
- [Reproducible publication experiments](publication/README.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## License and disclaimer

Released under the [MIT License](LICENSE).

This software is provided for research and educational purposes only. It does
not constitute investment advice, a recommendation, a solicitation, or a
guarantee of risk or performance. Users are responsible for validating data,
assumptions, numerical results, legal requirements, and suitability for their
own use.
