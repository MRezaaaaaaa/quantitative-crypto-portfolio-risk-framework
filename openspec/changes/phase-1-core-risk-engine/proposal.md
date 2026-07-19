# Phase 1 — Core Risk Engine: Proposal

## Why this phase is needed

Crypto portfolios face fat tails, regime shifts, and liquidity risks that
classical equity-derived risk models routinely understate. Before we can build
a Streamlit dashboard, run Monte Carlo, optimize a CVaR-constrained portfolio,
or backtest model coverage, we need a clean, well-tested **core risk engine**
that:

- Loads data from multiple sources interchangeably.
- Computes returns and portfolio aggregates correctly and reproducibly.
- Implements the standard VaR and CVaR methodologies with consistent sign
  conventions.
- Exposes a stable Python API that future phases can wrap rather than rewrite.

Phase 1 establishes that engine. Every later phase depends on it.

## What will be built

- A configuration-driven Python package `var_cvar_crypto_risk` exposing a
  small public API.
- CoinGecko and yfinance clients, plus a CSV loader, with a unified data
  loader that picks the source based on configuration.
- Returns, portfolio aggregation, and abstract-base-class-driven VaR/CVaR
  models with three VaR variants (Historical, Gaussian, Cornish-Fisher) and
  two CVaR variants (Historical, Gaussian).
- Risk summary table, distribution chart with VaR/CVaR overlays, cumulative
  return chart, and drawdown chart.
- End-to-end demo script (`run_phase1_demo.py`) that orchestrates the entire
  pipeline and prints a polished console summary.
- Pytest test suite covering returns, portfolio, VaR, CVaR, data loader, and
  risk metrics.
- README, OpenSpec specifications, and placeholder spec files for Phases 2-6.

## What is intentionally excluded from Phase 1 and why

- **Streamlit dashboard** — depends on a stable engine API; deferring it
  prevents premature coupling between calculations and UI.
- **Monte Carlo simulation** — orthogonal to the deterministic core; landing
  it later avoids polluting the core with simulation infrastructure.
- **CVaR portfolio optimization** — requires solver dependencies and a
  separate validation surface; building it on top of a known-correct engine
  is far cheaper than building both at once.
- **Backtesting / coverage tests** — only meaningful once VaR models are
  trusted at a single point in time; rolling windows come next.
- **Stress testing and risk contribution** — useful but additive; they consume
  the core engine and can be added without breaking the public API.
