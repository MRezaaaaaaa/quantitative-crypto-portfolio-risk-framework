# Core Risk Engine — OpenSpec

> **Status: Implemented — Phase 1**

## Purpose

Provide a flexible, modular, production-quality VaR/CVaR risk engine for crypto
portfolios that can be extended with backtesting, Monte Carlo, optimization,
stress testing, and risk contribution modules in subsequent phases.

## Scope

In scope for Phase 1:

- Configurable data ingestion (CoinGecko primary, yfinance fallback, CSV manual).
- Price preprocessing, alignment, and validation.
- Asset and portfolio-level return computation (simple and log).
- VaR via Historical, Gaussian, and Cornish-Fisher methods.
- CVaR via Historical and Gaussian (analytic) methods.
- Portfolio-level distribution statistics, drawdown, and Sharpe.
- Tabular and chart export to `outputs/`.

Explicitly out of scope for Phase 1:

- Streamlit dashboards.
- Monte Carlo simulation.
- CVaR portfolio optimization.
- Backtesting / coverage tests.
- Stress testing.
- Risk contribution / decomposition.

## Functional Requirements

1. Load configuration from `configs/config.yaml` and `configs/assets.yaml`.
2. Validate that all required configuration keys are present.
3. Fetch daily prices for each asset from the configured source.
4. Fall back from CoinGecko to yfinance automatically if CoinGecko fails and a
   fallback source is configured.
5. Validate price data (DatetimeIndex, ascending, no duplicates, all positive).
6. Compute simple or log returns according to configuration.
7. Validate portfolio weights against config and produce normalized weights.
8. Compute the daily portfolio return as the weighted sum of asset returns.
9. Compute VaR for every method listed in `risk.var_methods`.
10. Compute CVaR for every method listed in `risk.cvar_methods`.
11. Convert percentage VaR/CVaR to monetary values using `initial_capital`.
12. Generate a structured risk summary `(Metric, Value, Unit)` DataFrame.
13. Plot the return distribution with VaR threshold and CVaR tail shading.
14. Plot cumulative returns and drawdown over time.
15. Persist all tables and charts to the configured output directory.

## Non-Functional Requirements

- **Performance.** End-to-end demo on three assets and ~5 years of daily data
  must complete in well under a minute on a developer laptop.
- **Maintainability.** No calculation logic in scripts; ABCs for VaR/CVaR
  models; configuration-driven inputs; no hard-coded asset names or paths in
  calculation functions.
- **Testability.** Every calculation module has unit tests with deterministic
  fixtures. Tests must not require network access.

## Inputs

- `configs/config.yaml` — engine configuration.
- `configs/assets.yaml` — asset metadata including weights.
- Optional environment variable `COINGECKO_API_KEY`.
- Optional CSV file when `data.source = "csv"`.

## Outputs

- `outputs/tables/price_data.csv`
- `outputs/tables/asset_returns.csv`
- `outputs/tables/portfolio_returns.csv`
- `outputs/tables/portfolio_value.csv`
- `outputs/tables/risk_summary.csv`
- `outputs/charts/return_distribution_var_cvar.png`
- `outputs/charts/cumulative_returns.png`
- `outputs/charts/drawdown.png`

## Edge Cases

- **Empty data.** `validate_price_data` raises `ValueError`.
- **Single asset.** Pipeline must still produce returns, portfolio returns, and
  risk metrics; weights must sum to 1.
- **Very short history.** Cornish-Fisher VaR requires at least four
  observations; raise `ValueError` otherwise.
- **All-negative returns.** Max drawdown is the sample minimum cumulative
  return; VaR/CVaR remain well-defined positive numbers.
- **CoinGecko 429.** The client waits 60 seconds and retries once.

## Acceptance Criteria

1. Project installs successfully via `pip install -e .` and `pip install -r requirements.txt`.
2. `python run_phase1_demo.py` runs end-to-end and exits with status 0.
3. CoinGecko, yfinance fallback, and CSV loading are implemented.
4. Risk metrics are reported at the portfolio level.
5. Historical, Gaussian, and Cornish-Fisher VaR are implemented.
6. Historical and Gaussian CVaR are implemented.
7. Outputs are persisted to `outputs/tables` and `outputs/charts`.
8. Pytest suite passes.
9. README is complete and professional.
10. OpenSpec files document future upgrade paths.
11. No Streamlit, Monte Carlo, optimization, or backtesting code is present in Phase 1.
12. Modular architecture is ready for Phase 2 extension without rewrites.

## Future Extension Points

- **Phase 2 (Streamlit).** Imports the public API from
  `var_cvar_crypto_risk.__init__` and renders the same risk summary
  interactively. No engine changes required.
- **Phase 3 (Backtesting).** New module `backtesting.py` consumes
  `portfolio_returns` and a `VaRModel` instance, runs rolling-window VaR, and
  performs Kupiec / Christoffersen tests.
- **Phase 4 (Monte Carlo).** New `MonteCarloVaR(VaRModel)` and
  `MonteCarloCVaR(CVaRModel)` subclasses plug into `_DISPATCH` dictionaries.
- **Phase 5 (Optimization).** New `optimization.py` consumes asset returns and
  emits weights, which are then validated by the existing `validate_weights`.
- **Phase 6 (Advanced).** GARCH and copula models extend the VaR/CVaR ABCs;
  no consumer code changes are needed.

## Phase 5.5 additions

- **Horizon returns.** `returns.calculate_horizon_returns(returns, horizon_days,
  method, overlapping)` aggregates daily returns into h-day returns (simple ⇒
  `prod(1+r)-1`, log ⇒ `sum(r)`), overlapping (rolling) or non-overlapping blocks.
  Used to make the distribution view horizon-matched.
- **Asset-level drawdowns.** `risk_metrics.calculate_asset_drawdowns(asset_returns)`
  applies the drawdown formula per asset (values in `[-1, 0]`).
- **Correlation analytics.** new `correlation` module:
  `calculate_correlation_matrix` (pearson / spearman) and
  `calculate_rolling_average_correlation` (mean off-diagonal correlation through
  time).
- **Risk-free helper.** `utils.annual_to_horizon_rate(annual_rate, horizon_days,
  day_count)`.
