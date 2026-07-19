# Phase 4 — Monte Carlo Scenario Engine

## Why

Phase 1 gives historical and parametric VaR/CVaR, Phase 3 validates those
forecasts against realised returns. Both are backward-looking. To answer
the question *"what could plausibly happen next?"* we need a forward
simulator that:

1. Captures the joint behaviour of multiple crypto assets — not just
   each asset in isolation.
2. Allows fat-tailed distributions, since crypto markets routinely
   produce shocks far beyond Gaussian expectations.
3. Supports multi-day horizons directly, instead of relying solely on
   square-root-of-time scaling.

Additionally, Phase 3 currently backtests **daily one-step-ahead** VaR
even when the user has selected `time_horizon_days > 1` for the
headline cards. This is an inconsistency we fix in the same release.

## What we'll do

1. **Fix horizon-aware backtesting** (Phase 3 cleanup).
   * Add `calculate_realized_horizon_returns(returns, horizon_days, method)`.
   * Make `rolling_var_forecast` horizon-aware: for `horizon_days > 1`,
     the lookback is aggregated into rolling h-day returns and VaR is
     estimated from those h-day returns; the realised value is the
     realised h-day forward return. Horizon=1 behaviour is preserved.
   * Propagate `horizon_days` into `backtest_var_model`,
     `compare_var_models_backtest`, `create_backtesting_report_table`.
   * Wire the horizon into the Streamlit backtesting tab.

2. **Create `monte_carlo.py`** — pure analytics, no plotting, no UI.
   * `estimate_return_parameters`, `simulate_normal_returns`,
     `simulate_student_t_returns`, `calculate_portfolio_scenario_returns`,
     `scenario_var`, `scenario_cvar`, `monte_carlo_risk_summary`,
     `simulate_portfolio_paths`, `compare_monte_carlo_distributions`,
     `compare_all_risk_methods`.

3. **Plotting helpers** in `plotting.py`:
   `plot_mc_loss_distribution`, `plot_mc_portfolio_paths`,
   `plot_normal_vs_student_t_distribution`,
   `plot_var_cvar_method_comparison`.

4. **Streamlit** — add a `🎲 Monte Carlo Scenario Engine` tab with the
   widgets, KPI cards, charts and comparison table described in the spec.

5. **`run_demo.py`** — add Phase 4 as Step 3 of the unified pipeline.

6. **Tests** — `tests/test_monte_carlo.py` plus horizon-aware additions
   in `tests/test_backtesting.py`.

7. **Config** — extend `configs/config.yaml` with a `monte_carlo:`
   section and add `default_horizon_days` / `horizon_aware` to the
   `backtesting:` section.

8. **Docs** — update README and OpenSpec.

## Why Student-t for crypto fat tails

Crypto daily-return distributions exhibit excess kurtosis far above the
Gaussian baseline. A multivariate Student-t with `df ≈ 4–6` reproduces
the kurtosis and tail thickness empirically observed in BTC, ETH, SOL
returns much better than a multivariate Normal. Using the standard
covariance rescaling `cov * (df-2)/df`, the simulated samples retain
the historical covariance while exhibiting heavier tails — so VaR and
CVaR estimates produced from Student-t scenarios are more conservative
in the tail regions that actually drive risk.

## Out of scope (deferred to future phases)

* CVaR portfolio optimisation / efficient frontier (Phase 5).
* GARCH or filtered historical simulation (Phase 6).
* Copula-based joint modelling (Phase 6).
* Stress testing / scenario injection.
* Risk contribution / component VaR.

## Acceptance

Phase 4 is complete when every item in `tasks.md` is ticked, all tests
pass via `pytest`, and `python run_demo.py` produces every CSV/PNG
listed in `tasks.md` without error.
