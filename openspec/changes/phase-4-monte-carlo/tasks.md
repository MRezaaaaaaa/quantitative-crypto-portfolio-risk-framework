# Phase 4 — Monte Carlo & horizon-aware backtesting — task list

## A. Fix horizon-aware backtesting

- [x] Add `calculate_realized_horizon_returns(returns, horizon_days, method)`
      in `backtesting.py` (`simple` and `log` aggregation, last `h`
      observations dropped).
- [x] Make `rolling_var_forecast` horizon-aware (`horizon_days`,
      `return_method` arguments). Preserve existing behaviour for
      `horizon_days=1`. No look-ahead bias.
- [x] Add `horizon_days`, `method` columns to the rolling forecast
      DataFrame.
- [x] Propagate `horizon_days` through `backtest_var_model`,
      `compare_var_models_backtest`, `create_backtesting_report_table`
      (adds a "Horizon (days)" column).
- [x] Wire `horizon_days` into the Streamlit Backtesting tab and label
      every chart/table with the chosen horizon.
- [x] Update `tests/test_backtesting.py` with horizon-aware tests
      (simple/log realised returns, horizon column propagation, strict
      no-look-ahead for h>1).

## B. Monte Carlo module

- [x] Create `src/var_cvar_crypto_risk/monte_carlo.py`.
- [x] `estimate_return_parameters` (mean/cov/corr/vol, optional
      annualisation).
- [x] `simulate_normal_returns`.
- [x] `simulate_student_t_returns` (df > 2; covariance rescaled by
      `(df-2)/df`).
- [x] `calculate_portfolio_scenario_returns`.
- [x] `scenario_var`, `scenario_cvar` (positive losses; CVaR ≥ VaR).
- [x] `monte_carlo_risk_summary` (Metric / Value / Unit; optional
      money VaR/CVaR).
- [x] `simulate_portfolio_paths` (Normal or Student-t; first row =
      initial_value).
- [x] `compare_monte_carlo_distributions` (Normal vs Student-t).
- [x] `compare_all_risk_methods` (Historical / Gaussian / Cornish-Fisher
      / Normal MC / Student-t MC; Horizon Days + Notes columns).

## C. Plotting

- [x] `plot_mc_loss_distribution`.
- [x] `plot_mc_portfolio_paths` (sub-sampled paths + mean overlay).
- [x] `plot_normal_vs_student_t_distribution`.
- [x] `plot_var_cvar_method_comparison`.

## D. Streamlit integration

- [x] New `🎲 Monte Carlo Scenario Engine` tab.
- [x] Widgets: distribution, n_scenarios, horizon, confidence, df,
      seed, n_paths, path horizon.
- [x] KPI cards (MC VaR, MC CVaR, mean return, worst, scenarios).
- [x] Distribution chart, portfolio paths chart, comparison table.
- [x] Optional Normal vs Student-t comparison view.
- [x] Session-state persistence (results survive reruns from other tabs).
- [x] Friendly error handling when covariance has numerical issues.

## E. Demo, config, hygiene

- [x] Extend the single `run_demo.py` with Step 3 — Monte Carlo (no
      separate Phase-4 script per project convention).
- [x] Add `monte_carlo:` section and `default_horizon_days` /
      `horizon_aware` keys to `configs/config.yaml`.
- [x] Bump project version to `0.4.0`.
- [x] Ensure no Phase 1–3 functionality regresses.

## F. Tests

- [x] New `tests/test_monte_carlo.py` covering all module entry points.
- [x] Import-isolation guard (monte_carlo must not import streamlit;
      backtesting must not import yfinance).
- [x] All tests pass via `pytest`.

## G. Documentation

- [x] Rewrite `openspec/specs/monte-carlo-engine.md`.
- [x] Add `openspec/changes/phase-4-monte-carlo/` (proposal, tasks,
      design).
- [x] Add a Phase 4 section to `README.md` covering motivation, demo
      command, output files, and limitations.

## Required output files

CSVs under `outputs/tables/`:

- `mc_scenarios_normal.csv`
- `mc_scenarios_student_t.csv`
- `mc_portfolio_returns_normal.csv`
- `mc_portfolio_returns_student_t.csv`
- `mc_risk_summary.csv`
- `model_risk_comparison.csv`

PNGs under `outputs/charts/`:

- `mc_loss_distribution_normal.png`
- `mc_loss_distribution_student_t.png`
- `mc_portfolio_paths.png`
- `normal_vs_student_t_mc.png`
- `var_cvar_method_comparison.png`

Phase 1 + Phase 3 outputs must continue to be generated.

## Out of scope

- CVaR portfolio optimisation (Phase 5).
- GARCH / FHS (Phase 6).
- Copulas (Phase 6).
- Stress testing.
- Risk contribution.
