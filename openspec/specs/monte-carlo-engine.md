# Monte Carlo Scenario Engine

> **Status: Implemented — Phase 4 (`v0.4.0`)**

## Purpose

Add forward-looking risk estimation by simulating thousands of plausible
multi-asset return scenarios under parametric distributions and computing
scenario-based VaR / CVaR over the simulated distribution. The engine also
simulates portfolio value paths so the user can visualise plausible future
outcomes.

The engine sits between historical / parametric VaR (Phase 1) and the
backtesting machinery (Phase 3): users can compare Monte Carlo against
historical and parametric estimates in one table.

## Scope (implemented in Phase 4)

* `estimate_return_parameters` — mean vector, covariance, correlation,
  volatility from historical asset returns; optional annualisation
  (default crypto = 365 periods/year).
* `simulate_normal_returns` — multivariate Normal scenarios, with linear
  horizon scaling of mean and covariance.
* `simulate_student_t_returns` — multivariate Student-t scenarios with
  fat tails; df > 2; covariance rescaled by `(df-2)/df` so the resulting
  samples match the supplied covariance (asymptotically).
* `calculate_portfolio_scenario_returns` — weighted aggregation of asset
  scenarios into portfolio scenarios.
* `scenario_var` / `scenario_cvar` — empirical left-tail VaR / CVaR
  returned as **positive loss numbers**.
* `monte_carlo_risk_summary` — single-distribution summary table
  (Metric / Value / Unit), with optional dollar VaR / CVaR.
* `simulate_portfolio_paths` — Normal or Student-t portfolio value paths
  starting from an initial capital; shape `(horizon+1) × n_paths`.
* `compare_monte_carlo_distributions` — convenience wrapper that runs
  Normal and Student-t side-by-side on the same portfolio.
* `compare_all_risk_methods` — Historical, Gaussian, Cornish-Fisher,
  Normal MC, Student-t MC in a single DataFrame with `Horizon Days`
  and `Notes` columns. Cornish-Fisher CVaR is left as `NaN`.
* Visualisations: `plot_mc_loss_distribution`,
  `plot_mc_portfolio_paths`, `plot_normal_vs_student_t_distribution`,
  `plot_var_cvar_method_comparison`.
* Reproducibility: every simulator accepts an explicit `random_seed`;
  the same seed produces bitwise-identical samples.

## Non-functional requirements

* All risk numbers (VaR, CVaR, money equivalents) are **positive losses**.
* No live API calls — the module never imports `yfinance`, `coingecko`,
  or `streamlit`.
* Defensive Cholesky: jitter the diagonal on failure and retry.
* For `horizon_days > 1`, both Normal and Student-t scale mean and
  covariance linearly with the horizon (i.i.d. approximation).
* Crypto annualisation default = 365.

## Inputs

* `returns` — `pd.DataFrame` of asset returns (rows = dates).
* `weights` — `pd.Series` indexed by asset symbol; sum-to-one is not
  enforced inside Monte Carlo (the caller is expected to validate).
* `mean_vector`, `covariance_matrix` — typically produced by
  `estimate_return_parameters`.
* `n_scenarios` (default 5 000), `horizon_days` (default 1),
  `confidence_level` (default 0.95), `df` (default 5, must be > 2),
  `random_seed` (default 42).

## Outputs

* DataFrames of scenarios (`n_scenarios × n_assets`).
* Portfolio scenario `pd.Series` (`n_scenarios`).
* VaR / CVaR floats.
* Portfolio value paths DataFrame `(horizon+1) × n_paths` with the first
  row equal to `initial_value`.
* Comparison DataFrame for `compare_monte_carlo_distributions` and
  `compare_all_risk_methods`.
* PNG charts under `outputs/charts/`.
* CSVs under `outputs/tables/`.

## Edge cases

* Empty inputs raise `ValueError`.
* `df ≤ 2` raises `ValueError` (variance undefined).
* `n_scenarios ≤ 0`, `horizon_days < 1`, `initial_value ≤ 0` raise
  `ValueError`.
* Singular covariance matrices fall back to Cholesky with jitter.
* `compare_all_risk_methods` collects per-method errors as `NaN` rather
  than crashing the entire comparison.

## Acceptance criteria

1. Importing `var_cvar_crypto_risk.monte_carlo` does **not** pull in
   `streamlit` or `yfinance`.
2. Same `random_seed` → identical scenarios.
3. `CVaR ≥ VaR` for any non-degenerate scenario series.
4. `simulate_portfolio_paths` first row equals `initial_value` for every
   path.
5. `simulate_normal_returns(horizon_days=10)` has larger volatility than
   `simulate_normal_returns(horizon_days=1)` for the same seed.
6. `compare_all_risk_methods` includes all five methods and a
   `Horizon Days` column.
7. All required tables and charts are produced by
   `python run_demo.py`.

## Future extension points

* CVaR portfolio optimisation (Phase 5) consumes Monte Carlo scenarios
  as input to the linear program.
* Filtered Historical Simulation / GARCH-residual bootstrap (Phase 6).
* Stress testing and shock injection.
* Copula-based dependence modelling.
* Risk contribution decomposition (Euler / Component VaR).

## Dependencies

* Requires Phase 1 (returns, portfolio aggregation, VaR / CVaR
  abstractions).
* Reused inside the Phase 3 backtesting comparison via
  `compare_all_risk_methods`.
