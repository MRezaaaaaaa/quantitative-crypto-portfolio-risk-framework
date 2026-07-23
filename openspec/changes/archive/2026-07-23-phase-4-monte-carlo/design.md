# Phase 4 — Design notes

## 1. Data flow

```
prices  ───►  asset_returns  ──┬──►  portfolio_returns
                               │
                               ├──►  estimate_return_parameters
                               │         │
                               │         ▼
                               │     mean_vector, covariance_matrix
                               │         │
                               │         ├──►  simulate_normal_returns ─┐
                               │         └──►  simulate_student_t_returns ┘
                               │                              │
                               │                              ▼
                               │     calculate_portfolio_scenario_returns
                               │                              │
                               │                              ▼
                               │                  scenario_var / scenario_cvar
                               │                              │
                               │                              ▼
                               │                  monte_carlo_risk_summary
                               │
                               └──►  compare_all_risk_methods
                                         (Historical / Gaussian / CF / Normal MC / Student-t MC)
```

The Streamlit Monte Carlo tab orchestrates only the UI — every numeric
computation lives inside `monte_carlo.py`.

## 2. Scenario generation logic

### Normal

For `horizon_days = h`:

```
μ_h = μ * h
Σ_h = Σ * h
L   = chol(Σ_h)
Z   ~ N(0, I)_{n × d}
X   = μ_h + Z @ Lᵀ
```

### Student-t

```
μ_h         = μ * h
Σ_h         = Σ * h
Σ_scaled    = Σ_h * (df - 2) / df          # variance correction
L           = chol(Σ_scaled)
Z           ~ N(0, I)_{n × d}
U           ~ chi2(df) / df  (per scenario)
X           = μ_h + (Z @ Lᵀ) / √U
```

The variance correction means the marginal covariance of `X`
converges to `Σ_h` as `n_scenarios → ∞`. The Student-t draws still
produce heavier tails than the matching Normal — which is the whole
point.

## 3. Covariance estimation

`estimate_return_parameters` uses `pd.DataFrame.cov()` and `.std(ddof=1)`
on the dropna-cleaned returns. Optional annualisation multiplies
mean and covariance by `periods_per_year` (default 365 for crypto)
and volatility by `√periods_per_year`.

Singular or near-singular matrices are handled by `_safe_cholesky`,
which adds increasing diagonal jitter (`10^k × 10⁻¹⁰`) for up to 10
retries before re-raising.

## 4. Horizon treatment

* **Monte Carlo simulators** scale `μ` and `Σ` linearly with horizon
  (i.i.d. approximation). This is internally consistent with the way
  the simulator is used as input to `compare_all_risk_methods`.
* **Historical / Gaussian / Cornish-Fisher** rows in
  `compare_all_risk_methods` aggregate the portfolio return series
  into rolling h-day returns and compute VaR / CVaR on those — *not*
  by sqrt-time scaling. The `Notes` column records this.
* **Backtesting** uses the same convention: at forecast date `t`,
  lookback is `returns[t-window : t]`; for `horizon_days > 1`, the
  lookback is first aggregated into rolling h-day returns, then VaR
  is estimated. The realised value is `prod(1 + r[t..t+h-1]) − 1`
  starting at `t` (so `horizon_days=1` collapses to `r_t`, preserving
  the Phase 3 behaviour bit-for-bit).

## 5. Random seed & reproducibility

Every simulator accepts `random_seed: int | None`. With the same seed,
samples are identical (verified by
`test_simulate_normal_returns_reproducible`). The default is `42`.

## 6. Separation of calculation and UI

* `monte_carlo.py` imports only `numpy`, `pandas`, and the project's
  `var_models` / `cvar_models`. It must **not** import `streamlit`,
  `yfinance`, or any plotting library. Enforced via
  `test_monte_carlo_does_not_import_streamlit`.
* `plotting.py` owns every matplotlib figure.
* `app.py` only orchestrates widgets and renders artefacts produced by
  the analytics module.

## 7. Connection to future phases

* **CVaR optimisation (Phase 5):** the LP for min-CVaR portfolios
  consumes a scenario matrix of asset returns. `simulate_normal_returns`
  and `simulate_student_t_returns` already produce that exact shape
  (`n_scenarios × n_assets`).
* **GARCH / FHS (Phase 6):** these will be wired in as alternative
  scenario generators behind the same `simulate_*` interface, so
  consumers (UI, demo, comparison) won't change.
* **Risk contribution:** can be computed from the existing scenario
  arrays without modifying the engine.

## 8. Error semantics

The engine raises `ValueError` for all bad inputs. Higher-level
orchestrators (`run_demo.py`, the Streamlit tab) catch the exception,
record it (CSV row's `Notes` column, or `st.error`), and continue —
never crash the whole comparison.
