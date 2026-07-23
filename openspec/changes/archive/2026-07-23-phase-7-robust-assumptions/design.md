# Phase 7 design notes

## Assumption recipe, not assumption values

The Robust Assumptions tab stores an `AssumptionConfig` (estimator name +
parameters + manual views + covariance recipe), **not** a frozen
expected-return vector. When the Optimizer tab selects the "Robust
Assumptions Engine" estimator, the recipe is re-applied to the
optimizer's own scenario matrix. This guarantees the expected returns are
always per the optimizer's horizon and source — a stored vector built on
7-day historical scenarios would silently mismatch a 1-day Student-t run.

The one exception is the covariance override: covariance is estimated
from **daily** returns and passed into the MC scenario builders, which
own the ×h horizon scaling (same convention as `estimate_return_parameters`).

## Estimator conventions

* All expected-return estimators operate column-wise on a generic
  `(n_obs × n_assets)` frame; the caller owns horizon labelling.
* EWMA uses the RiskMetrics zero-mean squared-return convention with
  normalized weights `w_i ∝ λ^age`.
* Covariance shrinkage is plain linear shrinkage `(1-δ)S + δT` with a
  user-chosen δ (no auto-δ estimation) — transparent and dependency-free
  (no sklearn).

## Governance functions

* `diagnose_infeasibility` checks cheap algebraic causes first (min/max
  weight budget) and only then solves the two bounding LPs (min-CVaR,
  max-return) to test the CVaR cap and target return against what is
  achievable. It returns strings, not exceptions — the UI stays alive.
* `interpret_optimization_result` is pure post-processing of a result
  dict; the defensive/balanced/aggressive profile is the portfolio
  CVaR's position within the feasible [min_cvar, max_return_cvar] range.
* Optimizers never raise on infeasibility (unchanged invariant); the new
  `warning` key is additive and optional.

## Caching

`st.cache_data` keys on function arguments (content hash for pandas
objects). Base data still lives in `session_state["risk_results"]`
(computed on the Run click); the cache layer covers the two derived
artifacts that were recomputed per tab/rerun: horizon returns and
scenario matrices. Changing any input produces a new cache key, so there
is no stale-cache path; a new data run replaces the session-state frames
and thereby all downstream keys.

## Mixed-asset loading

`_fetch_prices` routes assets without a CoinGecko ID to yfinance even
when CoinGecko is the primary source, renames yfinance columns through
an explicit ticker→Symbol map (handling the client's `-USD` stripping),
outer-joins the frames, and lets `clean_price_data` align to the common
trading calendar (weekends drop when a TradFi asset is present). Missing
symbols produce a visible warning instead of a silent drop. The cache
signature changed to `(prices, used_source, warnings)`.
