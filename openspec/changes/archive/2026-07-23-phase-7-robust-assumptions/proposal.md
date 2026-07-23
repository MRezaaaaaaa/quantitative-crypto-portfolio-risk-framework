# Phase 7 — Robust Assumptions Engine & Optimizer Input Governance

## Why

The optimizer's outputs are highly sensitive to its inputs — expected
returns above all — but before Phase 7 those inputs were invisible: the
user could pick an estimator but never see the resulting per-asset
expected-return vector, the horizon convention was implicit, infeasible
runs said only "infeasible", and each tab recomputed its own base data.

## What changes

1. **New module `src/var_cvar_crypto_risk/assumptions.py`** — robust
   expected-return estimators (mean, median, trimmed mean, winsorized
   mean, shrinkage-to-zero, zero, manual-view blending), robust
   volatility (sample / winsorized / EWMA), robust covariance (sample /
   EWMA / shrinkage toward diagonal or constant-correlation), and an
   `AssumptionConfig` recipe dataclass.
2. **New 🧠 Robust Assumptions tab** — per-asset transparency tables for
   expected returns (all candidates + manual view + final), volatility,
   and covariance, all horizon-labelled, with CSV exports. The stored
   recipe is consumable by the Optimizer tab ("Robust Assumptions
   Engine" estimator) and is re-applied to the optimizer's own scenario
   matrix so horizons stay consistent.
3. **Optimizer governance** (`optimization.py`) —
   `compute_feasible_risk_return_bounds`, `diagnose_infeasibility`,
   `interpret_optimization_result`; zero-mu warnings on return-based
   objectives; Max-Sharpe `min_volatility` floor + cash-domination
   warning; `build_optimization_scenarios` mean/covariance overrides.
4. **Shared cached data layer** (`app.py`) — `st.cache_data`-backed
   horizon-return and scenario-matrix builders shared by the
   Distribution / MC / Assumptions / Optimizer tabs.
5. **Issue fixes** — horizon-convention labelling across tabs,
   asset-level VaR/CVaR annotations + table, mixed crypto/non-crypto
   loading (yfinance routing + ticker→symbol mapping + missing-symbol
   warnings), correlation interpretation (weighted average, stress vs
   normal, rolling-lag note), CVaR-cap / scenario-source / min-weight
   interpretation guidance.

## Out of scope (deferred)

* Black-Litterman / Entropy Pooling (the `AssumptionConfig` +
  `views.py` seam is where they plug in).
* Target-return sensitivity sweeps (now interpretable, revisit next).
* GARCH / copula scenario sources (Phase 6).
* Component CVaR / risk contribution (Phase 6).
