# Phase 7 tasks

- [x] `assumptions.py`: expected-return estimators (mean / median /
      trimmed / winsorized / shrinkage / zero), manual-view blending via
      `views.py`, `AssumptionConfig`, transparency tables
- [x] `assumptions.py`: robust volatility (sample / winsorized / EWMA)
      and covariance (sample / EWMA / shrinkage) estimators
- [x] `optimization.py`: `compute_feasible_risk_return_bounds`,
      `diagnose_infeasibility`, `interpret_optimization_result`
- [x] `optimization.py`: zero-mu warnings; Max-Sharpe volatility floor +
      cash-domination warning; scenario-builder mean/cov overrides
- [x] `correlation.py`: weighted average correlation, stress-vs-normal
      conditional correlation
- [x] `plotting.py`: VaR/CVaR values annotated on asset distributions
- [x] `app.py`: 🧠 Robust Assumptions tab (estimator recipe, tables,
      covariance heatmap, exports)
- [x] `app.py`: shared cached horizon-return / scenario-matrix layer
- [x] `app.py`: optimizer governance panel, constraint-applicability
      matrix, interpretation expanders, per-result diagnostics
- [x] `app.py`: mixed-asset loader (yfinance routing + symbol mapping +
      warnings); horizon-convention labels; asset risk table;
      correlation interpretation metrics
- [x] Tests: `test_assumptions.py` (30), `test_optimizer_governance.py`
      (15), `test_correlation_phase7.py` (5) — suite: 244 passed
- [ ] Follow-up: target-return sensitivity sweep (now interpretable)
- [ ] Follow-up: Black-Litterman / Entropy Pooling behind
      `AssumptionConfig`
- [ ] Follow-up: live-network test for mixed crypto + GLD/SPY loading
      (logic unit-covered; end-to-end fetch needs network)
