# CVaR Portfolio Optimization

> **Status: Implemented — Phase 5 (v0.5.0)**
> Provides scenario-based portfolio optimization built on top of the
> Phase-4 Monte Carlo engine.

## Purpose

Turn the project from a pure measurement / simulation platform into a
**portfolio decision-making** tool. Given a scenario return matrix
(historical or Monte Carlo), the optimization layer solves for portfolio
weights that minimize tail risk, maximise return under a tail-risk cap,
or balance the two along an efficient frontier.

## Scope (delivered)

* Linear-programming reformulation of CVaR optimisation
  (Rockafellar & Uryasev, 2000).
* Three discrete optimisers:
  1. **Minimum CVaR** — minimise CVaR with full investment + box
     constraints.
  2. **Maximum expected return under a CVaR cap** —
     `max μᵀw  s.t.  CVaR(w) ≤ L`.
  3. **Minimum CVaR for a target return** —
     `min CVaR(w)  s.t.  μᵀw ≥ r*`.
* CVaR efficient frontier sweep across target returns.
* Multiple scenario sources: historical, Normal Monte Carlo, Student-t
  Monte Carlo (sharing the Phase-4 simulator).
* Optional cash asset injected into the scenario matrix at a constant
  per-horizon return.
* Long-only / short-allowed switch with explicit min/max weight box
  constraints.
* Current vs optimised comparison table with money VaR / CVaR.
* Streamlit UI tab with scenario, objective, and constraint controls;
  KPI cards; weights tables and charts; comparison and frontier
  visualisations; downloadable CSV / PNG artefacts.
* Command-line demo `run_phase5_optimization_demo.py`.
* OpenSpec docs (this file + `phase-5-cvar-optimization/`) and README
  Phase-5 section.

## Mathematical formulation

Let `R ∈ R^{N × d}` be the scenario return matrix (rows = scenarios,
columns = assets) and let `w ∈ R^d` be the weights vector. Define
portfolio loss per scenario as `ℓ_i(w) = −R_i w`. For confidence level
`β`:

```
minimize_{w, t, u}   t + (1 / ((1 − β) · N)) · Σ_i u_i
subject to           u_i ≥ ℓ_i(w) − t      ∀ i
                     u_i ≥ 0               ∀ i
                     Σ_j w_j = 1
                     w_j ∈ [w_min, w_max]    (or w_j ≥ 0 for long-only)
```

The optimal `t` is the VaR threshold; the objective value is CVaR.

For the **CVaR-capped** problem we maximise `μᵀw` subject to the same
auxiliary structure plus `t + (1 / ((1 − β) · N)) · Σ_i u_i ≤ L`.

For the **target-return** problem we minimise the CVaR expression
subject to `μᵀw ≥ r*`.

## Inputs

* `scenario_returns: pd.DataFrame` (rows = scenarios, cols = assets).
* `expected_returns: pd.Series | None` — defaults to scenario column
  means; can be supplied externally (e.g. analyst views, shrinkage).
* `confidence_level: float ∈ (0, 1)`.
* Constraint knobs: `long_only`, `min_weight`, `max_weight`,
  `include_cash`, `cash_return`.
* Solver knob: `solver` (default auto; prefers ECOS → CLARABEL → SCS).

## Outputs

Per optimiser:

```python
{
  "status":          str,              # 'optimal' | 'infeasible' | ...
  "objective_value": float,
  "weights":         pd.Series,        # indexed by asset
  "expected_return": float,
  "VaR":             float,
  "CVaR":            float,
  "volatility":      float,
  "solver":          str,
  "message":         str,
  ... # plus objective-specific extras (cvar_limit, target_return)
}
```

Frontier output: `pd.DataFrame` with `target_return`, `expected_return`,
`volatility`, `VaR`, `CVaR`, `status`, and one `weight_<ASSET>` column
per asset.

Saved artefacts (under `outputs/`):

```
tables/optimized_weights_min_cvar.csv
tables/optimized_weights_target_return.csv
tables/optimized_weights_cvar_cap.csv
tables/optimization_summary.csv
tables/portfolio_comparison.csv
tables/cvar_efficient_frontier.csv
charts/optimized_weights_min_cvar.png
charts/current_vs_optimized_risk.png
charts/cvar_efficient_frontier.png
charts/portfolio_allocation_comparison.png
```

## Functional requirements

| # | Requirement |
|---|-------------|
| F1 | Validates the scenario matrix (numeric, ≥ 2 scenarios, no NaN / inf). |
| F2 | Supports historical, Normal MC, and Student-t MC scenario sources via `build_optimization_scenarios`. |
| F3 | Historical source aggregates rolling h-day simple returns when `horizon_days > 1`. |
| F4 | Long-only constraint forces `w ≥ 0`; short-allowed widens to `min_weight ≤ w ≤ max_weight`. |
| F5 | Optional cash asset added as a constant-return column before optimisation. |
| F6 | Min-CVaR, max-return-under-CVaR-cap, and min-CVaR-for-target-return all return a uniform result dict. |
| F7 | Infeasible problems return `status != "optimal"` and a human-readable `message` instead of raising. |
| F8 | Efficient frontier sweeps target returns between the unconstrained min-CVaR and the unconstrained max-return endpoint and skips infeasible points. |
| F9 | `compare_current_vs_optimized` computes metrics for the current portfolio and every optimised portfolio using the same scenario matrix and confidence level. |
| F10 | Streamlit tab presents widgets, KPI cards, tables, charts, and download buttons; the optimisation module never imports Streamlit. |
| F11 | All numeric outputs follow the project convention: VaR / CVaR as positive loss decimals. |

## Non-functional requirements

* **No commercial solvers.** Solver preference: ECOS → CLARABEL → SCS.
  Module imports CVXPY lazily so callers that only need the validation
  / builder helpers don't pay the import cost.
* **Pure analytics.** No I/O, no plotting, no Streamlit in
  `optimization.py` — enforced by
  `test_optimization_import_without_streamlit`.
* **Reproducibility.** Same `random_seed` → same Monte Carlo scenario
  matrix → same optimal weights.
* **Backwards compatibility.** No changes to Phase 1-4 public APIs.

## Edge cases

| Case | Handling |
|------|----------|
| All-cash portfolio in scenario matrix | Treated as a regular asset; LP still respects budget = 1. |
| Singular covariance matrix in MC source | Phase-4 `_safe_cholesky` falls back to jittered Cholesky. |
| Infeasible target return / CVaR cap | Returns `status = "infeasible"`; demo and Streamlit show the message and continue. |
| Long-only + negative `min_weight` | Effective lower bound is `max(0, min_weight)`. Streamlit warns the user. |
| Empty / degenerate scenario matrix | `validate_scenario_matrix` raises `ValueError` before solver runs. |
| All-NaN or non-finite column | `validate_scenario_matrix` raises. |
| Solver failure / numerical issue | `_solve` falls back through `CLARABEL → SCS → default`; final fallback returns a `solver_error` result instead of raising. |

## Acceptance criteria

1. `pytest` succeeds with ≥ 16 new tests covering every optimisation
   entry point and the scenario builder.
2. `python run_phase5_optimization_demo.py` produces every CSV / PNG
   listed under *Saved artefacts* without error.
3. Streamlit app shows a working **🎯 Portfolio Optimization** tab.
4. The min-CVaR LP returns weights with `Σ w = 1`, `w ≥ 0` (long-only),
   `w ≤ max_weight`, and `CVaR ≤ CVaR(current)` on typical inputs.
5. `compare_current_vs_optimized` returns one row per portfolio with all
   metric columns populated for solved cases and the `Status` column
   reflecting the solver status otherwise.

## Future extensions

* **Risk contribution / component CVaR** — derive each asset's marginal
  contribution to portfolio CVaR from the optimal scenario weights.
* **Stress testing** — inject deterministic scenarios on top of the
  Monte Carlo matrix before optimisation.
* **Filtered historical / GARCH-based scenario generators** — plug into
  `build_optimization_scenarios` behind the same interface.
* **Multi-period (drawdown-aware) optimisation** — replace the single-
  period LP with a path-CVaR formulation.

## Dependencies

* `cvxpy >= 1.4`
* Phase-1 (`risk_metrics`), Phase-4 (`monte_carlo`) modules.

## Phase 5.5 additions

- **Maximum Sharpe portfolio.** `maximize_sharpe_ratio(...)` selects the highest
  Sharpe `(E[r] - rf) / vol` portfolio from constraint-feasible candidates
  generated along the CVaR efficient frontier — globally feasible, open-source
  solvers only.
- **Shrinkage estimator.** `estimate_expected_returns(..., method="shrinkage_to_zero",
  shrinkage_weight=w)` returns `w × mean`.
- **Manual views seam.** `views.apply_manual_expected_return_views(base, views,
  blend_weight)` blends user `AssetReturnView`s into the expected-return vector.
  Reserved for a future Black-Litterman / Entropy-Pooling layer (not implemented).
- **Risk-free / Sharpe.** `calculate_portfolio_scenario_metrics` and
  `compare_current_vs_optimized` accept `risk_free_rate` and report a Sharpe
  ratio / column. Per-horizon rates come from `utils.annual_to_horizon_rate`.
