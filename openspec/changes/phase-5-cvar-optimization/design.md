# Phase 5 — Design notes

## 1. Architecture

```
                       ┌──────────────────────────┐
prices ─► returns ────►│ build_optimization_      │── Historical scenarios
                       │   scenarios              │── Normal MC scenarios
                       │   (one entry point)      │── Student-t MC scenarios
                       └─────────────┬────────────┘
                                     │
                                     ▼
                       ┌──────────────────────────┐
                       │ optional cash column     │
                       └─────────────┬────────────┘
                                     │
       expected_returns ◄── estimate_expected_returns
                                     │
                                     ▼
            ┌─────────────────────────────────────────┐
            │  CVXPY LPs (Rockafellar-Uryasev)        │
            │  ┌─────────────┐  ┌──────────────────┐  │
            │  │ minimize_   │  │ maximize_return_ │  │
            │  │ cvar        │  │ with_cvar_cap    │  │
            │  └─────────────┘  └──────────────────┘  │
            │  ┌──────────────────────────────────┐   │
            │  │ minimize_cvar_for_target_return  │   │
            │  └──────────────────────────────────┘   │
            │  ┌──────────────────────────────────┐   │
            │  │ generate_cvar_efficient_frontier │   │
            │  └──────────────────────────────────┘   │
            └─────────────────────┬───────────────────┘
                                  │
                  result dicts ◄──┘
                                  │
                                  ▼
                   compare_current_vs_optimized ──► DataFrame
                                  │
                                  ▼
       Streamlit / run_phase5_optimization_demo.py ──► CSV + PNG
```

Pure analytics in `optimization.py`. All plotting lives in `plotting.py`.
All UI lives in `app.py`. `run_phase5_optimization_demo.py` is an
orchestrator only.

## 2. Why the Rockafellar-Uryasev LP

Naïve CVaR optimisation has to differentiate through a quantile, which
is non-smooth. Rockafellar & Uryasev showed that introducing the
auxiliary `t` (the VaR level) and slack variables `u_i = max(loss_i − t,
0)` produces an exact LP:

```
minimize       t + (1 / ((1 − β) · N)) · Σ_i u_i
subject to     u_i ≥ ℓ_i − t,   u_i ≥ 0
               feasibility constraints on w
```

At optimum, `t` equals the VaR, and the objective equals the CVaR.
Crucially:

* It is **linear** — globally optimal, no curvature, no warm-starts.
* `β` enters only as a constant — different confidence levels reuse the
  same scenario matrix without re-sampling.
* It scales linearly with `N` (one slack per scenario) — even 50 000
  scenarios solve in well under a second with SCS / CLARABEL.

## 3. Scenario sources

`build_optimization_scenarios` is the single adapter for all sources.

| Source         | Implementation |
|----------------|----------------|
| historical     | `asset_returns.dropna()` for `h = 1`; rolling `prod(1 + r) − 1` for `h > 1`. |
| normal_mc      | `simulate_normal_returns` (Phase 4). |
| student_t_mc   | `simulate_student_t_returns` (Phase 4) with `(df − 2)/df` covariance rescaling. |

The LP makes no assumption about how the scenarios were generated — it
only consumes `R`.

## 4. Constraints

* Budget: `Σ w = 1` (always).
* Long-only: `w ≥ 0` if `long_only=True`. A negative `min_weight` passed
  together with `long_only=True` is clipped to 0 (Streamlit warns).
* Box: `w ≤ max_weight` (default 1) and `w ≥ min_weight` (default
  unbounded below when shorts are allowed).
* Cash: optional CASH column with constant per-horizon return; treated
  like any other asset.

## 5. Solver strategy

CVXPY ships with several open-source solvers. The wrapper:

1. Honours an explicit `solver=` argument if available.
2. Otherwise tries `ECOS → CLARABEL → SCS` in order.
3. On `SolverError` falls back through the remaining solvers, then to
   CVXPY's default solver, then surfaces a `solver_error` result.

This keeps the project working out-of-the-box even on minimal installs
(modern CVXPY wheels ship CLARABEL but not ECOS).

## 6. Infeasibility handling

CVXPY signals infeasibility via `problem.status`. The wrappers translate
this into:

```python
{
    "status": "infeasible",
    "objective_value": nan,
    "weights": pd.Series([nan, ...], index=assets),
    "VaR":  nan, "CVaR": nan,
    "message": "Problem is infeasible under the given constraints.",
    ...
}
```

The Streamlit tab and `run_phase5_optimization_demo.py` both render the
message and continue — they never crash on infeasibility.

`generate_cvar_efficient_frontier` filters out infeasible target-return
points but records the count in `frontier.attrs["n_infeasible"]` so
upstream code can warn.

## 7. Confidence-level / horizon convention

* `confidence_level` is `β` in the LP (e.g. `0.95`). Tail probability is
  `1 − β` (5 % of scenarios drive CVaR).
* Historical horizon aggregation uses simple-return compounding
  (`prod(1 + r) − 1`), consistent with Phase-3 backtesting.
* Monte Carlo horizon scaling is **inside** the Phase-4 simulator (mean
  × `h`, covariance × `h`), so the optimiser sees one row per scenario
  already at the chosen horizon.

## 8. Reproducibility

* All Monte Carlo paths through `optimization.py` take `random_seed`
  and forward it to Phase 4.
* The LP is convex and deterministic given the scenario matrix and
  constraints, so the optimal weights are stable across runs.

## 9. Connection to future phases

* **Risk contribution** (Phase 6): given the optimal scenario losses
  `ℓ_i*` and tail mask `{i : ℓ_i* > t*}`, the per-asset CVaR
  contribution is `−E[R_{·j} | tail] · w_j*`. The current outputs
  already preserve the optimal `t` and `w`, so this is a thin
  post-processing layer.
* **Stress testing**: inject deterministic rows into the scenario
  matrix before optimisation; nothing else needs to change.
* **GARCH / filtered historical**: another `build_optimization_scenarios`
  branch, then identical LP plumbing.

## 10. Error semantics

| Source of error | Where it surfaces |
|-----------------|-------------------|
| Bad scenario matrix | `validate_scenario_matrix` raises `ValueError`. |
| Bad expected-returns indexing | LP wrappers raise `ValueError` before solver call. |
| Infeasible LP | Solver returns; wrapper returns result dict with status / message. |
| Solver crash | `_solve` catches and falls back; if everything fails, returns `solver_error` result. |
| Numerical singularity in MC covariance | `_safe_cholesky` in Phase 4 retries with jitter. |
