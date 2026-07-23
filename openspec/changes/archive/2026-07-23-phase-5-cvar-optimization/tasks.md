# Phase 5 — CVaR Portfolio Optimization — task list

## A. Optimization module

- [x] Create `src/var_cvar_crypto_risk/optimization.py`.
- [x] `validate_scenario_matrix` (no NaN / inf, ≥ 2 scenarios, ≥ 1 asset).
- [x] `add_cash_asset` (non-mutating, raises on duplicate).
- [x] `estimate_expected_returns` (`mean` / `median` / `zero`).
- [x] `calculate_portfolio_scenario_metrics` (return / vol / VaR / CVaR /
      worst / best; optional money values).
- [x] `minimize_cvar` (Rockafellar-Uryasev LP).
- [x] `maximize_return_with_cvar_constraint`.
- [x] `minimize_cvar_for_target_return`.
- [x] `generate_cvar_efficient_frontier` (range sweep, infeasible-safe).
- [x] `compare_current_vs_optimized` (returns DataFrame with money
      VaR / CVaR; tolerates infeasible optimisers).
- [x] `format_weights_table` (descending Asset / Weight).
- [x] `build_optimization_scenarios` (historical / normal_mc /
      student_t_mc; horizon-aware historical aggregation).

## B. Plotting

- [x] `plot_optimized_weights` (bar chart, negative weights flagged).
- [x] `plot_portfolio_comparison` (grouped bars across Expected Return /
      Volatility / VaR / CVaR).
- [x] `plot_cvar_efficient_frontier` (Expected Return vs CVaR; min-CVaR
      and max-return points highlighted).
- [x] `plot_allocation_comparison` (grouped bars across portfolios).

## C. Streamlit tab

- [x] New `🎯 Portfolio Optimization` tab.
- [x] Scenario source widgets (Historical / Normal MC / Student-t MC,
      n_scenarios, horizon, df, seed).
- [x] Objective selector (Min CVaR / Max return under cap / Target
      return / Efficient frontier / Compare all).
- [x] Constraint widgets (long-only, min/max weight, include cash,
      cash return, CVaR cap, target return, frontier points).
- [x] KPI cards.
- [x] Optimised weights tables and charts per objective.
- [x] Current vs optimised comparison table and chart.
- [x] Efficient frontier chart + dataframe.
- [x] Allocation-comparison chart.
- [x] CSV / PNG download buttons.
- [x] Session-state persistence (results survive sidebar reruns).
- [x] Friendly error handling for infeasibility and solver issues.

## D. Demo, config, hygiene

- [x] `run_phase5_optimization_demo.py` (thin orchestrator).
- [x] Add `optimization:` block to `configs/config.yaml`.
- [x] Bump project version to `0.5.0` in config + pyproject.
- [x] Add `cvxpy >= 1.4` to `requirements.txt` and `pyproject.toml`.

## E. Tests

- [x] New `tests/test_optimization.py` (≥ 16 tests).
- [x] Validation tests (valid / invalid / non-DataFrame).
- [x] Cash asset tests (adds column / detects duplicate).
- [x] Expected-returns tests (mean / zero / bad-method).
- [x] Portfolio metrics test (CVaR ≥ VaR, money values).
- [x] `minimize_cvar` basic + max-weight + cash variants.
- [x] `maximize_return_with_cvar_constraint` honours cap.
- [x] `minimize_cvar_for_target_return` honours target.
- [x] Infeasible target-return path (no crash).
- [x] Efficient frontier returns weight columns.
- [x] Current-vs-optimised comparison.
- [x] Scenario-builder tests for all three sources + bad-source guard.
- [x] `format_weights_table` sorted descending.
- [x] Import-isolation guard (optimization must not import Streamlit).
- [x] Full test suite continues to pass.

## F. Documentation

- [x] Rewrite `openspec/specs/cvar-optimization.md` (Implemented status,
      math, requirements, edge cases, acceptance).
- [x] Add `openspec/changes/phase-5-cvar-optimization/proposal.md`.
- [x] Add `openspec/changes/phase-5-cvar-optimization/tasks.md`.
- [x] Add `openspec/changes/phase-5-cvar-optimization/design.md`.
- [x] Update `README.md` with Phase-5 section, demo command, output
      list, limitations, resume bullets, updated test count, roadmap.

## Required output files

CSVs under `outputs/tables/`:

- `optimized_weights_min_cvar.csv`
- `optimized_weights_target_return.csv`
- `optimized_weights_cvar_cap.csv`
- `optimization_summary.csv`
- `portfolio_comparison.csv`
- `cvar_efficient_frontier.csv`

PNGs under `outputs/charts/`:

- `optimized_weights_min_cvar.png`
- `current_vs_optimized_risk.png`
- `cvar_efficient_frontier.png`
- `portfolio_allocation_comparison.png`

Phase 1 + Phase 3 + Phase 4 outputs continue to be generated.

## Out of scope

- CVaR risk contribution / component VaR (Phase 6).
- Stress testing / deterministic scenario injection.
- Copulas, GARCH, filtered historical simulation (Phase 6).
- PDF report generation.
- Commercial solvers.
