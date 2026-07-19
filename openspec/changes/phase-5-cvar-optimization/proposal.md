# Phase 5 — CVaR Portfolio Optimization

## Why

Phases 1-4 measure and simulate risk. They answer:

* What is my portfolio's VaR / CVaR today?
* Was last year's model adequate?
* What could the next week look like?

What they **don't** answer is the question that drives every portfolio
manager's decision: *"Given what I now believe about the future, which
weights should I hold?"*

Phase 5 closes that loop. It treats the Phase-4 scenario engine and the
historical return matrix as **inputs to an optimiser** and produces
weights that minimise tail risk, maximise return under a tail-risk cap,
or trace out the CVaR efficient frontier.

## Why CVaR rather than variance

* **Asymmetry.** Variance penalises upside as much as downside. CVaR
  penalises losses only.
* **Tail awareness.** CVaR is the *expected loss in the worst β-tail*,
  so it directly measures the kind of event that wipes out portfolios.
* **LP tractability.** Thanks to Rockafellar & Uryasev (2000), CVaR
  optimisation reduces to a linear programme — extremely fast, globally
  optimal, no curvature pathologies.
* **Coherence.** Unlike VaR, CVaR is a *coherent* risk measure (it is
  sub-additive), so it plays nicely with diversification arguments.

## Why scenario-based rather than parametric

Mean-variance assumes Gaussian returns and a stable covariance matrix.
Crypto markets violate both. The scenario approach inherits whatever
distribution the user feeds in:

* **Historical scenarios** — uses the empirical joint distribution
  directly.
* **Normal Monte Carlo** — useful as a smooth baseline.
* **Student-t Monte Carlo** — adds fat tails that match crypto's
  observed kurtosis.

The same LP machinery works in every case; only `R` changes.

## What we'll do

1. New module `src/var_cvar_crypto_risk/optimization.py`:
   * Scenario validation, optional cash asset, expected-return
     estimator.
   * Three Rockafellar-Uryasev LPs (min-CVaR, max-return-under-cap,
     target-return min-CVaR).
   * CVaR efficient frontier sweep.
   * Current-vs-optimised comparison.
   * `build_optimization_scenarios` adapter for the three sources.
2. Four new plotting helpers in `plotting.py`:
   `plot_optimized_weights`, `plot_portfolio_comparison`,
   `plot_cvar_efficient_frontier`, `plot_allocation_comparison`.
3. New Streamlit tab **🎯 Portfolio Optimization** with scenario,
   objective, and constraint widgets, KPI cards, tables and charts.
4. New `run_phase5_optimization_demo.py` script that exercises the
   full pipeline and writes 6 CSVs + 4 PNGs.
5. New tests in `tests/test_optimization.py` (≥ 16 cases) plus an
   import-isolation guard.
6. Configuration: new `optimization:` section in
   `configs/config.yaml`; version bumped to `0.5.0`.
7. Dependencies: add `cvxpy >= 1.4` to `requirements.txt` and
   `pyproject.toml`.
8. Documentation: rewrite `openspec/specs/cvar-optimization.md` and add
   `openspec/changes/phase-5-cvar-optimization/{proposal,tasks,design}.md`.
9. README: new **Phase 5** section with motivation, demo command,
   output list, limitations, and resume-friendly bullets.

## Out of scope (deferred)

* Risk contribution / component CVaR (Phase 6).
* Stress testing / scenario injection.
* Copula-based joint modelling, GARCH dynamics (Phase 6).
* PDF report generation.
* Commercial solvers (Gurobi, Mosek, etc.).

## Acceptance

Phase 5 is complete when every item in `tasks.md` is ticked,
`pytest` reports 0 failures, and `python run_phase5_optimization_demo.py`
produces every CSV / PNG listed in `tasks.md` without error.
