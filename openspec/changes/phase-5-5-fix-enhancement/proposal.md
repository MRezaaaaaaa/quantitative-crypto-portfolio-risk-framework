# Phase 5.5 — Fix & Enhancement Update

## Why

Phases 1–5 deliver a complete measure → validate → simulate → optimise loop.
Before layering on the heavier Phase 6 machinery (GARCH, copulas, risk
contribution, stress testing) the platform needs a robustness and
interpretability pass so that the existing numbers are trustworthy and the
dashboard is decision-useful. Phase 5.5 fixes a small number of real defects and
adds focused interpretability features **without** touching the core model
architecture.

## What we'll fix

* **Overlapping-only horizon backtests.** For `horizon_days > 1` the backtester
  always used overlapping daily rolling observations, which inflates the sample
  and invalidates the independence interpretation of the Christoffersen tests.
  Phase 5.5 adds a `non_overlapping` mode (step = `horizon_days`).
* **Daily-only distribution chart.** The Distribution tab showed daily VaR/CVaR
  even when the selected risk horizon was multi-day. It now shows horizon-matched
  *h*-day returns.
* **Asset-editor state reset.** Edits to the portfolio table could be dropped on
  rerun because the widget's return value was written back into the same
  `session_state` key that seeded it. The conflicting write-back is removed.

## What we'll add (interpretability)

* Rolling breach-rate chart, worst-realised-losses table, and by-period
  (per-year) breach summary in the Backtesting tab.
* Asset-level return distributions, a QQ-plot vs Normal, a left-tail zoom, a
  "show all VaR lines" overlay, and a methodology panel in the Distribution tab.
* Asset-level cumulative-return and drawdown charts.
* A new **Correlation & Diversification** tab (matrix, heatmap, rolling average
  pairwise correlation).
* A **Maximum Sharpe** portfolio objective (constraint-feasible candidate
  selection over the CVaR frontier).
* A **shrinkage-to-zero** expected-return estimator.
* A **manual expected-return views** input layer (`views.py`) — the seam for a
  future Black-Litterman / Entropy-Pooling layer.
* A **risk-free rate** layer (annual→per-horizon conversion, Sharpe column,
  cash-return wiring).

## Out of scope (intentionally excluded — Phase 6)

* Stress testing / deterministic scenario injection.
* Risk contribution / component VaR / component CVaR.
* GARCH dynamics, copula / DCC dependence, filtered historical simulation.
* Black-Litterman and Meucci Entropy-Pooling implementations (only the input
  seam is added).
* PDF report generation.
* Commercial solvers.

## Acceptance

Phase 5.5 is complete when every item in `tasks.md` is ticked, `pytest` reports
0 failures (194 tests), `python run_demo.py` and
`python run_phase5_optimization_demo.py` still produce all their outputs, and the
Streamlit app renders the new tab, persists asset edits, shows the horizon-matched
distribution, and runs the Max-Sharpe optimiser without error.
