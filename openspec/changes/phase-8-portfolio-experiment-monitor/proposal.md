# Phase 8 — Portfolio Experiment Registry and Forward-Testing Monitor

## Why

The framework can currently produce validated optimizer results, but those
results exist only in Streamlit session state. There is no persistent identity,
immutable optimizer recipe, point-in-time launch record, daily portfolio state,
or out-of-sample evidence. As a result, an optimized allocation cannot be
audited after the session ends and an in-sample result can too easily be
confused with realized performance.

Phase 8 adds a research-grade experiment registry and monitoring workflow. It
separates Historical Out-of-Sample Replay from genuine Live Forward Testing,
supports a Hybrid mode, and preserves the exact cutoff, optimizer recipe,
validated solution, daily valuation, allocation drift, risk forecasts, data
quality, and update history for every experiment.

## What changes

1. Add persistent Experiment identity, lifecycle, events, and archive semantics.
2. Freeze one validated optimization snapshot and target allocation per
   experiment before activation.
3. Add a cutoff-safe Historical Out-of-Sample Replay that rebuilds optimization
   inputs from the training slice and reveals evaluation data sequentially.
4. Add idempotent Live Forward and Hybrid update workflows with an explicit
   scheduler/CLI boundary.
5. Persist daily portfolio, cash, benchmark, allocation, drift, risk forecast,
   realized outcome, and data-quality records.
6. Add a SQLAlchemy repository boundary, Alembic migrations, and a local SQLite
   adapter that can later be replaced by PostgreSQL without changing the domain.
7. Add a thin Streamlit monitoring presentation package and Plotly charts that
   consume chart-ready data without recalculating financial metrics.
8. Add CSV/JSON exports carrying experiment identity, methodology, provenance,
   and hashes.
9. Add deterministic tests, synthetic fixtures, documentation, and public/private
   repository controls for monitoring artifacts.

## Capabilities

### Added

- `portfolio-experiment-monitor`: persistent experiment registration,
  point-in-time historical replay, live/hybrid monitoring, valuation, allocation
  drift, risk forecast evaluation, comparison, exports, and monitoring UI.

### Modified

- No existing quantitative capability is modified by this proposal. Existing
  VaR, CVaR, backtesting, simulation, return, covariance, and optimizer formulas
  remain authoritative and are reused through explicit adapters.

## Impact

Planned implementation areas:

```text
src/var_cvar_crypto_risk/monitoring/
streamlit_ui/
migrations/
tests/monitoring/
docs/
app.py                         # minimal integration only
pyproject.toml / uv.lock       # reviewed dependency additions
.env.example / .gitignore
```

Expected later dependencies are SQLAlchemy 2.x, Alembic, and Plotly with bounded
versions compatible with Python 3.10–3.13. No dependency or implementation
change is part of this design-only batch.

## Model-risk boundary

- Historical replay is labelled Historical Out-of-Sample Replay, never Live
  Forward Test.
- Optimization uses only observations at or before `optimization_as_of`.
- Launch occurs at the next valid observation and earns no launch-day return.
- Holdings are fixed quantities after launch; rebalancing is not implied.
- Daily risk forecasts use information available at their origin and current
  drifted weights.
- VaR is the exception threshold; CVaR is tail severity and never a breach rule.
- Missing prices are not silently forward-filled in monitoring.
- Passing statistical tests or outperforming in a replay does not validate the
  full portfolio model or guarantee future performance.

## Out of scope

- rebalancing, orders, transaction ingestion, exchange or broker integration;
- transaction costs, slippage, liquidity, taxes, custody, or execution models;
- intraday monitoring, trading signals, alerts, or an internal scheduler loop;
- authentication, multi-tenancy, production PostgreSQL deployment, or cloud
  operations;
- general multi-asset calendars or changes to the current crypto-focused
  365-day convention;
- changes to existing financial formulas or numerical golden outputs.

## Delivery gates

This change is delivered in reviewed batches. The current authorized batch is
design only: proposal, design, task plan, and capability requirements. Python,
Streamlit, database, migration, dependency, test, and runtime changes begin only
after this design is reviewed.
