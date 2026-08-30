# Architecture

## Design objective

The platform separates reusable quantitative calculations from data access,
output generation, and Streamlit presentation. The long-term objective is a
testable Python package that can support multiple interfaces without duplicating
financial logic.

## Data flow

```text
CoinGecko / yfinance / CSV
            ↓
data_loader.py and preprocessing.py
            ↓
returns.py and portfolio.py
            ↓
┌───────────────┬─────────────────┬──────────────────┐
│ VaR / CVaR    │ Backtesting     │ Monte Carlo      │
│ risk metrics  │ validation      │ scenarios        │
└───────────────┴─────────────────┴──────────────────┘
            ↓                         ↓
       diagnostics              optimization
            └──────────────┬──────────┘
                           ↓
                 plotting / exports / UI
```

## Persistent monitoring path

Phase 8 adds a separate persistence path without replacing the Risk Lab:

```text
Streamlit monitoring workspace / one-shot CLI
                    ↓
Experiment creation, historical replay, or live-update service
                    ↓
Existing assumptions / scenarios / optimization / VaR-CVaR adapters
                    ↓
Monitoring domain and fixed-holdings valuation
                    ↓
Repository protocols and SQLAlchemy unit of work
                    ↓
Alembic-managed SQLite database
                    ↓
Read-only dashboard frames → Plotly presentation / private exports
```

`src/var_cvar_crypto_risk/monitoring/` owns the experiment domain, immutable
snapshot, repository adapters, historical/live orchestration, valuation,
forecast evaluation, read models, charts, exports, and one-shot CLI.
`src/var_cvar_crypto_risk/streamlit_ui/monitoring.py` composes those services.
It does not own financial formulas, run a scheduler, or rebalance portfolios.

The local database is a private operational artifact. SQLAlchemy repository and
unit-of-work boundaries isolate persistence from domain objects; Alembic owns
schema history. SQLite is the supported local MVP. PostgreSQL deployment,
authentication, multi-tenancy, and high-availability operations remain outside
the current architecture.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Load and validate YAML configuration. |
| `coingecko_client.py`, `yfinance_client.py` | Vendor-specific price retrieval. |
| `data_loader.py`, `preprocessing.py` | Normalize and validate price data. |
| `returns.py`, `portfolio.py` | Return conventions, horizon aggregation, weights, and portfolio series. |
| `return_conventions.py` | Resolve the Automatic/Advanced return policy and calculation boundaries. |
| `var_models.py`, `cvar_models.py`, `risk_metrics.py` | Core risk measures. |
| `backtesting.py` | Rolling forecasts, breaches, coverage tests, and reports. |
| `monte_carlo.py` | Parameter estimates, scenarios, paths, and scenario risk. |
| `assumptions.py`, `views.py` | Robust optimizer inputs and manual return views. |
| `covariance.py` | Covariance validation, numerical diagnostics, and deterministic repair. |
| `optimization.py` | Scenario construction, CVaR programs, frontier analysis, and diagnostics. |
| `correlation.py` | Dependence diagnostics. |
| `plotting.py`, `export.py` | Presentation-neutral figures and generated files. |
| `monitoring/domain.py`, `monitoring/recipes.py` | Experiment lifecycle, fixed recipes, and monitoring contracts. |
| `monitoring/workflows.py`, `historical_replay.py`, `live_update.py` | Point-in-time creation, sequential replay, and bounded live append. |
| `monitoring/valuation.py`, `risk_forecasts.py` | Fixed-quantity daily states and origin-safe risk evaluation. |
| `monitoring/repository.py`, `models.py`, `database.py` | Persistence protocols, SQLAlchemy adapter, and transaction setup. |
| `monitoring/dashboard.py`, `charts.py`, `exports.py` | Read-only dashboard frames, Plotly charts, and private experiment bundles. |
| `app.py`, `streamlit_ui/` | Streamlit navigation, state, orchestration, and rendering. |

## Current technical debt

`app.py`, `optimization.py`, `plotting.py`, and `backtesting.py` are oversized.
The Streamlit file also owns substantial orchestration and state invalidation.
This increases the regression surface and makes it harder to distinguish UI
behavior from financial behavior.

Refactoring must occur only after numerical regression fixtures exist. The
target direction is:

```text
streamlit_app/
  app.py
  pages/
  components/

src/var_cvar_crypto_risk/
  data/
  risk/
  validation/
  simulation/
  optimization/
  reporting/
  services/
```

The target structure is directional, not a commitment to move every file.
Public APIs and numerical outputs should remain stable unless a methodology
change is separately specified and reviewed.

## State and caching

Streamlit session state stores analysis, backtesting, simulation, assumption,
and optimization results. Cached calculations improve responsiveness but create
a stale-state risk if an input is omitted from a cache key or invalidation path.
Future extraction should use explicit immutable request/configuration objects
and attach data provenance to each result.

Monitoring results are not stored in session state. Each experiment has a UUID,
immutable activated optimizer snapshot, database-backed daily records, and
audited lifecycle events. The monitoring UI consumes persisted read models and
never treats the current Risk Lab session as an authoritative launch snapshot.

## Architectural rules

1. Domain modules must not import Streamlit.
2. Vendor clients must not contain portfolio or risk calculations.
3. Plotting must consume results rather than recompute them.
4. All horizon conversions must be explicit and tested.
5. Scenario, wealth, and optimization boundaries must reject Log inputs rather
   than silently applying Simple-return arithmetic.
6. Methodology changes and architectural refactors must not share a commit.
7. Generated data, caches, and reports must remain outside version control.
8. Parametric simulation must pass covariance governance, and optimizer success
   must pass independent residual validation before presentation as solved.
9. Monitoring must rebuild from its declared point-in-time recipe and must not
   reuse an unproven session-state optimizer result.
10. Finalized daily states and activated snapshots are immutable; archive retains
    history and hard deletion is not exposed.
11. Streamlit performs at most a bounded update. Repeated updates belong to the
    external one-shot CLI and an operator-controlled scheduler.
