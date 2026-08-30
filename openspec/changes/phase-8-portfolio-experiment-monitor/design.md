# Phase 8 design — Portfolio Experiment Registry and Forward-Testing Monitor

## 1. Design goals

Phase 8 turns a validated optimizer result into an auditable research
experiment. The design must make it impossible to confuse:

- in-sample optimizer metrics with realized performance;
- Historical Out-of-Sample Replay with Live Forward Testing;
- immutable target weights with drifted current weights;
- daily monitoring cadence with the configured risk horizon;
- CVaR tail severity with a VaR exception threshold.

The core invariant is point-in-time reproducibility: every stored calculation
must identify the information set, portfolio definition, method, horizon,
convention, code version, and data provenance that produced it.

## 2. Current-state audit

The design is based on the repository at `main` commit `594061e`, with the
`v1.0.0` numerical baseline still authoritative. The existing suite contains
412 passing tests and enforces an 80% coverage floor.

### Existing strengths to reuse

- Simple-return wealth and optimization policy is centralized in
  `return_conventions.py`.
- signed loss-space conversion is centralized in `risk_conventions.py`;
- VaR, CVaR, rolling backtests, scenario generation, robust assumptions, and
  optimizer residual validation are tested domain functions;
- solver results expose status, solver, weights, metrics, and independent
  constraint validation;
- GitHub CI checks Python 3.10–3.13, build/install, Streamlit startup, CodeQL,
  dependency changes, public-boundary rules, and Git-history rules;
- a deterministic synthetic numerical baseline prevents silent formula drift.

### Gaps and implementation risks

1. `app.py` is approximately 3,400 lines and owns substantial orchestration and
   ephemeral session state. Monitoring logic must not be added there.
2. Optimizer results and assumption recipes are held in `session_state`; there
   is no immutable, serializable portfolio-construction snapshot.
3. Existing price cleaning may forward-fill one missing observation. Monitoring
   requires a stricter ingestion policy: no silent price fabrication.
4. There is no application service that can rebuild an optimizer result from a
   point-in-time recipe independently of Streamlit. A thin optimization adapter
   is required; copying formulas is forbidden.
5. Existing source clients provide research data but do not expose a unified,
   persistable source contract or complete-data timestamp. Live/hybrid creation
   must reject a source that cannot be refreshed reproducibly.
6. Existing plotting is Matplotlib and large. New monitoring charts use Plotly
   in a separate package and receive chart-ready data only.
7. There is no database, migration, repository, transaction, lifecycle, or
   correction policy.
8. Crypto source timestamps and current-day completeness vary by vendor. The
   live updater must default to the last complete UTC observation and record the
   actual cutoff.
9. A current symbol list is not a point-in-time universe. Each experiment must
   freeze the universe and disclose survivorship/selection risk.
10. The framework has no transaction-cost or execution model. Launch is a
    disclosed close-price research convention, not an executable fill.

## 3. Architecture

```text
Streamlit presentation / monitoring CLI
                  |
                  v
Monitoring application services
                  |
       +----------+----------+
       |                     |
       v                     v
Monitoring domain     Existing quant adapters
       |              VaR/CVaR/backtest/optimizer
       v
Repository interfaces
       |
       v
SQLAlchemy unit of work + database adapter
       |
       v
SQLite local MVP (future PostgreSQL adapter)
```

Proposed package boundaries:

```text
src/var_cvar_crypto_risk/monitoring/
├── __init__.py
├── domain.py                 # immutable value objects and lifecycle rules
├── hashing.py                # canonical source/recipe fingerprints
├── repository.py             # protocols and unit-of-work interfaces
├── database.py               # engine/session construction, sanitized errors
├── models.py                 # SQLAlchemy persistence models only
├── experiment_service.py     # create, activate, archive, query
├── optimization_adapter.py   # calls existing assumptions/scenario/optimizer code
├── valuation.py              # fixed-quantity NAV, cash, weights, drift, drawdown
├── historical_replay.py      # sequential OOS orchestration
├── live_update.py            # append-only update orchestration
├── risk_monitoring.py        # forecast creation/evaluation through existing models
├── comparison.py             # explicit alignment policies
├── chart_data.py             # persisted rows -> chart-ready frames
├── export.py                 # CSV/JSON bundles without secrets
└── cli.py                    # one-shot update commands

streamlit_ui/
├── __init__.py
├── portfolio_monitor.py
├── monitoring_forms.py
└── monitoring_charts.py

alembic.ini
migrations/
├── env.py
└── versions/
```

Rules:

- monitoring domain and services do not import Streamlit;
- SQLAlchemy models do not contain chart or financial calculation logic;
- chart functions do not calculate NAV, returns, risk, or drift;
- existing quantitative functions remain the single source of formulas;
- historical and live workflows call the same valuation and risk services;
- database writes occur through a unit of work and transaction;
- `app.py` only composes monitoring views and supplies configuration.

## 4. Domain model

### Experiment identity

`experiment_id` is a UUID generated once and is also the dashboard identifier.
`experiment_name` is required after trimming whitespace but need not be unique.
The service exposes the associated one-to-one `portfolio_snapshot_id` after a
valid snapshot is saved. UUIDs are stored in portable 36-character form.

Modes:

```text
historical_oos
live_forward
hybrid
```

Statuses:

```text
draft
backfilling
active
completed
failed
archived
```

Allowed transitions:

| From | To |
|---|---|
| draft | backfilling, active, archived |
| backfilling | active, completed, failed, archived |
| active | completed, failed, archived |
| failed | backfilling, active, archived |
| completed | archived |
| archived | none in Phase 8 |

Mode-specific normal paths:

- Historical: `draft -> backfilling -> completed`;
- Live: `draft -> active -> completed`;
- Hybrid: `draft -> backfilling -> active -> completed`.

Every transition creates an event and UTC timestamps. Archive is a status
change, never hard deletion. A retry from `failed` is explicit and audited.

### Date contract

For Historical and Hybrid modes:

```text
training_start <= training_end <= optimization_as_of
optimization_as_of < launch_date <= historical_evaluation_end
```

For Hybrid mode, live tracking begins after the historical boundary. For Live
mode, training dates and `optimization_as_of` still describe the estimation
information set, while launch is the first subsequent valid price observation.
If `live_tracking_end` is absent or in the future, the experiment remains
active and future realized rows remain absent.

All dates are normalized to UTC calendar dates. Lifecycle timestamps are
timezone-aware UTC timestamps. Date validators reject ambiguous or reversed
boundaries before any optimization or database mutation.

### Immutable optimization snapshot

An experiment has exactly one optimization snapshot in Phase 8. Draft snapshot
fields may be assembled transactionally, but activation freezes:

- identity, package/code version, source and recipe hashes;
- universe, target weights, cash, capital, currency, objective and constraints;
- estimator, covariance, scenario, seed, horizon, confidence, and convention
  metadata;
- solver state and independent residual validation;
- launch forecast and scenario metadata;
- launch allocations, prices, values, and quantities.

The service accepts a snapshot only when the existing optimizer status is
`optimal` or explicitly reviewed `optimal_inaccurate` and independent residual
validation has `passed=True`. A database update hook and service guard reject
changes after `activated_at` is set. Target allocations are never overwritten
by current weights.

## 5. Persistence design

### Database configuration

Default local location:

```text
data/monitoring/portfolio_monitor.db
```

Environment variable:

```text
QCPRF_MONITORING_DATABASE_URL
```

If unset, the application constructs the local SQLite URL. Logs show only the
dialect and sanitized database label, never credentials or a full secret-bearing
URL. SQLite foreign keys are enabled for every connection. SQLAlchemy 2.x
sessions use explicit transaction scopes. Alembic owns deterministic schema
migrations.

### Tables

#### `experiments`

- `experiment_id` UUID text primary key;
- required `name`, optional `description`;
- checked `mode` and `status` values;
- `base_currency`, positive `initial_capital`, optional benchmark symbol;
- training start/end, optimization as-of, launch, historical end, live end;
- source metadata JSON and schema version;
- UTC created/updated/archived timestamps;
- indexes on status, mode, launch date, and updated time.

Names are indexed but not unique. Hard delete is not exposed by the repository.

#### `optimization_snapshots`

- snapshot UUID primary key and unique experiment foreign key;
- package and code version, objective, solver, solver status;
- assumptions, constraints, launch forecast, scenario metadata, return policy,
  loss convention, and residual validation JSON;
- source-data and assumption-recipe SHA-256 hashes;
- created and activated UTC timestamps.

#### `snapshot_allocations`

- snapshot foreign key plus asset key as composite primary key;
- asset type, target weight, launch price, initial value, quantity, and cash flag;
- check constraints for finite non-negative long-only values where applicable;
- target weights including cash must sum to one within service tolerance.

#### `price_observations`

- surrogate primary key;
- symbol, UTC observation date, positive price, quote currency, source;
- retrieval timestamp and checked data status;
- unique `(symbol, observation_date, quote_currency, source)`;
- indexes on date and symbol/date.

The experiment's source metadata identifies which source row is authoritative.
Conflicting sources are never blended silently.

#### `daily_portfolio_states`

- experiment/date composite uniqueness;
- nullable financial fields for explicitly incomplete dates: NAV, Base-100 NAV,
  cash, daily and cumulative return, realized volatility, running peak,
  drawdown, maximum drawdown, total drift, benchmark NAV and return;
- actual return-interval days plus data-quality metadata so a post-gap return is
  not mislabeled as a one-day observation;
- data-quality status, calculation version, finalized flag, timestamps;
- indexes on experiment/date and data-quality status.

Launch-date return is zero. Complete finalized rows are immutable. An incomplete
row may be completed later when the missing observation becomes available; that
transition is audited.

Realized volatility is an expanding, point-in-time statistic over eligible
post-launch returns with `return_interval_days == 1`. It uses sample standard
deviation (`ddof=1`) and crypto's disclosed `sqrt(365)` annualization. The
launch zero and multi-day post-gap returns are excluded, and the value remains
null until two eligible daily returns exist. This avoids look-ahead and avoids
silently treating a multi-day move as a one-day observation.

#### `daily_asset_states`

- unique experiment/date/asset;
- price, fixed quantity, market value, immutable target weight, current weight,
  and percentage-point drift;
- complete asset weights plus cash must sum to one within tolerance.

#### `daily_risk_forecasts`

- forecast UUID primary key and experiment foreign key;
- origin and target dates, horizon, evaluation mode and estimation window;
- VaR/CVaR methods, confidence, horizon construction and convention versions;
- forecast VaR, CVaR, and volatility in matching units;
- nullable realized horizon loss and VaR breach;
- checked evaluation status, model version, created/evaluated timestamps;
- natural-key uniqueness over experiment, origin, target, horizon, evaluation
  mode, methods, confidence, and model version.

#### `monitoring_runs`

- run UUID, experiment foreign key, run type and status;
- requested and actual cutoffs, UTC start/end;
- inserted, updated, skipped, and warning counts;
- sanitized error code and summary.

#### `experiment_events`

- event UUID, experiment foreign key, effective date, type, metadata JSON, and
  creation timestamp;
- index on experiment and creation timestamp.

### Migration and repository policy

- Migration tests create an empty temporary database and upgrade to head.
- No binary database is committed. Synthetic seed data is generated by code.
- Repositories expose domain objects, not ORM models, beyond the adapter.
- A unit-of-work interface owns commits and rollbacks.
- Uniqueness conflicts are converted to stable domain errors.
- PostgreSQL compatibility influences keys, JSON serialization, and constraints,
  but PostgreSQL deployment is not part of Phase 8.

## 6. Source and data-quality contract

Monitoring receives normalized, positive, daily close observations plus explicit
source metadata. It does not call the current one-day forward-fill cleaning path.
Instead it:

1. normalizes and validates UTC dates;
2. rejects duplicates and non-positive/non-finite prices;
3. identifies the last complete observation across the frozen universe;
4. excludes a partial current UTC day by default;
5. records missing symbols/dates and source staleness;
6. persists only explicit observations;
7. marks an incomplete portfolio date without fabricating NAV.

Historical CSV/input frames are fingerprinted and can create Historical OOS
experiments. Live and Hybrid experiments require a refreshable provider mapping
captured at creation. A one-time uploaded file is not represented as a live
source. CoinGecko and yfinance adapters must expose the actual source used after
fallback; fallback is recorded, not hidden.

The default correction policy permits filling a previously incomplete,
non-finalized date. Changing a finalized observation requires a future explicit
correction workflow and is not silently performed in Phase 8.

## 7. Point-in-time optimization adapter

The adapter receives a serializable `OptimizationRecipe`, normalized prices,
and the cutoff. It does not accept an optimizer result from Streamlit session
state for a historical experiment.

Process:

1. slice prices to the frozen universe and `<= optimization_as_of`;
2. enforce `training_start` and `training_end` within that slice;
3. calculate Simple returns using the existing return policy;
4. reconstruct the configured robust assumptions through `AssumptionConfig`;
5. build scenarios with the existing scenario functions and recorded seed;
6. call the selected existing optimizer function;
7. require a solved status and passed residual validation;
8. canonicalize the recipe and source slice and calculate SHA-256 hashes;
9. create the snapshot and target allocations;
10. refuse activation if any date later than the cutoff entered an input.

The adapter records the maximum input date for expected returns, covariance,
scenarios, and solver inputs independently. Tests spy on these boundaries. This
is a data-leakage control, not only a UI validation.

## 8. Launch and valuation

The launch date is the next complete valid observation after
`optimization_as_of`. It is never shifted silently. If the requested launch
date lacks any frozen-universe price, experiment activation is blocked and the
missing assets/dates are reported.

At launch:

```text
initial_asset_value_i = initial_capital * target_weight_i
quantity_i = initial_asset_value_i / launch_price_i
asset_value_i,t = quantity_i * price_i,t
NAV_t = cash_value_t + sum(asset_value_i,t)
current_weight_i,t = asset_value_i,t / NAV_t
total_drift_t = 0.5 * sum(abs(current_weight_i,t - target_weight_i))
drawdown_t = NAV_t / running_max_NAV_t - 1
```

Quantities remain fixed. No trade, rebalance, or synthetic quantity adjustment
occurs after launch.

Cash modes:

- zero return;
- deterministic annual rate converted as
  `(1 + annual_rate) ** (1 / 365) - 1`.

Cash at date `t` is derived from initial cash and elapsed calendar days, not by
re-compounding a stored result. This makes repeated updates idempotent. Cash is
explicit in allocation tables and charts.

Launch NAV equals initial capital and launch-day return is zero. The next valid
complete observation begins performance. A date after an incomplete date is
not mislabeled as a one-day return; the actual interval is retained in quality
metadata.

Benchmark NAV is Base-100 or capital-scaled from the same launch convention.
Missing benchmark data is disclosed independently and does not silently replace
portfolio observations.

## 9. Historical Out-of-Sample Replay

Historical replay is sequential application of the live workflow to historical
data:

1. validate boundaries and rebuild optimization only from the training slice;
2. freeze and activate the validated snapshot at launch prices;
3. expose evaluation observations one complete date at a time;
4. value fixed quantities and deterministic cash;
5. create each risk forecast using data available at its origin;
6. evaluate only forecasts whose target observation has become available;
7. persist daily state, quality, forecasts, outcomes, run counts, and events;
8. finish a Historical experiment at its evaluation end;
9. transition a Hybrid experiment to Active after the historical boundary.

Vectorized calculation is permitted only inside a single origin's already
available estimation slice. The replay loop owns information revelation and
prevents a full future frame from entering an estimator.

Batch 4 implements this as a calendar-daily crypto replay. Forecast targets are
`origin + horizon_days`; non-overlapping origins are anchored to launch and
spaced by that calendar horizon. Each risk estimate uses the last configured
number of daily Simple returns through the origin, applies the origin's current
drifted weights, and constructs rolling compounded horizon returns without
square-root scaling. Multi-day estimator samples are therefore overlapping and
carry an explicit dependence warning. A missing target observation is never
forward-filled; its forecast remains pending. Snapshot content is rebuilt from
the cutoff slice and compared with any persisted snapshot before replay, so an
in-memory optimizer result cannot be substituted for the point-in-time build.

## 10. Live and Hybrid updates

One service powers `update_experiment`, `update_all_active`, Streamlit
`Update Now`, and the one-shot CLI.

For each run it:

1. opens a monitoring-run record;
2. reconstructs the frozen provider mapping;
3. requests only needed observations plus any estimation lookback;
4. selects the last complete UTC observation and excludes partial current day;
5. validates data and computes missing dates/assets;
6. inserts new explicit prices and daily states in order;
7. creates forecasts from origin-available data;
8. evaluates matured forecasts once;
9. atomically commits related records and final run counts;
10. rolls back and stores only a sanitized failed-run summary on error.

Repeated input produces no duplicate state, forecast, or event. A future end
date creates no future realized rows. Streamlit does not run a loop or scheduler;
an external scheduler may invoke the documented CLI command.

Batch 5 implements the refresh boundary through a dependency-injected provider
protocol. The complete-data policy is conservative: at an invocation timestamp,
the current UTC calendar day is excluded and the effective cutoff is further
bounded by the provider's declared completeness date and any frozen live end.
The full secret-free recipe and symbol mapping are persisted at experiment
creation and revalidated against their SHA-256 fingerprint before each update.
Provider fallback is permitted only when declared in that recipe; the actual
source, fallback flag, retrieval cutoff, and sanitized capability flags are
stored with the run/event trail.

Each invocation first commits a `running` audit record. All subsequent price,
valuation, forecast, outcome, lifecycle, count, and completion writes share one
financial transaction. Failure rolls that transaction back and records a new,
sanitized failed-run outcome separately. A retry always creates a new run linked
to the prior failure rather than mutating it. Previously finalized daily states
provide the experiment-specific immutable price history; freshly fetched data
is used only for the required risk lookback and new candidate dates. Fixed
quantities and the frozen optimization snapshot are never recomputed.

## 11. Risk-monitoring contract

Monitoring cadence is daily, while risk horizon is stored per experiment.

At origin `t`:

- data ends at `t`;
- current drifted weights at `t` define current exposure;
- the estimation window and horizon construction are explicit;
- existing VaR and CVaR functions calculate signed loss-space estimates;
- forecast volatility is stored for the same horizon and portfolio definition;
- the target date is the actual valid observation boundary required by the
  configured overlapping or non-overlapping evaluation mode.

When the target matures:

```text
realized_horizon_return = NAV_target / NAV_origin - 1
realized_horizon_loss = -realized_horizon_return
VaR_breach = realized_horizon_loss > forecast_VaR
```

CVaR is stored and compared as tail severity but does not generate an exception.
Forecast and realized values must share horizon, return convention, units,
portfolio definition, origin, and target boundary. Insufficient estimation
history produces an explicit `insufficient_window` status rather than a number.

The implementation reuses existing VaR/CVaR and horizon functions. It does not
introduce alternate formulas. Multi-day overlapping forecasts are labelled and
their exception-rate dependence limitation is shown in the UI.

## 12. Idempotency and atomicity

Natural uniqueness plus service behavior provides idempotency:

- one snapshot per experiment;
- one allocation per snapshot/asset;
- one explicit price per symbol/date/source/currency;
- one portfolio state per experiment/date;
- one asset state per experiment/date/asset;
- one forecast per full forecast natural key.

The updater first compares canonical inputs with persisted rows. Identical rows
are skipped; incomplete rows may be completed; finalized rows are unchanged.
Cash is derived from launch state, so reruns cannot double-compound. A single
date's portfolio state, asset states, forecasts, and related event are committed
atomically. A failed date does not leave a partially valued portfolio.

## 13. Streamlit presentation

`app.py` adds only a monitoring entry point and navigation. New views live in
`streamlit_ui` and call services/repositories.

### Experiments

List ID/name, mode/status, boundaries, latest complete NAV, latest update, and
quality status. Archived experiments are filterable but not deleted.

### Create Forward Test

Collect mode, strict dates, refreshable source when required, optimizer recipe,
capital, benchmark, cash method, and risk recipe. Show a methodology preview and
point-in-time cutoff before saving. Historical and Hybrid modes always rebuild
the optimization from the cutoff slice.

### Portfolio Monitor

Select by UUID plus name and show immutable snapshot, target allocation,
provenance hashes, performance/risk KPIs, and historical/live boundary.

### Allocation

Show target/current weights, 100% stacked allocation, and drift. Text must not
recommend rebalancing.

### Risk and Breaches

Show VaR/CVaR history, realized loss, VaR exceptions, and an eligible rolling
exception rate. State explicitly that CVaR is not an exception threshold.

### Forecast versus Realized

Use horizon-aligned point comparisons unless a genuine frozen launch path
distribution exists. Never synthesize a fan from point forecasts.

### Comparison and Data Quality

Comparison requires an explicit common-calendar intersection or days-since-
launch alignment. Data Quality exposes missing/stale observations, actual
source, failed runs, and last complete cutoff.

The comparison read service uses the shared intersection for the selected
policy, rebases every NAV series to 100 at that intersection's first point, and
derives comparison return, volatility, drawdown, and breach counts only from
that same aligned sample. Days-since-launch means actual calendar age rather
than compressed row number, so missing dates are not disguised.

## 14. Chart contracts

All new charts use Plotly and chart-ready service outputs.

| Chart | Plotly form | Calculation boundary |
|---|---|---|
| NAV | line; currency/Base-100 toggle | persisted NAV and benchmark only |
| Allocation | `go.Scatter(stackgroup="one", groupnorm="percent")` | persisted current weights; stable colors; explicit cash |
| Target/current | grouped horizontal bar or dumbbell | persisted target/current and percentage-point drift |
| Drift | asset/date heatmap plus total-drift line | persisted asset drift and service-calculated total drift |
| Drawdown | underwater area | persisted `NAV/running_max-1` |
| VaR/CVaR | realized loss plus VaR/CVaR lines | persisted horizon-aligned forecast/evaluation rows |
| Breaches | VaR threshold plus red exception markers | only persisted `realized_loss > VaR` |
| Forecast/realized | grouped/dumbbell by default | frozen fan only when path percentiles actually exist |
| Comparison | Base-100 lines, risk/return scatter, table | explicit common-calendar or launch-age alignment |

Allocation hover contains date, asset, weight, market value, fixed quantity, and
price. Complete daily weights including cash must sum to one within tolerance.
Historical/live boundaries are vertical annotations. Chart code does not fetch
data or calculate financial metrics.

## 15. Exports

An export service produces CSV tables and JSON metadata for:

- experiment and immutable snapshot;
- target allocations;
- daily NAV/performance;
- daily allocation/drift;
- risk forecasts, evaluations, and VaR exceptions;
- forecast-versus-realized comparison;
- data-quality history and experiment comparison.

Every file includes or is linked by manifest to experiment ID/name, mode,
boundaries, as-of date, horizon, confidence, methods, package/code version,
recipe hash, and source hash. Database URLs, credentials, local absolute paths,
and raw secret-bearing source metadata are excluded.

## 16. Public/private and security boundary

Public:

- schema, migrations, domain/services/repositories, UI and charts;
- deterministic synthetic seeds and temporary-database tests;
- documentation and sanitized examples.

Ignored/private:

- `.db`, `.sqlite`, `.sqlite3`, WAL, SHM, and journal files;
- `data/monitoring/*` except `.gitkeep`;
- real holdings, prices not licensed for redistribution, transactions, client
  data, live records, credentials, and proprietary strategy parameters.

The existing public scanner is extended to reject monitoring databases and
secret-shaped configuration. Logs and exports use sanitized errors. Tests use
temporary SQLite databases and no network.

## 17. Dependency plan

Planned bounded additions, subject to Batch 2 review and lock regeneration:

```text
SQLAlchemy >=2.0,<3
Alembic >=1.13,<2
Plotly >=5.20,<7
```

Plotly belongs to the app extra unless chart-data tests require otherwise.
SQLAlchemy and Alembic are runtime monitoring dependencies because the public
monitoring API and CLI require them. Exact lower bounds will be resolved against
Python 3.10–3.13 and the locked environment. No package version bump occurs;
work remains under `CHANGELOG.md` Unreleased until a later `v1.1.0` gate.

## 18. Testing strategy

### Domain and lifecycle

- UUID/name/mode validation, dates and transitions;
- immutable activated snapshot and canonical hashing;
- solved status plus residual-validation requirement.

### Database and migrations

- upgrade empty temporary SQLite database to head;
- foreign keys, checks, indexes, uniqueness, persistence across restart;
- archive semantics, transaction rollback, duplicate prevention;
- sanitized database errors and URL handling.

### Valuation

- quantities, launch-day zero return, cash accrual, NAV and benchmark;
- target/current weights, sum-to-one, drift and total drift;
- running peak, non-positive drawdown, and maximum drawdown;
- incomplete-date behavior and no hidden forward fill.

### Historical OOS and leakage

- every optimizer input ends at or before cutoff;
- launch is next complete observation and does not earn a return;
- sequential replay equals deterministic expected output;
- altering post-cutoff data cannot change the frozen snapshot;
- historical/live boundary is exact.

### Live/hybrid

- append-only complete dates, future pending dates, and repeated-run idempotency;
- incomplete current day excluded; missing asset blocks valuation;
- matured forecast evaluated once; failure rolls back atomically.

### Risk

- origin/target/horizon alignment and current-weight exposure;
- VaR breach rule and no CVaR breach field/marker;
- overlapping mode metadata and insufficient-window status.

### Charts, UI, export, and security

- allocation sums to 100%, stable colors, boundary annotations;
- target/current alignment, non-positive drawdown, comparison alignment;
- no unsupported fan chart; empty database handled;
- archive does not delete; export metadata complete and secret-free;
- scanners reject databases and local paths; Streamlit startup passes.

Final gates retain 412 existing tests and numerical values, add new tests, keep
coverage at least 80%, and pass Ruff, build/install, strict OpenSpec, public and
history scanners, Markdown links, `git diff --check`, and Streamlit smoke.

## 19. Alternatives rejected

### Store experiments in Streamlit session state or CSV

Rejected because restart persistence, relational integrity, migrations,
transactions, idempotency, and concurrent-safe updates cannot be governed.

### Put monitoring orchestration in `app.py`

Rejected because it enlarges existing technical debt and makes CLI/historical
and live workflows diverge.

### Reuse a current optimizer result for historical replay

Rejected because it can contain post-cutoff observations and create look-ahead
bias. Historical and Hybrid snapshots are rebuilt from the cutoff slice.

### Treat historical replay as live testing

Rejected because historical data was already observed by the present-day
researcher and data vendor revisions/survivorship can remain.

### Forward-fill missing monitoring prices

Rejected because it fabricates an unchanged asset price and can understate
volatility, drift, drawdown, and tail loss.

### Embed a scheduler in Streamlit

Rejected because Streamlit reruns are not a durable scheduler. The app exposes
manual update; external infrastructure calls the one-shot CLI.

## 20. Implementation decisions and remaining questions

Resolved implementation decisions:

- Each provider declares `complete_through`; the service also excludes the
  current UTC day and applies the frozen experiment end before accepting a
  realized cutoff.
- `optimal_inaccurate` is rejected by default and requires an explicit flag in
  the persisted optimization recipe in addition to independent residual checks.
- A failed update is immutable. A retry creates a new run linked to the most
  recent failed run and never resumes or rewrites the prior run.

Questions remaining for later batches:

1. Which VaR/CVaR forecast recipes are exposed in the Phase 8 MVP? The design
   supports existing methods, but the first UI should limit combinations to
   those with clearly matched horizon evaluation.
2. SQLite supports a single-user local MVP. Concurrent updater and UI writes
   need a documented lock timeout; production concurrency remains deferred.

These questions do not change the point-in-time, fixed-holdings, idempotency,
or public/private requirements.
