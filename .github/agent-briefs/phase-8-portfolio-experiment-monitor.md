# Phase 8 Agent Brief — Portfolio Experiment Registry and Forward-Testing Monitor

## Authority and current execution gate

This file is the authoritative implementation brief for Phase 8 of the
**Quantitative Crypto Portfolio Risk Framework**.

Repository:

```text
https://github.com/MRezaaaaaaa/quantitative-crypto-portfolio-risk-framework
```

Public baseline:

```text
Release tag: v1.0.0
Commit: 3e9d0671924ad88879f132a25ff8795c149d02c3
Package: quantitative-crypto-portfolio-risk-framework
Import namespace: var_cvar_crypto_risk
Python: 3.10–3.13
Expected baseline: 412 tests passing
Coverage gate: at least 80%
```

### Current authorization

For the first run, execute **Batch 0 and Batch 1 only**:

1. read-only architecture and methodology audit;
2. Phase 8 OpenSpec proposal, design, tasks, and capability specification.

Do not implement Python, Streamlit, database, migration, dependency, or test
changes during this first run.

After completing Batch 0 and Batch 1:

1. commit the design-only changes to `phase-8-spec`;
2. push only `phase-8-spec`;
3. open a Draft Pull Request against `main`;
4. use the title:
   `design: specify Phase 8 portfolio experiment monitor`;
5. stop for review.

Do not modify this agent brief.

The remaining sections define the complete Phase 8 target so the design can be
evaluated against the full intended scope. They do not authorize implementation
beyond Batch 1 yet.

---

## Role

Act as a senior quantitative developer, data architect, and model-risk
reviewer.

Phase 8 must transform validated optimizer outputs into registered portfolio
experiments that can be evaluated through:

1. Historical Out-of-Sample Replay;
2. Live Forward Testing;
3. Hybrid Historical OOS + Live Forward.

Every experiment must have a persistent ID, human-readable name, immutable
optimization snapshot, daily portfolio state, daily allocation state, daily
risk records, data-quality history, and an interactive monitoring dashboard.

---

## Non-negotiable Git and safety rules

1. Work only on the branch authorized for the current batch.
2. Never push directly to `main`.
3. Do not merge Pull Requests.
4. Do not rewrite Git history.
5. Do not create or modify tags or releases.
6. Do not deploy the application or publish packages.
7. Do not weaken branch protection, CI, security, dependency, coverage, or
   public-boundary controls.
8. Do not add real portfolio data, secrets, credentials, API keys, or private
   databases.
9. Do not modify files under any external or parent `sources/` directory.
10. Preserve unexpected user changes and stop rather than overwrite them.
11. Do not perform broad unrelated refactoring.
12. Do not modify existing financial formulas as part of Phase 8.
13. Do not regenerate or change existing numerical baselines unless an
    independently reviewed methodology change explicitly requires it.
14. Do not use live API calls in automated tests.
15. Do not bump the package version during implementation. Record work under
    `CHANGELOG.md` Unreleased. A future release gate will prepare `v1.1.0`.

---

## Current architectural context

The repository already implements:

- VaR and CVaR methods;
- explicit return and loss conventions;
- rolling VaR backtesting;
- Monte Carlo scenarios;
- scenario-based CVaR optimization;
- robust expected-return and covariance assumptions;
- optimizer residual validation;
- Streamlit analytics;
- OpenSpec specifications;
- numerical regression fixtures;
- publication and repository-governance checks.

Known architectural constraint:

- `app.py` already owns substantial orchestration and Streamlit state;
- `app.py`, `optimization.py`, `plotting.py`, and `backtesting.py` are
  oversized;
- Phase 8 must not place the monitoring domain, persistence, or financial
  calculations directly inside `app.py`.

Target separation:

```text
Streamlit presentation
        ↓
Monitoring/application services
        ↓
Quantitative domain + repository interface
        ↓
Database adapter
```

Architectural rules:

1. Monitoring domain modules must not import Streamlit.
2. Database repositories must not render charts.
3. Chart functions must consume chart-ready data and must not silently
   recalculate financial metrics.
4. Existing VaR, CVaR, backtesting, optimization, return, and risk-convention
   functions must be reused where applicable.
5. Historical replay and live updating must share the same valuation and risk
   workflow.
6. Methodology changes and architectural refactors must remain separate.

---

## Phase 8 scope

Implement in later authorized batches:

1. Experiment registry
2. Immutable optimization snapshots
3. Historical OOS replay
4. Live forward-test tracking
5. Hybrid historical/live experiments
6. Persistent daily portfolio valuation
7. Persistent daily asset allocation and weight drift
8. Persistent daily risk forecasts and realized outcomes
9. Benchmark tracking
10. Streamlit monitoring interface
11. Interactive Plotly monitoring charts
12. Experiment comparison
13. CSV and JSON exports
14. Database migrations
15. Tests and documentation
16. CLI/manual update command suitable for an external scheduler

Explicitly out of scope:

- rebalancing;
- transaction or order execution;
- broker or exchange integration;
- transaction ingestion;
- intraday monitoring;
- trading signals or trading alerts;
- user authentication;
- multi-tenant permissions;
- production scheduler infrastructure;
- transaction-cost optimization;
- tax modelling;
- general multi-asset calendar support;
- performance guarantees;
- public storage of real holdings or monitoring databases.

Phase 8 remains crypto-focused and uses the repository's disclosed 365-day
annualization convention where applicable.

---

## Experiment identity and lifecycle

Each monitoring dashboard represents one Experiment.

Required identity:

```text
experiment_id: immutable UUID
experiment_name: required human-readable name
description: optional
portfolio_snapshot_id: immutable UUID
```

The Experiment ID is also the dashboard ID. Do not create a redundant
dashboard identifier.

Supported modes:

```text
historical_oos
live_forward
hybrid
```

Supported statuses:

```text
draft
backfilling
active
completed
failed
archived
```

Requirements:

- names do not need to be globally unique because the UUID is authoritative;
- validate all status transitions;
- record UTC `created_at` and `updated_at`;
- use archive semantics rather than hard delete in the UI;
- record important lifecycle events.

---

## Methodology contracts

### Historical OOS is not live forward testing

Use these labels:

```text
Historical Out-of-Sample Replay
Live Forward Test
Hybrid Historical OOS + Live Forward
```

Never describe historical replay as a genuine live forward test.

### Point-in-time boundary

For historical and hybrid experiments:

```text
training_start
    <= training_end
    <= optimization_as_of
    < launch_date
    <= historical_evaluation_end
```

The optimizer may receive only observations available on or before
`optimization_as_of`.

An optimizer result calculated with observations after the selected historical
cutoff must not be reused for a historical experiment. Historical experiments
must rebuild the optimization from the training slice and the saved optimizer
recipe.

### Launch convention

The current project uses daily close-based prices.

Use this explicit research convention:

1. `optimization_as_of` is the final observation available to estimation.
2. `launch_date` is the next valid price observation after
   `optimization_as_of`.
3. Initial quantities are calculated using launch-date prices.
4. No return is credited on launch date.
5. Performance starts from the following valid observation.
6. Fees, slippage, liquidity, taxes, and execution uncertainty are absent and
   must be disclosed.

Do not silently shift launch dates. If launch prices are missing, block the
experiment and report the affected assets and dates.

### Fixed holdings

Phase 8 uses a fixed-quantity, buy-and-hold convention after launch:

```text
quantity_i = initial_asset_value_i / launch_price_i

asset_value_i,t = quantity_i * price_i,t

NAV_t = cash_value_t + sum(asset_value_i,t)

current_weight_i,t = asset_value_i,t / NAV_t
```

Quantities remain constant because rebalancing is outside Phase 8.

### Cash

Cash must be explicit. Store:

- initial cash allocation;
- cash-accrual method;
- annual cash rate;
- daily conversion convention.

Support:

1. zero cash return;
2. configured deterministic annual rate converted consistently to a daily
   return.

Do not infer cash returns from future market data.

### Return convention

Wealth, NAV, quantities, cumulative performance, and portfolio valuation use
Simple-return arithmetic.

Do not use Log returns for wealth evolution.

When an upstream analysis uses Log returns, resolve the existing return policy
at the documented wealth boundary. Do not silently mix return conventions.

### Target and current weights

Persist and expose:

- immutable target weights from the optimization snapshot;
- current drifted weights derived from current market values.

Current-exposure risk must use current drifted weights unless explicitly
labelled as target-portfolio risk.

### Daily cadence versus risk horizon

Monitoring cadence and risk horizon are separate:

- portfolio and monitoring records are stored daily;
- risk horizon remains an explicit experiment parameter;
- a forecast created as of date `t` may use only information through `t`;
- store forecast origin, target date, horizon, estimation window, methods,
  confidence level, and convention version;
- evaluate a forecast only when its realized horizon loss is available;
- store overlapping or non-overlapping multi-day evaluation mode.

Do not treat CVaR/Expected Shortfall as a breach threshold.

VaR breach:

```text
realized_loss > forecast_VaR
```

CVaR is a tail-severity estimate, not an exception boundary.

### Forecast versus realized

Forecast and realized metrics must use identical:

- horizon;
- unit;
- portfolio definition;
- return convention;
- date boundary.

Do not fabricate a forecast fan.

A fan chart is permitted only when a path distribution was genuinely
calculated and frozen at launch. If only horizon scenarios or point forecasts
exist, show a horizon-aligned metric comparison and label forecast paths as
unavailable.

### Future dates

A live experiment ending in the future remains Active and Collecting Data.
Future realized values must remain absent until actual observations arrive.

---

## Immutable optimization snapshot

Persist:

### Identity and provenance

- snapshot ID;
- experiment ID;
- creation timestamp;
- package/code version;
- source-data fingerprint;
- assumption-recipe fingerprint.

### Construction

- asset universe;
- target weights;
- cash weight;
- initial capital;
- base currency;
- objective;
- long-only or short-enabled mode;
- minimum and maximum weights;
- target return, if applicable;
- CVaR cap, if applicable;
- solver and status;
- independent residual-validation result.

### Assumptions

- expected-return estimator;
- trim or winsor parameters;
- covariance estimator;
- EWMA lambda;
- shrinkage delta and target;
- scenario source and count;
- Student-t degrees of freedom, if applicable;
- random seed;
- risk-free rate;
- horizon;
- confidence level;
- return-policy metadata;
- loss-convention version.

### Launch forecast

- expected horizon return;
- expected volatility;
- forecast VaR;
- forecast CVaR;
- expected Sharpe where meaningful;
- scenario metadata;
- frozen path percentiles only when genuinely available.

Snapshots become immutable after activation.

Do not save an optimization as valid unless the existing optimizer reports a
solved state and independent residual validation passes.

---

## Persistence architecture

Use repository and service boundaries.

Suggested direction:

```text
src/var_cvar_crypto_risk/monitoring/
    __init__.py
    domain.py
    repository.py
    database.py
    experiment_service.py
    snapshot_service.py
    valuation.py
    historical_replay.py
    live_update.py
    risk_monitoring.py
    comparison.py
    chart_data.py
    export.py
    cli.py
```

A smaller variation is acceptable if responsibilities remain explicit.

Design for:

- SQLAlchemy 2.x;
- Alembic migrations;
- SQLite local MVP;
- repository boundaries compatible with a future PostgreSQL adapter.

Do not implement production PostgreSQL deployment during Phase 8.

Default local database:

```text
data/monitoring/portfolio_monitor.db
```

Requirements:

- track only a `.gitkeep` where needed;
- ignore database, WAL, SHM, and journal files;
- support a database URL from an environment variable;
- add only safe placeholders to `.env.example`;
- never log credentials or secret-bearing database URLs;
- enable SQLite foreign keys;
- use transactions;
- add appropriate indexes and uniqueness constraints;
- make migrations deterministic and testable.

---

## Logical database schema

The design must include at least these tables.

### `experiments`

- experiment ID;
- name and description;
- mode and status;
- base currency;
- initial capital;
- benchmark symbol;
- training start/end;
- optimization as-of;
- launch date;
- historical evaluation end;
- live tracking end;
- UTC lifecycle timestamps;
- source metadata;
- schema version.

### `optimization_snapshots`

- snapshot ID and experiment ID;
- package version;
- objective;
- assumptions JSON;
- constraints JSON;
- forecast JSON;
- source-data hash;
- assumption-recipe hash;
- solver and status;
- residual-validation JSON;
- creation and activation timestamps.

### `snapshot_allocations`

- snapshot ID;
- asset and asset type;
- target weight;
- launch price;
- initial value;
- quantity;
- cash flag.

### `price_observations`

- symbol;
- observation date;
- price;
- quote currency;
- source;
- retrieval timestamp;
- data status;
- uniqueness sufficient to prevent duplicate observations from one source.

### `daily_portfolio_states`

- experiment ID and date;
- NAV and cash value;
- daily and cumulative return;
- realized volatility;
- running peak;
- drawdown;
- maximum drawdown to date;
- benchmark NAV and return;
- data-quality status;
- calculation version;
- lifecycle timestamps;
- uniqueness on experiment and date.

### `daily_asset_states`

- experiment ID and date;
- asset;
- price and quantity;
- market value;
- target and current weights;
- drift in percentage points;
- uniqueness on experiment, date, and asset.

### `daily_risk_forecasts`

- forecast ID and experiment ID;
- as-of and target dates;
- horizon days and evaluation mode;
- estimation window;
- VaR and CVaR methods;
- confidence level;
- forecast VaR, CVaR, and volatility;
- realized horizon loss;
- VaR breach;
- evaluation status;
- model version;
- creation and evaluation timestamps.

### `monitoring_runs`

- run ID and experiment ID;
- run type;
- requested and actual cutoffs;
- start/end timestamps;
- status;
- inserted, updated, and skipped rows;
- warning count;
- sanitized error summary.

### `experiment_events`

- event ID and experiment ID;
- event date;
- event type;
- metadata;
- creation timestamp.

Allowed Phase 8 events may include:

- created;
- optimization snapshot saved;
- launched;
- historical backfill started/completed;
- live update;
- evaluation completed;
- archived;
- data-quality warning.

Do not add rebalancing events yet.

---

## Idempotency and data quality

Running an update repeatedly with identical data must not:

- duplicate daily rows;
- duplicate forecasts;
- modify immutable snapshots;
- double-compound cash;
- change finalized records without an explicit correction workflow.

Each update must:

1. validate source data;
2. determine the last complete UTC observation;
3. exclude partial current-day data by default;
4. identify missing dates and assets;
5. calculate within a controlled transactional workflow;
6. commit related rows atomically;
7. roll back on failure;
8. record the run result.

Do not silently forward-fill missing prices.

If a required price is unavailable:

- mark the date incomplete;
- do not fabricate NAV;
- expose the data-quality problem;
- continue only under an explicit documented policy.

---

## Historical OOS workflow

Input:

- normalized daily prices;
- training start/end;
- optimization as-of;
- launch date;
- evaluation end;
- exact optimization recipe;
- initial capital;
- benchmark;
- risk-monitoring recipe.

Process:

1. slice training data through the optimization cutoff;
2. rebuild expected returns, covariance, scenarios, and optimizer inputs from
   the training slice;
3. call the existing optimizer;
4. apply existing residual governance;
5. freeze the snapshot;
6. obtain launch-date prices;
7. calculate fixed initial quantities;
8. replay valuation sequentially through the OOS period;
9. create each risk forecast using information available only at that origin;
10. persist daily states, forecasts, outcomes, and quality metadata.

The historical engine must behave like the live process with historical data
revealed sequentially.

Do not vectorize in a way that leaks future observations.

---

## Live forward workflow

Later implementation must support:

- update one experiment;
- update all active experiments;
- Streamlit `Update Now`;
- a CLI suitable for cron or an external scheduler.

The updater must:

- use source mappings captured by the experiment;
- append only new complete observations;
- exclude partial current-day data by default;
- evaluate matured VaR forecasts;
- preserve finalized records;
- record update-run metadata;
- leave future dates pending;
- fail clearly when source metadata cannot reproduce the feed.

Do not run an infinite loop or scheduler inside Streamlit.

Do not store a live SQLite database in GitHub Actions. Document how an
external scheduler can invoke the CLI.

---

## Daily portfolio and risk metrics

Persist:

### Performance

- NAV;
- Base-100 indexed NAV;
- daily return;
- cumulative return;
- realized volatility;
- running peak;
- drawdown;
- maximum drawdown;
- benchmark NAV and return.

### Allocation

- target weights;
- current weights;
- asset values;
- cash;
- asset-level drift;
- total portfolio drift.

Total drift:

```text
total_drift_t = 0.5 * sum(abs(current_weight_i,t - target_weight_i))
```

### Risk

- rolling VaR forecast;
- rolling CVaR/Expected Shortfall forecast;
- forecast volatility;
- realized horizon loss;
- VaR exception;
- rolling exception rate where valid.

Use current drifted weights for current-exposure risk.

Reuse tested existing functions and conventions. Do not create alternative
VaR/CVaR formulas.

---

## Streamlit design

Keep integration into `app.py` minimal.

Use a separate presentation package, for example:

```text
streamlit_ui/
    __init__.py
    portfolio_monitor.py
    monitoring_forms.py
    monitoring_charts.py
```

Required views:

### Experiments

- ID and name;
- mode and status;
- date boundaries;
- latest NAV;
- latest update;
- data-quality status.

### Create Forward Test

- Historical, Live, or Hybrid mode;
- strict date validation;
- optimizer-recipe selection;
- historical optimization rebuilt from the training slice;
- methodology preview before saving.

### Portfolio Monitor

- select by experiment ID and name;
- immutable snapshot summary;
- current performance and risk KPIs;
- historical/live boundaries.

### Asset Allocation

- target/current weights;
- allocation evolution;
- weight drift.

### Risk and Breaches

- VaR/CVaR history;
- realized losses;
- VaR exceptions;
- rolling exception rate;
- explicit warning that CVaR is not an exception threshold.

### Forecast versus Realized

- horizon-aligned comparison;
- fan chart only with a frozen path distribution;
- explicit limitation when no path forecast exists.

### Experiment Comparison

- common-calendar alignment;
- days-since-launch alignment;
- no undisclosed unequal-period comparison.

### Data Quality

- missing and stale observations;
- failed update runs;
- actual source;
- last complete data date.

The UI may archive but must not hard-delete an experiment.

---

## Chart contracts

Use Plotly for new monitoring charts.

The chart layer receives chart-ready data and does not own hidden financial
logic.

### NAV

- line chart;
- actual portfolio and benchmark;
- currency and Base-100 modes;
- historical/live boundary.

### 100% stacked asset allocation

Use:

```text
go.Scatter(stackgroup="one", groupnorm="percent")
```

Requirements:

- stable asset colors;
- Cash included explicitly;
- hover includes date, asset, weight, value, quantity, and price;
- daily values sum to 100% within tolerance;
- launch boundary displayed;
- no implication that rebalancing occurred.

### Target versus current weights

- grouped horizontal bar or dumbbell;
- target, current, and percentage-point difference;
- over/under-target state.

### Weight drift

- asset/date heatmap;
- total-drift line;
- no rebalancing recommendation language.

### Drawdown

- underwater area chart;
- peak, trough, and recovery where available;
- calculation:
  `drawdown_t = NAV_t / running_max_NAV_t - 1`.

### VaR/CVaR history

- realized return or loss;
- VaR forecast;
- CVaR/Expected Shortfall estimate;
- method, horizon, confidence, and window shown;
- correct sign if plotted in return space;
- no CVaR exception markers.

### VaR breach timeline

- realized return/loss;
- VaR threshold;
- red exception markers;
- historical/live boundary;
- optional rolling exception-rate panel.

### Forecast versus realized

When frozen path forecasts exist:

- percentile fan;
- realized path overlay.

Otherwise:

- horizon-aligned grouped or dumbbell comparison;
- label `Forecast path unavailable`.

### Experiment comparison

- Base-100 NAV lines;
- risk/return scatter;
- comparison table;
- explicit common-calendar or days-since-launch alignment.

---

## Exports

Later implementation must export:

- experiment metadata;
- optimization snapshot;
- target allocations;
- daily NAV;
- daily allocation;
- daily risk forecasts;
- VaR exceptions;
- forecast-versus-realized metrics;
- data-quality report;
- experiment comparison.

Use CSV for tables and JSON for metadata.

Every export must retain:

- experiment ID and name;
- mode;
- date boundaries;
- as-of date;
- horizon and confidence;
- methods;
- package version;
- assumption-recipe hash;
- source-data hash.

Never export credentials or secret-bearing database configuration.

---

## Dependency direction

Expected later additions:

- SQLAlchemy;
- Alembic;
- Plotly.

Requirements:

- constrained lower and upper bounds consistent with current repository style;
- Python 3.10–3.13 compatibility;
- updated `uv.lock`;
- no removal or weakening of existing constraints;
- no unnecessary dependency additions;
- existing Matplotlib plots remain supported.

No dependency changes are authorized in the current design-only run.

---

## OpenSpec deliverable for Batch 1

Create a change following the installed repository OpenSpec conventions.

Change name:

```text
phase-8-portfolio-experiment-monitor
```

Expected logical content:

```text
openspec/changes/phase-8-portfolio-experiment-monitor/
├── proposal.md
├── design.md
├── tasks.md
└── specs/
    └── portfolio-experiment-monitor/
        └── spec.md
```

Adapt the exact layout only if the installed OpenSpec version requires a
different valid structure.

The design must cover:

- experiment identity and status;
- immutable snapshots;
- historical OOS;
- live forward tracking;
- hybrid mode;
- point-in-time boundaries;
- persistent daily state;
- idempotency;
- risk-forecast alignment;
- allocation drift;
- chart semantics;
- database design;
- scheduler boundary;
- data quality;
- public/private boundary;
- exports;
- testing;
- acceptance criteria;
- future extension points.

Do not archive the change before implementation and review are complete.

Run strict OpenSpec validation.

---

## Testing requirements for later implementation

Preserve all existing tests and numerical baselines.

Planned domain tests:

- UUID identity;
- required name;
- mode validation;
- status transitions;
- date boundaries;
- immutable snapshots;
- recipe and source-data hashing.

Planned database tests:

- migration from empty database;
- foreign keys;
- uniqueness;
- repository CRUD;
- archive behavior;
- persistence across restart;
- transaction rollback;
- duplicate prevention.

Planned valuation tests:

- initial quantities;
- cash allocation and accrual;
- NAV;
- daily/cumulative returns;
- weights sum to one;
- immutable targets;
- drift and total drift;
- drawdown.

Planned historical OOS tests:

- no post-cutoff optimizer input;
- next valid launch observation;
- no launch-day return;
- no future leakage;
- deterministic replay;
- correct historical/live boundary.

Planned live tests:

- append only new dates;
- idempotent repeated update;
- exclude incomplete day;
- atomic rollback;
- pending future dates;
- evaluate matured forecasts once.

Planned risk tests:

- as-of boundary;
- target-date alignment;
- VaR breach logic;
- CVaR not used as threshold;
- multi-day mode recorded;
- current weights used;
- insufficient-window status.

Planned chart tests:

- allocation sums to 100%;
- stable colors;
- aligned target/current series;
- non-positive drawdown;
- explicit experiment alignment;
- no unsupported fan chart.

Planned security and UI tests:

- database files ignored;
- no secrets in exports/logs;
- synthetic fixtures only;
- public scanner rejects database files;
- local paths not exposed;
- UI handles empty database;
- archive does not delete;
- existing Streamlit smoke still passes.

Automated tests must use temporary SQLite databases and no live APIs.

Final implementation gates will include:

- full pytest and coverage;
- Ruff;
- build;
- clean wheel install;
- Streamlit smoke;
- strict OpenSpec validation;
- public-boundary and history scanners;
- Markdown link validation;
- `git diff --check`.

All existing tests must remain passing and coverage must remain at least 80%.

---

## Documentation plan

Later implementation should add or update:

- `README.md`
- `CHANGELOG.md`
- `docs/architecture.md`
- `docs/model-risk.md`
- `docs/reproducibility.md`
- `docs/portfolio-monitoring.md`
- `docs/forward-testing.md`
- `docs/monitoring-database.md`
- `docs/monitoring-user-guide.md`
- `docs/public-release-checklist.md`
- `.env.example`
- `.gitignore`

Document:

- Historical OOS versus Live Forward;
- Hybrid mode;
- point-in-time cutoffs;
- launch convention;
- fixed-quantity assumption;
- weight drift;
- database location;
- SQLite limitations;
- why Streamlit is not a scheduler;
- manual and CLI updates;
- future external scheduling;
- database configuration;
- archive behavior;
- exports;
- data-quality behavior;
- daily cadence versus risk horizon;
- CVaR not being a breach threshold;
- missing costs, slippage, liquidity, taxes, and execution;
- no future-performance guarantee;
- no rebalancing in Phase 8;
- local/private database boundary.

Do not claim that historical OOS proves future performance.

Do not claim that a passing VaR test validates the full portfolio model.

---

## Public/private boundary

Public repository may contain:

- schema and migrations;
- repository interfaces;
- monitoring services;
- Streamlit UI;
- tests;
- synthetic fixtures;
- deterministic seed scripts;
- documentation;
- synthetic screenshots.

Private or ignored:

- actual SQLite/PostgreSQL databases;
- real holdings;
- transaction history;
- client or investor data;
- API keys;
- database credentials;
- live monitoring records;
- proprietary strategy parameters;
- deployment credentials.

Prefer a deterministic synthetic database-seeding workflow over a committed
binary sample database.

---

## Execution batches

### Batch 0 — Preflight and read-only audit

- verify branch, baseline, status, tests, architecture, and OpenSpec version;
- inspect optimizer result boundaries, return conventions, risk conventions,
  Streamlit state, data loading, CI, and public-boundary controls;
- identify conflicts, missing information, and model risks;
- do not edit until the audit is complete.

### Batch 1 — OpenSpec and architecture

- create proposal, design, tasks, and capability requirements;
- define domain contracts and database schema;
- define historical/live workflows and chart semantics;
- run design validation;
- push design only and open a Draft PR;
- stop for review.

### Batch 2 — Persistence foundation

- dependencies;
- migrations;
- domain entities;
- repository;
- database tests.

### Batch 3 — Valuation and experiment registry

- snapshots;
- quantities;
- daily state;
- drift;
- exports;
- tests.

### Batch 4 — Historical OOS

- cutoff-safe optimization;
- sequential replay;
- daily risk forecasts;
- tests.

### Batch 5 — Live update

- append workflow;
- idempotency;
- CLI;
- monitoring-run audit;
- tests.

### Batch 6 — Streamlit and charts

- monitoring UI;
- experiment views;
- Plotly charts;
- exports;
- UI tests.

### Batch 7 — Documentation and final verification

- documentation;
- changelog;
- security boundary;
- full validation.

Do not begin a batch without explicit authorization when a prior gate requires
review.

---

## Final Phase 8 acceptance criteria

The eventual Phase 8 implementation is acceptable only if:

1. A user can create an experiment with UUID and name.
2. Historical, Live, and Hybrid modes work.
3. Historical optimization uses only pre-cutoff data.
4. The optimizer recipe and validated solution are frozen.
5. Initial quantities and daily NAV are reproducible.
6. Daily target/current weights are persisted.
7. Allocation history produces a valid 100% stacked chart.
8. Weight drift is persisted and displayed.
9. Risk forecasts preserve as-of and target dates.
10. VaR exceptions are evaluated without treating CVaR as a threshold.
11. Historical and live periods are visually separated.
12. Future live dates remain pending.
13. Repeated updates are idempotent.
14. The database survives restart.
15. Real databases and credentials remain outside Git.
16. Existing numerical baselines remain unchanged.
17. Existing and new tests pass.
18. Coverage remains at least 80%.
19. CI, security, and public-boundary checks pass.
20. No unauthorized merge, tag, deployment, or release occurs.

---

## Required report for the current design-only run

After Batch 0 and Batch 1, return:

1. Draft PR URL
2. Exact changed-file list
3. Read-only audit findings
4. Architecture summary
5. Proposed database schema
6. Historical/live/hybrid workflow
7. Point-in-time and look-ahead controls
8. Chart design and calculation boundaries
9. Public/private boundary
10. Proposed dependency changes
11. Test plan
12. Model-risk decisions
13. Unresolved questions
14. OpenSpec validation result
15. Markdown-link validation result
16. `git diff --check` result
17. Confirmation that only design/OpenSpec documentation changed
18. Confirmation that no implementation, merge, tag, deployment, or release
    occurred

Stop after the report and Draft Pull Request. Do not begin Batch 2.
