# Portfolio Experiment Monitor — capability delta

## ADDED Requirements

### Requirement: Persistent experiment identity and lifecycle

The system SHALL assign every experiment an immutable UUID, require a
human-readable name, persist its mode and status, validate every lifecycle
transition, record UTC lifecycle events, and archive rather than hard-delete an
experiment.

#### Scenario: Two experiments use the same name

- **WHEN** two valid experiments are created with the same human-readable name
- **THEN** each receives a different authoritative UUID and both remain
  independently addressable

#### Scenario: An invalid transition is requested

- **WHEN** a transition is not permitted by the documented state machine
- **THEN** the system rejects it without changing the status or deleting the
  event history

#### Scenario: A user archives an experiment

- **WHEN** an admissible experiment is archived from the UI or service
- **THEN** its status becomes `archived` and its snapshot and monitoring history
  remain queryable

### Requirement: Experiment modes are labelled accurately

The system SHALL support `historical_oos`, `live_forward`, and `hybrid` modes
and MUST label their outputs respectively as Historical Out-of-Sample Replay,
Live Forward Test, and Hybrid Historical OOS + Live Forward.

#### Scenario: Historical data is replayed after a cutoff

- **WHEN** an experiment evaluates already-available historical observations
  after rebuilding from a past cutoff
- **THEN** every view and export labels it Historical Out-of-Sample Replay and
  does not describe it as a live forward test

#### Scenario: A hybrid replay reaches its live boundary

- **WHEN** the historical portion completes and no live end date has matured
- **THEN** the experiment transitions to `active` and subsequent observations
  are identified as live rather than historical

### Requirement: Point-in-time dates prevent look-ahead

For Historical and Hybrid experiments, the system MUST enforce
`training_start <= training_end <= optimization_as_of < launch_date <=
historical_evaluation_end`. All expected-return, covariance, scenario, and
optimizer inputs MUST use observations at or before `optimization_as_of`.

#### Scenario: A post-cutoff observation is available in the source frame

- **WHEN** optimization is rebuilt for a historical cutoff from a frame that
  also contains later observations
- **THEN** the later observations are excluded from every optimizer input and
  cannot affect the frozen snapshot

#### Scenario: Date boundaries overlap incorrectly

- **WHEN** launch is not after optimization as-of or another required ordering
  is violated
- **THEN** experiment creation fails before optimization or persistence begins

### Requirement: Historical optimization is rebuilt from its recipe

The system SHALL rebuild Historical and Hybrid optimization from the frozen
training slice and serialized optimizer recipe, call the existing optimizer,
and reject reuse of a current-session optimizer result whose information set is
not proven to match the cutoff.

#### Scenario: A current optimizer result includes later market data

- **WHEN** the user creates a historical experiment with a past cutoff
- **THEN** the system ignores the current result, rebuilds assumptions and
  scenarios through the cutoff, and records the rebuilt input dates

#### Scenario: Independent residual validation fails

- **WHEN** the solver reports success but existing residual governance fails
- **THEN** no valid optimization snapshot is activated

### Requirement: Optimization snapshots are immutable after activation

The system SHALL persist one optimization snapshot per experiment containing
construction, assumptions, constraints, provenance hashes, solver state,
residual validation, launch forecast, target allocations, and convention
metadata. The snapshot and target allocations MUST become immutable when
activated.

#### Scenario: A valid solved portfolio is activated

- **WHEN** solver state is accepted and residual validation passes
- **THEN** the system freezes a snapshot UUID, target weights, recipe hash,
  source hash, package/code version, and activation timestamp

#### Scenario: An activated target weight is edited

- **WHEN** any service or repository attempts to modify an activated snapshot
  or target allocation
- **THEN** the transaction is rejected and the stored snapshot is unchanged

### Requirement: Launch uses the next complete price observation

The system SHALL define `optimization_as_of` as the final estimation
observation, use the next complete valid observation as launch, calculate
initial quantities at launch prices, set launch-day return to zero, and begin
performance on the following complete observation.

#### Scenario: One asset lacks a requested launch price

- **WHEN** the requested launch observation is incomplete for the frozen
  universe
- **THEN** activation is blocked and the missing date and assets are reported
  without silently shifting launch

#### Scenario: Launch succeeds

- **WHEN** all asset and benchmark requirements are complete at launch
- **THEN** launch NAV equals initial capital, quantities are reproducible, and
  no return is credited on launch date

### Requirement: Monitoring uses fixed quantities and explicit cash

The system SHALL value fixed post-launch quantities with Simple-return wealth
arithmetic and SHALL represent cash explicitly. Cash MUST use either zero return
or a configured annual rate converted consistently to a 365-day daily rate.

#### Scenario: Market prices change after launch

- **WHEN** a complete later observation is processed
- **THEN** asset quantities remain unchanged while values and current weights
  drift with prices

#### Scenario: An identical update is repeated

- **WHEN** deterministic cash and market inputs are processed again
- **THEN** cash and NAV remain identical and cash is not double-compounded

### Requirement: Persistent monitoring storage is transactional and portable

The system SHALL use repository and unit-of-work boundaries over SQLAlchemy 2.x,
Alembic migrations, and a local SQLite default with foreign keys enabled. The
repository interfaces SHALL remain compatible with a future PostgreSQL adapter.

#### Scenario: A new local database is initialized

- **WHEN** migrations run against an empty temporary SQLite database
- **THEN** all Phase 8 tables, constraints, indexes, and foreign keys are created
  deterministically

#### Scenario: A multi-row daily write fails

- **WHEN** any portfolio, asset, forecast, event, or run write in the unit of
  work fails
- **THEN** the related transaction rolls back without a partial daily state

### Requirement: Daily records and forecasts are idempotent

The system MUST prevent duplicate snapshots, allocations, price observations,
daily portfolio states, daily asset states, and risk forecasts through natural
uniqueness plus idempotent application services.

#### Scenario: The same complete source data is updated twice

- **WHEN** an active experiment receives identical observations and recipe
- **THEN** the second run skips existing finalized rows, records run counts, and
  leaves all financial values unchanged

#### Scenario: An incomplete non-finalized date becomes complete

- **WHEN** a previously missing explicit observation later arrives
- **THEN** the incomplete date may be completed once with an audited quality
  transition rather than duplicated

### Requirement: Monitoring does not fabricate missing prices

The system SHALL validate positive finite prices, identify the last complete
UTC observation, exclude partial current-day data by default, and MUST NOT
silently forward-fill a missing monitoring price.

#### Scenario: One required asset is missing on a date

- **WHEN** the daily universe is incomplete
- **THEN** the date is marked incomplete, no fabricated NAV is finalized, and
  the missing asset appears in data-quality output

#### Scenario: A provider falls back to another source

- **WHEN** a configured provider uses a fallback feed
- **THEN** the actual source and retrieval cutoff are persisted rather than
  reported as the requested source

### Requirement: Historical replay reveals data sequentially

The Historical OOS engine SHALL use the shared live valuation and risk workflow
while revealing evaluation observations in date order. It MUST NOT pass a full
future evaluation frame into an estimator.

#### Scenario: A future evaluation price is perturbed

- **WHEN** a price after a forecast origin is changed in a test fixture
- **THEN** the optimization snapshot and every earlier forecast remain unchanged

#### Scenario: Historical replay completes

- **WHEN** the final requested complete historical observation is processed
- **THEN** the experiment stores reproducible daily states and transitions to
  `completed`, or to `active` when mode is Hybrid

### Requirement: Live updates are append-only and externally schedulable

The system SHALL support one-experiment and all-active one-shot updates from a
service, Streamlit `Update Now`, and a CLI suitable for an external scheduler.
It MUST NOT run an infinite loop or scheduler inside Streamlit.

#### Scenario: A live experiment ends in the future

- **WHEN** observations exist only through today's last complete UTC date
- **THEN** only available complete dates are appended, future realized values
  remain absent, and the experiment remains active

#### Scenario: A non-refreshable source is selected for Live mode

- **WHEN** source metadata cannot reproduce a future feed
- **THEN** Live or Hybrid creation fails with a clear source-capability error

### Requirement: Daily portfolio performance and allocation are auditable

For each complete date the system SHALL persist NAV, Base-100 NAV, daily and
cumulative return, realized volatility, running peak, drawdown, maximum
drawdown, benchmark values, cash, per-asset value, immutable target weight,
current weight, asset drift, and total drift.

#### Scenario: Allocation is complete

- **WHEN** a daily portfolio state is finalized
- **THEN** current asset and cash weights sum to one within tolerance and total
  drift equals `0.5 * sum(abs(current_weight - target_weight))`

#### Scenario: NAV reaches a new high

- **WHEN** current NAV exceeds every prior complete NAV
- **THEN** running peak equals current NAV and drawdown equals zero

### Requirement: Risk forecasts preserve origin, target, and current exposure

The system SHALL store a daily forecast origin, target date, horizon,
evaluation mode, estimation window, methods, confidence, conventions, and model
version. Forecast inputs MUST end at the origin and current-exposure risk MUST
use current drifted weights unless explicitly labelled target-portfolio risk.

#### Scenario: Estimation history is insufficient

- **WHEN** the required window or horizon samples do not exist at an origin
- **THEN** the forecast records `insufficient_window` and does not fabricate a
  risk value

#### Scenario: Target weights differ from current weights

- **WHEN** prices have caused allocation drift at an origin
- **THEN** the default current-exposure VaR/CVaR forecast uses the persisted
  current weights and records that portfolio definition

### Requirement: VaR exceptions and CVaR semantics remain distinct

The system SHALL evaluate a matured forecast only with a realized horizon loss
matching its horizon, units, portfolio definition, convention, origin, and
target. A VaR breach SHALL mean `realized_loss > forecast_VaR`. CVaR MUST NOT be
used as a breach threshold.

#### Scenario: Realized loss exceeds VaR but not CVaR

- **WHEN** the matured realized loss is greater than forecast VaR
- **THEN** the row is a VaR breach regardless of its relationship to CVaR

#### Scenario: A chart displays CVaR history

- **WHEN** CVaR/Expected Shortfall is plotted with realized loss
- **THEN** no CVaR exception marker or CVaR breach count is created

### Requirement: Monitoring charts consume persisted chart-ready data

The system SHALL render new monitoring charts with Plotly from chart-ready data
and MUST NOT hide financial recalculation inside chart functions.

#### Scenario: Allocation evolution is rendered

- **WHEN** complete daily allocation rows are supplied
- **THEN** a stable-color 100% stacked chart includes cash, sums to 100% within
  tolerance, exposes value/quantity/price in hover, and does not imply
  rebalancing

#### Scenario: No frozen path distribution exists

- **WHEN** only point or horizon forecasts were stored
- **THEN** Forecast versus Realized uses a horizon-aligned comparison labelled
  `Forecast path unavailable` rather than fabricating a fan chart

### Requirement: Experiment comparison uses explicit alignment

The system SHALL compare experiments only under a disclosed common-calendar
intersection or days-since-launch alignment and SHALL retain each experiment's
mode and evaluation boundary.

#### Scenario: Experiments have unequal launch dates

- **WHEN** a user selects multiple experiments
- **THEN** the user selects or sees the alignment policy before Base-100 NAV,
  risk/return scatter, or comparison metrics are shown

### Requirement: Exports retain provenance and exclude secrets

The system SHALL export experiment metadata and snapshots as JSON and tabular
monitoring data as CSV. Every export MUST retain experiment identity,
boundaries, methods, horizon, confidence, package/code version, recipe hash, and
source hash and MUST exclude credentials and secret-bearing database settings.

#### Scenario: A monitoring export is generated

- **WHEN** a user exports an experiment
- **THEN** its manifest links all tables to the immutable experiment and
  snapshot metadata without containing a database credential or absolute local
  user path

### Requirement: Monitoring databases and real portfolio data remain private

The public repository SHALL contain schema, migrations, services, UI, tests,
documentation, and synthetic fixtures only. Actual databases, WAL/SHM/journal
files, real holdings, transactions, client data, credentials, live records, and
proprietary parameters MUST remain ignored or private.

#### Scenario: A database file is added to the working tree

- **WHEN** the public-boundary scanner encounters a monitoring database or its
  sidecar file
- **THEN** publication checks fail without printing secret contents

#### Scenario: Automated monitoring tests run

- **WHEN** CI exercises persistence and live-update behavior
- **THEN** it uses temporary SQLite databases, synthetic fixtures, and fake
  providers without live API calls

### Requirement: Claims disclose research limitations

Public UI, documentation, and exports MUST disclose that replay and forward
monitoring omit execution costs, slippage, liquidity, taxes, custody, and
rebalancing and MUST NOT claim performance guarantees, suitability, regulatory
compliance, or complete model validation.

#### Scenario: Historical OOS outperforms its benchmark

- **WHEN** replay metrics exceed the selected benchmark over the evaluation
  period
- **THEN** interpretation remains conditional on the frozen sample, universe,
  data source, assumptions, and missing implementation costs

#### Scenario: A VaR backtest passes

- **WHEN** breach-frequency or independence evidence does not reject its null
- **THEN** the UI does not state that the full portfolio or forecasting model is
  validated
