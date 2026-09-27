## ADDED Requirements

### Requirement: Explicit portfolio evolution

The Risk Lab MUST build realized portfolio performance from a selected and
versioned evolution policy, and MUST NOT present constant weights as a generic
Buy & Hold portfolio.

#### Scenario: Buy & Hold
- **WHEN** the user selects Buy & Hold with valid launch weights and prices
- **THEN** quantities remain fixed, NAV equals marked holdings and weights drift

#### Scenario: Legacy daily policy
- **WHEN** the user selects daily rebalancing with zero costs
- **THEN** the path reproduces the prior constant-weight daily return series
  within numerical tolerance and is explicitly labeled as daily rebalanced

### Requirement: Auditable rebalance timing

The engine MUST value carried holdings before trading at a complete UTC close,
MUST apply new quantities only to the next interval, and MUST keep the risk
horizon independent of the rebalance calendar.

#### Scenario: Completed period
- **WHEN** a theoretical weekly, monthly or quarterly UTC boundary has full
  prices
- **THEN** that boundary is recorded as both the scheduled and effective
  rebalance date

#### Scenario: Missing scheduled close
- **WHEN** the scheduled date is absent or one or more scheduled-date prices are
  missing
- **THEN** the price is not forward-filled, the event is never moved backward,
  execution is deferred to the first later complete observation, and both dates
  plus a warning are retained

#### Scenario: Multiple pending boundaries
- **WHEN** multiple theoretical boundaries become due before the next complete
  observation
- **THEN** every boundary is retained in the event audit but they coalesce into
  one target-reset trade and one transaction-cost charge

### Requirement: Transaction-cost accounting

The engine MUST apply non-negative commission and slippage to gross traded
notional and MUST solve a post-cost allocation whose weights equal target.

#### Scenario: Positive proportional costs
- **WHEN** a rebalance trades with a positive combined cost rate
- **THEN** net NAV equals pre-cost NAV less the recorded cost, cumulative cost
  equals event costs, and gross NAV remains separately observable

### Requirement: Current risk provenance

Risk Lab outputs described as current risk MUST use the latest policy NAV and
current cutoff weights and MUST export the policy, costs, timing and methodology
version.

#### Scenario: Current monetary VaR and CVaR
- **WHEN** return-space risk is converted to money after a policy path is built
- **THEN** the multiplier is latest net NAV and its value, date and type are
  displayed and exported

### Requirement: Monitoring boundary remains fixed holdings

The system MUST preserve Portfolio Monitor's existing manual fixed-quantity
behavior and MUST NOT introduce optimization or rebalancing into monitoring.

#### Scenario: Existing monitoring experiment
- **WHEN** a manual or legacy experiment is read or updated
- **THEN** its stored quantities, snapshots and persistence contract remain
  unchanged by the Risk Lab portfolio-path feature
