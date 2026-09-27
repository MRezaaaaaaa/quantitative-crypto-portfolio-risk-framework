# Portfolio Experiment Monitor

## Purpose

Persist and monitor manually entered long-only crypto portfolios. New creation
is manual-only; existing optimized records retain their original provenance.
Risk Lab optimization is independent and unchanged. No trading or rebalancing
is implemented.

## Requirements

### Requirement: Manual-only experiment creation

The system SHALL accept explicitly entered asset weights totaling one without
silent normalization and SHALL NOT run optimization in new monitoring creation.
Weights MUST be positive, finite and unique after symbol normalization. Optional
cash MUST be an explicitly weighted asset and at least one market asset is required.

#### Scenario: A valid portfolio is supplied
- **WHEN** valid manual weights and complete launch prices are supplied
- **THEN** the exact weights are frozen and each market quantity is capital times
  weight divided by launch price, with no optimizer call

#### Scenario: Invalid weights are supplied
- **WHEN** weights are duplicated, non-finite, non-positive or total is not one
- **THEN** creation fails clearly rather than normalizing or choosing weights

### Requirement: Manual snapshot provenance is honest and immutable

The system SHALL mark manual snapshots with objective `manual`, solver `none`
and status `manual_validated`, SHALL validate quantities/values/weights before
activation, and SHALL NOT fabricate expected-return predictions. Later evaluation
prices MUST NOT affect construction. Activated snapshots MUST remain immutable.

#### Scenario: Later evaluation prices change
- **WHEN** prices strictly after the launch are perturbed
- **THEN** the initial allocation, quantities and construction source hash remain unchanged

#### Scenario: Historical weights are entered retrospectively
- **WHEN** the user declares a historical allocation decision date
- **THEN** the UI warns that this is not proof of ex-ante weight selection

### Requirement: Manual launch is explicit

The system SHALL use the user's selected completed UTC launch day with complete
asset and benchmark prices, SHALL NOT shift it silently, and SHALL initialize NAV
at initial capital with zero launch return. The decision date MUST precede launch.

#### Scenario: Launch data are incomplete
- **WHEN** a required launch price is missing or the day is partial or future
- **THEN** snapshot activation is rejected explicitly

### Requirement: Shared historical and live monitoring

The system SHALL support correctly labelled Historical OOS, Live Forward and
Hybrid modes, retain fixed quantities, persist daily NAV/allocation/drift/drawdown,
and compute origin-safe VaR/ES with outcomes evaluated only after maturity.
Missing prices MUST remain visible without silent filling. CVaR MUST NOT be
interpreted as an exception threshold.

#### Scenario: A complete later date is processed
- **WHEN** prices change after launch
- **THEN** quantities remain fixed, weights drift and daily records are persisted

#### Scenario: An identical live update is repeated
- **WHEN** an already-finalized input/cutoff is processed again
- **THEN** no duplicate financial rows or repeated maturity evaluations are created

### Requirement: Preserve provenance and existing records

The system SHALL retain named UUID experiments, recipe/source hashes and private
exports, and SHALL preserve original optimized snapshots and recipe formats.
Static CSV prices SHALL create Historical OOS only; Live/Hybrid require a
refreshable mapped source. No database credentials or private data SHALL be published.

#### Scenario: A legacy optimized record is updated
- **WHEN** a saved optimized experiment receives new complete prices
- **THEN** its initial snapshot and historical provenance remain unchanged
