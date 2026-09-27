## ADDED Requirements

### Requirement: Manual-only experiment creation

The system SHALL create new monitoring experiments only from explicitly entered
manual portfolio weights and SHALL NOT invoke portfolio optimization in that path.

#### Scenario: User supplies initial portfolio
- **WHEN** the user supplies a valid manual portfolio and complete launch prices
- **THEN** the system freezes those weights and their fixed quantities without
  calling an optimizer or changing the weights

#### Scenario: Invalid weights
- **WHEN** weights are non-finite, duplicated, non-positive, or do not sum to one
- **THEN** creation fails explicitly without silently normalizing them

### Requirement: Preserve provenance and existing records

The system SHALL identify manual construction honestly and SHALL preserve
previously recorded optimized experiments without overwriting their history.

#### Scenario: Manual historical replay
- **WHEN** a manual portfolio is replayed through a historical evaluation interval
- **THEN** later prices cannot change the initial allocation snapshot and the UI
  warns that historical manual weight selection may contain hindsight bias

#### Scenario: Existing optimized experiment
- **WHEN** an existing optimized experiment is read or updated
- **THEN** its original recipe and snapshot remain unchanged
