# Monte Carlo Scenario Engine Specification

## Purpose

Define deterministic multivariate Normal and Student-t scenario generation for
portfolio VaR, CVaR, path visualization, and scenario-based optimization. The
engine explores outcomes conditional on estimated inputs; it does not predict
the true future distribution.

## Requirements

### Requirement: Historical parameter estimation

The system SHALL estimate an asset mean vector and covariance matrix from an
aligned return table and SHALL preserve asset labels and ordering.

#### Scenario: Valid asset returns are supplied

- **WHEN** the return table contains at least two valid observations per asset
- **THEN** the estimated mean and covariance use matching asset indices

#### Scenario: Inputs are empty or non-finite

- **WHEN** parameter inputs cannot define a valid finite covariance matrix
- **THEN** the system rejects them before simulation

### Requirement: Multivariate Normal scenarios

The system SHALL generate multivariate Normal asset-return scenarios from a
supplied mean vector and covariance matrix.

#### Scenario: A multi-day horizon is requested

- **WHEN** `horizon_days` is greater than one
- **THEN** the system scales mean and covariance linearly under the documented
  i.i.d. approximation before sampling

### Requirement: Covariance-matched Student-t scenarios

The system SHALL generate multivariate Student-t scenarios with degrees of
freedom greater than two and SHALL rescale the scatter matrix so the simulated
covariance targets the supplied covariance asymptotically.

#### Scenario: Fat-tailed scenarios are requested

- **WHEN** a valid covariance matrix and `df > 2` are supplied
- **THEN** the engine returns the requested number of labeled Student-t
  scenarios

#### Scenario: Invalid degrees of freedom are supplied

- **WHEN** `df <= 2`
- **THEN** the system rejects the request because finite covariance is not
  defined

### Requirement: Reproducible pseudo-random generation

Every simulation entry point SHALL accept an explicit random seed and SHALL
use local random-generator state.

#### Scenario: A run is repeated with the same inputs

- **WHEN** distribution parameters, shape, horizon, and random seed are
  unchanged
- **THEN** the generated scenario matrix is identical

### Requirement: Robust covariance factorization

The system SHALL validate covariance dimensions and SHALL use a documented
jitter fallback when numerical rounding prevents direct Cholesky
factorization.

#### Scenario: Covariance is nearly positive semidefinite

- **WHEN** direct Cholesky factorization fails because of small numerical
  eigenvalue errors
- **THEN** the system retries with bounded diagonal jitter

#### Scenario: Covariance remains invalid

- **WHEN** the matrix cannot be factorized after the bounded fallback
- **THEN** the system raises a clear error instead of silently changing the
  dependence structure

### Requirement: Scenario portfolio aggregation

The system SHALL align portfolio weights to scenario columns and calculate
each portfolio scenario as the weighted sum of asset scenarios.

#### Scenario: Weight labels do not match

- **WHEN** scenario assets and supplied weight labels differ
- **THEN** the system rejects aggregation before reporting portfolio risk

### Requirement: Scenario VaR and CVaR

The system SHALL calculate empirical left-tail VaR and CVaR from portfolio
scenarios and SHALL return both in signed loss space: positive for loss, zero
for break-even, and negative for gain at the modeled tail.

#### Scenario: Tail risk is summarized

- **WHEN** valid portfolio scenarios and a confidence level are supplied
- **THEN** VaR is the sign-reversed empirical return threshold and CVaR is the
  sign-reversed mean return beyond that threshold

### Requirement: Portfolio path simulation

The system SHALL simulate portfolio-value paths beginning at the supplied
initial value and SHALL include the initial value as the first row.

#### Scenario: Paths are generated

- **WHEN** a positive initial value, horizon, path count, and distribution are
  supplied
- **THEN** the output shape is `(horizon + 1, n_paths)` and row zero equals the
  initial value

### Requirement: Simulation assumptions are disclosed

User-facing simulation reports MUST identify distribution, horizon, scenario
count, seed, estimated inputs, and relevant i.i.d. or stationarity assumptions.
They MUST NOT be described as forecasts with known accuracy.

#### Scenario: Normal and Student-t results are compared

- **WHEN** both distributions are simulated from the same historical input
- **THEN** the comparison identifies their assumptions rather than claiming
  that the more conservative estimate is necessarily correct

## Model-Risk Notes

- Estimated means and covariances are subject to large sampling error.
- Linear horizon scaling ignores volatility clustering and regime change.
- Normal and Student-t marginals do not establish correct joint tail
  dependence.
- Simulation volume reduces Monte Carlo error but not model error.
