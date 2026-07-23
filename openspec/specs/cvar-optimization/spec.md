# CVaR Portfolio Optimization Specification

## Purpose

Define scenario-based portfolio optimization using the
Rockafellar-Uryasev CVaR formulation, explicit portfolio constraints, and
auditable assumption inputs. Optimized weights are conditional research
outputs, not trading recommendations or evidence of out-of-sample superiority.

## Requirements

### Requirement: Scenario and weight-domain validation

The system SHALL validate finite scenario returns, unique asset labels,
confidence level, weight bounds, short-selling mode, and feasibility of the
full-investment constraint before solving.

#### Scenario: Constraint bounds cannot fund the portfolio

- **WHEN** the supplied minimum and maximum weights cannot sum to a feasible
  fully invested portfolio
- **THEN** the system returns a clear infeasibility diagnostic without
  reporting weights as optimal

### Requirement: Minimum-CVaR portfolio

The system SHALL minimize empirical scenario CVaR using the
Rockafellar-Uryasev auxiliary-variable formulation.

#### Scenario: A feasible minimum-CVaR problem is solved

- **WHEN** a valid scenario matrix, confidence level, and feasible constraints
  are supplied
- **THEN** the system returns weights summing to one, solver status, empirical
  VaR, and empirical CVaR

### Requirement: Return maximization under a CVaR cap

The system SHALL maximize the supplied expected-return vector subject to an
explicit empirical CVaR limit and portfolio constraints.

#### Scenario: The requested cap is feasible

- **WHEN** at least one admissible portfolio satisfies the CVaR cap
- **THEN** the selected weights maximize the stated expected-return objective
  within the feasible set

#### Scenario: The requested cap is infeasible

- **WHEN** no admissible portfolio satisfies the CVaR cap
- **THEN** the output reports infeasibility and does not fabricate a fallback
  optimum

### Requirement: Minimum CVaR for a target return

The system SHALL minimize empirical CVaR subject to a supplied expected-return
target and portfolio constraints.

#### Scenario: The target is attainable

- **WHEN** the target lies within the feasible expected-return range
- **THEN** returned weights satisfy the target within the documented numerical
  tolerance

### Requirement: CVaR efficient-frontier construction

The system SHALL generate frontier points by solving target-return problems
over a disclosed target grid. When at least one point is feasible, the returned
frontier SHALL contain feasible points and SHALL retain the excluded-point
count as metadata.

#### Scenario: Some target points are infeasible

- **WHEN** part of the requested target grid exceeds feasible portfolio returns
- **THEN** infeasible points are excluded from the valid frontier and their
  count is retained in `n_infeasible`

### Requirement: Cash and weight constraints are explicit

The system SHALL support long-only or explicitly enabled short positions,
minimum and maximum asset weights, and an optional deterministic cash scenario
using the configured horizon return.

#### Scenario: Cash is included

- **WHEN** the user enables cash and supplies its horizon return
- **THEN** cash appears as an explicit scenario column and portfolio weight
  rather than an implicit residual

### Requirement: Expected-return and covariance assumptions are auditable

The optimizer SHALL accept explicit expected-return and covariance inputs and
SHALL record the estimator recipe, horizon, confidence level, scenario source,
random seed, and constraints used for a result.

#### Scenario: Robust assumptions are selected

- **WHEN** trimmed, winsorized, shrunk, or EWMA inputs are passed to the
  optimizer
- **THEN** the displayed and exported result identifies the actual estimator
  parameters received by the solver workflow

### Requirement: Maximum-Sharpe selection is bounded by its search process

When maximum-Sharpe analysis is used, the system SHALL report the risk-free
rate convention and feasible candidate process used. It MUST NOT imply a
globally optimal continuous solution when the implementation selects from a
finite candidate set.

#### Scenario: A maximum-Sharpe candidate is reported

- **WHEN** feasible candidates exist
- **THEN** the selected portfolio has the highest calculated Sharpe ratio among
  those candidates and the output states that scope

### Requirement: Solver output is independently checked

The system SHALL validate returned weights, budget, bounds, target return, and
CVaR constraints within documented numerical tolerances before presenting a
solution as valid.

#### Scenario: Solver status and constraints disagree

- **WHEN** a solver returns a nominal solution that violates post-solve checks
- **THEN** the system marks the result invalid rather than presenting it as an
  admissible portfolio

### Requirement: Optimization claims remain conditional

Public methodology documentation MUST disclose estimation error, in-sample
optimization risk, sensitivity to constraints, and the absence of transaction
costs, liquidity, slippage, taxes, and future-performance guarantees. UI
outputs MUST NOT claim out-of-sample superiority or suitability.

#### Scenario: An optimized portfolio dominates in sample

- **WHEN** optimized metrics are better than the current portfolio on the
  estimation scenarios
- **THEN** public interpretation limits the comparison to the selected
  scenarios and assumptions

## Mathematical Convention

For scenario losses `l_i(w) = -R_i w` and confidence level `beta`, the
implemented linear-programming objective is:

```text
minimize   t + [1 / ((1 - beta) N)] sum_i u_i
subject to u_i >= l_i(w) - t
           u_i >= 0
           sum_j w_j = 1
           portfolio constraints
```

The optimized `t` is the scenario VaR threshold and the objective is empirical
scenario CVaR.
