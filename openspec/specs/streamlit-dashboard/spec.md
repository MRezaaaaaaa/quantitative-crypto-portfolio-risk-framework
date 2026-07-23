# Streamlit Dashboard Specification

## Purpose

Define the implemented single-page, tabbed Streamlit interface for configuring
crypto portfolios and inspecting risk measurement, diagnostics, simulations,
assumptions, and optimization results without placing financial calculations
inside the UI layer.

## Requirements

### Requirement: Explicit data and portfolio configuration

The dashboard SHALL expose data source, assets, date range, return convention,
portfolio weights, confidence level, and risk horizon controls before running
analysis.

#### Scenario: The user runs a configured analysis

- **WHEN** the user presses `Run risk analysis` with valid inputs
- **THEN** the dashboard stores a new risk-result snapshot for that
  configuration

### Requirement: Shared validated analysis state

The dashboard SHALL derive its tabs from a shared validated risk-result
snapshot. A successful new core run SHALL clear dependent backtest, Monte
Carlo, assumption, and optimization state.

#### Scenario: Input validation fails

- **WHEN** prices, weights, dates, or configuration are invalid
- **THEN** downstream tabs display a clear blocking message rather than
  calculating from partial state

#### Scenario: A new core run succeeds

- **WHEN** a valid analysis replaces the stored risk-result snapshot
- **THEN** dependent result snapshots are reset before they can be reused

### Requirement: Risk and distribution views

The dashboard SHALL present portfolio overview, distribution diagnostics,
VaR/CVaR comparisons, cumulative performance, drawdown, and asset-level
context using the selected return and horizon conventions.

#### Scenario: Multi-day distribution analysis is displayed

- **WHEN** the selected horizon exceeds one day
- **THEN** the dashboard identifies whether the chart uses realized aggregated
  returns or an analytical scaling approximation

### Requirement: Backtesting view

The dashboard SHALL expose backtest model, window, horizon mode, breaches,
coverage diagnostics, independence diagnostics, and traffic-light
interpretation.

#### Scenario: A backtest completes

- **WHEN** sufficient historical observations exist
- **THEN** KPI cards, forecast charts, breach views, and downloadable
  diagnostics correspond to the same backtest configuration

### Requirement: Monte Carlo view

The dashboard SHALL expose distribution, scenario count, horizon, seed, and
Student-t degrees of freedom where applicable and SHALL visualize scenario
risk and portfolio paths.

#### Scenario: The simulation distribution changes

- **WHEN** the user switches between Normal and Student-t simulation
- **THEN** all displayed simulation outputs are regenerated from the selected
  distribution and its disclosed parameters

### Requirement: Robust-assumption view

The dashboard SHALL display expected-return estimators, volatility estimates,
covariance estimators, shrinkage parameters, and EWMA decay with their units
and horizon conventions.

#### Scenario: EWMA covariance is selected

- **WHEN** the user selects EWMA and supplies a decay parameter
- **THEN** the resulting table and heatmap identify EWMA and the actual decay
  value used

### Requirement: Optimization view

The dashboard SHALL expose objective, scenario source, expected-return recipe,
constraints, optional cash, and solver diagnostics and SHALL display the
inputs actually passed into the optimization workflow.

#### Scenario: An optimization is infeasible

- **WHEN** requested constraints have no admissible solution
- **THEN** the dashboard displays the infeasibility reason and does not present
  stale or fallback weights as the requested optimum

### Requirement: Exported artifacts retain context

Downloadable tables and figures SHALL correspond to the currently displayed
analysis state and SHALL include sufficient labels or accompanying metadata to
identify method, horizon, confidence, and relevant assumptions.

#### Scenario: A user downloads a result

- **WHEN** a download control is used
- **THEN** the artifact represents the currently rendered configuration rather
  than an earlier cached run

### Requirement: UI and analytics remain separated

The Streamlit module SHALL call tested analytics modules for calculations.
Core analytics modules MUST NOT import Streamlit.

#### Scenario: Analytics are tested without Streamlit

- **WHEN** the core test suite imports risk, backtesting, simulation, or
  optimization modules
- **THEN** those modules execute without initializing a Streamlit runtime

### Requirement: Dashboard claims remain research-scoped

The dashboard MUST use conditional analytical language and MUST NOT claim
investment advice, guaranteed performance, or regulatory approval.

#### Scenario: A risk or optimization result is shown

- **WHEN** the dashboard displays VaR, CVaR, a passing backtest, or optimized
  weights
- **THEN** the result is labeled as a model output or diagnostic rather than a
  guaranteed forecast or recommendation

## Current Boundary

- Live sources are CoinGecko and yfinance; the analytics layer also supports
  local CSV prices, but the current dashboard does not expose file upload.
- The current interface is a single `app.py` with tabs, not a multi-page app.
- Sidebar edits do not invalidate an already stored snapshot until the user
  runs the core analysis again; displayed results therefore remain the prior
  run until explicit recomputation.
- Persistent optimized-portfolio monitoring and rebalancing are not yet
  implemented.
