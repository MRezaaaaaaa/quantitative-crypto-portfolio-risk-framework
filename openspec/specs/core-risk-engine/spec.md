# Core Risk Engine Specification

## Purpose

Define the implemented, reusable analytics contract for loading crypto price
data, calculating returns and portfolio losses, and estimating portfolio VaR,
CVaR, drawdown, and distribution statistics. Presentation, simulation,
backtesting, and optimization consume this contract without changing its
financial conventions.

## Requirements

### Requirement: Configurable price-data ingestion

The system SHALL load configured crypto assets from CoinGecko, yfinance, or a
local CSV source and SHALL return a date-indexed numeric price table. When
CoinGecko is selected and fails, the system SHALL use yfinance only when a
fallback source is configured.

#### Scenario: Primary crypto source succeeds

- **WHEN** CoinGecko is selected and returns valid observations
- **THEN** the system returns aligned prices using the configured asset symbols

#### Scenario: Configured fallback is required

- **WHEN** CoinGecko fails and yfinance is configured as the fallback
- **THEN** the system logs the fallback and attempts to return validated
  yfinance prices

#### Scenario: Offline CSV contains duplicate dates

- **WHEN** a CSV contains more than one observation for a date
- **THEN** the loader applies its documented last-observation rule before
  validation

#### Scenario: Offline CSV input is invalid

- **WHEN** CSV prices are empty, non-numeric, or non-positive
- **THEN** the system rejects the input with a clear validation error

### Requirement: Explicit return conventions

The system SHALL calculate either simple or log returns according to the
explicit caller or configuration choice. Crypto annualization SHALL default to
365 observations per year. Conversion between simple and log conventions SHALL
not occur implicitly.

#### Scenario: Simple returns are selected

- **WHEN** the return method is `simple`
- **THEN** the system calculates percentage price changes and removes only the
  initial undefined observation

#### Scenario: Log returns are selected

- **WHEN** the return method is `log`
- **THEN** the system calculates log price ratios and labels downstream results
  with the selected convention

### Requirement: Portfolio-weight validation

The system SHALL align portfolio weights to the asset-return columns, enforce
the configured short-selling policy, and require weights to sum to one after
any explicitly enabled normalization.

#### Scenario: Long-only weights are valid

- **WHEN** all configured weights are non-negative and sum to one
- **THEN** portfolio returns equal the row-wise weighted sum of asset returns

#### Scenario: A prohibited short position is supplied

- **WHEN** a negative weight is supplied while short selling is disabled
- **THEN** the system rejects the weights before calculating portfolio risk

### Requirement: Signed loss-space VaR estimates

The system SHALL implement Historical, Gaussian, and Cornish-Fisher VaR from
the left tail of portfolio returns and SHALL report VaR in signed loss space:
positive for a loss threshold, zero for break-even, and negative for a gain
threshold.

#### Scenario: Historical VaR is requested

- **WHEN** the caller supplies a valid confidence level and return series
- **THEN** the system returns the sign-reversed empirical left-tail return
  quantile

#### Scenario: Parametric VaR is requested

- **WHEN** Gaussian or Cornish-Fisher VaR is selected
- **THEN** the system applies the selected distributional approximation and
  preserves the signed loss-space convention

### Requirement: Signed loss-space CVaR estimates

The system SHALL implement Historical and Gaussian CVaR and SHALL report CVaR
in the same signed loss-space convention as VaR.

#### Scenario: Historical CVaR is requested

- **WHEN** historical observations exist at or beyond the empirical VaR
  threshold
- **THEN** the system returns the sign-reversed mean of the left-tail returns

#### Scenario: Gaussian CVaR is requested

- **WHEN** Gaussian CVaR is selected with a valid confidence level
- **THEN** the system applies the analytical Normal expected-shortfall formula

### Requirement: Horizon scaling is explicit

The system SHALL identify square-root-of-time scaling as an i.i.d.
approximation and SHALL keep it distinct from realized multi-day return
aggregation.

#### Scenario: One-day volatility is scaled

- **WHEN** a caller requests an approximate multi-day volatility or VaR scale
- **THEN** the system applies the documented square-root-of-time convention
  without claiming it is a realized multi-day distribution

### Requirement: Portfolio risk summary and exports

The system SHALL provide portfolio-level return, volatility, skewness,
kurtosis, drawdown, VaR, and CVaR outputs and SHALL support explicit export of
tables and figures to caller-selected destinations.

#### Scenario: A risk summary is generated

- **WHEN** valid portfolio returns, confidence level, and capital are supplied
- **THEN** the output identifies metric values, units, and monetary loss
  equivalents where applicable

#### Scenario: Export is disabled

- **WHEN** no output destination is supplied
- **THEN** the analytics return in memory without creating generated files

### Requirement: Core analytics remain dependency-light

Core calculation modules MUST NOT import Streamlit or perform implicit network
requests. Deterministic unit tests MUST exercise calculation behavior without
live APIs.

#### Scenario: The package is imported for analytics

- **WHEN** `var_cvar_crypto_risk` or a calculation module is imported
- **THEN** no dashboard starts and no external data request is made

## Implementation Boundaries

- Streamlit presentation is specified by `streamlit-dashboard`.
- VaR forecast validation is specified by `backtesting`.
- Parametric scenario generation is specified by `monte-carlo-engine`.
- Scenario-based portfolio decisions are specified by `cvar-optimization`.
- GARCH, copula, DCC, stress-testing, and risk-contribution models are outside
  the current implemented contract.
