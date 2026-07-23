# Backtesting and Model Validation Specification

## Purpose

Define the implemented diagnostics for comparing VaR forecasts with realized
portfolio returns. Coverage and independence tests provide model evidence;
they do not by themselves establish complete model validity, future
performance, or regulatory compliance.

## Requirements

### Requirement: Strictly out-of-sample rolling forecasts

The system SHALL calculate every forecast for date `t` using observations
available strictly before `t`. The realized return evaluated at `t` MUST NOT be
included in its estimation window.

#### Scenario: A one-step forecast is created

- **WHEN** the rolling window ends immediately before date `t`
- **THEN** the system estimates VaR from that window and compares it with the
  return realized at `t`

#### Scenario: The estimation window is insufficient

- **WHEN** the return history is not longer than the selected window
- **THEN** the system rejects the backtest instead of leaking future data

### Requirement: Horizon and observation mode are explicit

The system SHALL record the selected horizon and whether horizon observations
are overlapping or non-overlapping. Results from different observation modes
MUST NOT be presented as directly equivalent without a dependence warning.

#### Scenario: Overlapping multi-day returns are selected

- **WHEN** the horizon exceeds one day and overlapping mode is used
- **THEN** the system retains the larger sample and identifies the induced
  serial dependence

#### Scenario: Non-overlapping multi-day returns are selected

- **WHEN** non-overlapping mode is used
- **THEN** each realized horizon return uses a disjoint observation block

### Requirement: VaR breach classification

The system SHALL classify a breach when the realized portfolio loss exceeds
the positive VaR forecast and SHALL report observations, actual breaches,
expected breaches, breach rates, and the breach ratio.

#### Scenario: Realized loss exceeds VaR

- **WHEN** the realized return is less than the negative VaR magnitude
- **THEN** the observation is marked as a breach

#### Scenario: Realized loss remains inside VaR

- **WHEN** the realized return is greater than or equal to the negative VaR
  magnitude
- **THEN** the observation is not marked as a breach

### Requirement: Kupiec unconditional-coverage diagnostic

The system SHALL calculate the Kupiec proportion-of-failures likelihood-ratio
statistic and p-value using the expected breach probability implied by the
confidence level. Boundary cases with zero or all breaches SHALL produce
defined results rather than NaN.

#### Scenario: Observed breach frequency is evaluated

- **WHEN** a non-empty Boolean breach sequence and confidence level are supplied
- **THEN** the output includes the likelihood-ratio statistic, p-value, and
  pass/fail interpretation at the configured significance level

### Requirement: Christoffersen dependence diagnostics

The system SHALL calculate Christoffersen independence transition counts and
the conditional-coverage statistic that combines unconditional coverage and
independence evidence.

#### Scenario: Breaches are clustered

- **WHEN** consecutive breach transitions occur more often than implied by
  independence
- **THEN** the independence diagnostic reflects that pattern in its statistic
  and p-value

#### Scenario: The sequence is degenerate

- **WHEN** the sequence is all breaches, no breaches, or too short for stable
  transitions
- **THEN** the system returns a defined diagnostic or an explicit
  not-applicable interpretation without crashing

### Requirement: Traffic-light interpretation is scoped

The system SHALL support an absolute-count mode for an approximately 250-day
window and a generalized rate-based mode for other sample lengths. The
selected mode SHALL be visible in the result.

#### Scenario: Auto mode receives a standard-length window

- **WHEN** auto mode receives between 240 and 260 observations
- **THEN** the system selects the absolute-count traffic-light logic

#### Scenario: Auto mode receives a non-standard window

- **WHEN** auto mode receives any other valid observation count
- **THEN** the system selects the generalized rate-based logic

### Requirement: Supported VaR models are compared consistently

The system SHALL backtest Historical, Gaussian, and Cornish-Fisher VaR through
the same forecast, breach, and diagnostic pipeline.

#### Scenario: One model fails during comparison

- **WHEN** one requested model raises an estimation error
- **THEN** the comparison records that model's error and continues evaluating
  the remaining models

### Requirement: Backtest outputs are auditable

The system SHALL expose forecast dates, realized returns, VaR forecasts,
breaches, test statistics, p-values, traffic-light mode, horizon convention,
and model identifier in tabular or structured outputs.

#### Scenario: A user downloads a backtest

- **WHEN** a completed backtest is exported
- **THEN** the exported records contain enough metadata to identify the model,
  horizon, confidence level, window, and observation mode

### Requirement: Backtesting claims remain bounded

The application and documentation MUST describe Kupiec and Christoffersen
outputs as diagnostics and MUST NOT present a passing result as proof of future
accuracy, Expected Shortfall validity, or regulatory approval.

#### Scenario: A model passes both diagnostics

- **WHEN** the reported p-values exceed the selected significance threshold
- **THEN** the interpretation states only that the observed sample did not
  reject the tested coverage and independence hypotheses

## Statistical Notes

- Overlapping horizon returns reduce the effective independence of
  observations.
- Multiple model comparisons increase false-discovery risk.
- Structural breaks and volatility clustering can invalidate unconditional
  assumptions even when a historical backtest passes.
- Expected Shortfall backtesting and conditional-volatility tests are not part
  of this capability.
