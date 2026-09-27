## ADDED Requirements

### Requirement: Retrospective policy-specific Risk Summary

The system MUST derive every visible Risk Summary portfolio statistic from the
finalized selected portfolio path and MUST describe it as historical rather
than predictive.

#### Scenario: Policy and costs affect the path
- **WHEN** the user changes Hold/Rebalance policy or proportional costs
- **THEN** ending net NAV, cumulative return, drawdown, costs and policy context
  reflect that exact audited path

#### Scenario: Initial peak precedes a loss
- **WHEN** the first post-launch interval loses value
- **THEN** maximum drawdown includes initial capital as the initial peak

### Requirement: Daily labels require daily observations

The system MUST expose relevant source and finalized-path quality defects and
MUST NOT label mixed-length observations as daily statistics.

#### Scenario: Calendar gap or incomplete close
- **WHEN** a finalized return interval spans other than one calendar day or an
  input row has missing prices
- **THEN** the cumulative NAV remains audited, a warning is displayed, and the
  daily descriptive and shape statistics are unavailable

#### Scenario: Clean completed daily sample
- **WHEN** all observations are finalized, complete, ordered and one day apart
- **THEN** the page shows a concise clean status and daily sample statistics

#### Scenario: Provisional current UTC date
- **WHEN** the source contains a row dated on or after the injected current UTC
  date
- **THEN** that row is retained for audit but excluded from finalized NAV,
  returns, drawdown, tail metrics, and sample counts

#### Scenario: Insufficient finalized history
- **WHEN** provisional-date exclusion leaves fewer than two finalized prices
- **THEN** analysis stops with a clear insufficient-history error and does not
  substitute the provisional mark

### Requirement: Historical tail interpretation

The system MUST label VaR/CVaR rows as historical-sample distribution
statistics and MUST identify monetary values as equivalents at ending NAV.

#### Scenario: Monetary tail value
- **WHEN** a return-space tail statistic is converted to money
- **THEN** the value preserves the model result's sign, uses ending net NAV and
  is explicitly distinguished from a realized historical loss

### Requirement: Machine-readable historical summary

The system MUST use one authoritative summary object for application display,
CSV export, and publication output.

#### Scenario: Long-form export
- **WHEN** Risk Summary is exported
- **THEN** it contains `Record Type`, `Section`, `Name`, `Value`, `Display Value`,
  `Unit`, and `Sample Size`, with raw numeric values preserved separately from
  display formatting and with context, quality, policy, weights, costs, dates,
  methodology, visible metrics, and rebalance audit represented

#### Scenario: Publication bundle
- **WHEN** a publication Risk Summary is generated
- **THEN** it is built from an explicit portfolio path and does not call the
  backward-compatible legacy summary helper
