# Advanced Risk Layer Boundary Specification

## Purpose

Define the current public boundary for advanced risk capabilities that are not
implemented. This specification prevents planned research topics from being
misrepresented as working features.

## Requirements

### Requirement: Unimplemented advanced models are not exposed as available

The project MUST NOT claim implemented support for GARCH-family conditional
volatility, copula dependence, DCC correlation, automated stress testing, or
Euler risk-contribution decomposition until each capability is implemented,
tested, documented, and accepted through a reviewed OpenSpec change.

#### Scenario: Public documentation lists future research

- **WHEN** an unimplemented advanced model appears in a roadmap
- **THEN** it is labeled as planned and is not included in current feature or
  acceptance claims

#### Scenario: The application is executed

- **WHEN** a user inspects available model controls
- **THEN** the application does not offer an unimplemented advanced model as a
  working calculation choice

### Requirement: Advanced capabilities require explicit change control

Each advanced capability SHALL be introduced through a separate,
verb-led OpenSpec change containing purpose, scope, model assumptions,
validation criteria, implementation tasks, and model-risk disclosures.

#### Scenario: An advanced model is proposed

- **WHEN** work begins on a GARCH, copula, DCC, stress-testing, or
  risk-contribution feature
- **THEN** a strict-validating delta specification is reviewed before
  behavioral code is implemented

### Requirement: Advanced model validation is method-specific

Future advanced models MUST define tests appropriate to their statistical
assumptions and MUST NOT reuse passing VaR coverage tests as proof that
volatility, dependence, stress, or contribution estimates are valid.

#### Scenario: A future model reports successful VaR coverage

- **WHEN** an advanced model passes a VaR coverage diagnostic
- **THEN** its documentation limits that evidence to the tested VaR property
  and separately evaluates its other model assumptions

## Non-Normative Research Candidates

- GARCH(1,1), EGARCH, and GJR-GARCH conditional volatility.
- Gaussian and Student-t copulas with explicit tail-dependence diagnostics.
- Dynamic conditional correlation.
- Historical replay and hypothetical-shock stress testing.
- Marginal and component VaR/CVaR attribution.
