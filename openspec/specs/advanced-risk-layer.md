# Advanced Risk Layer

> **Status: Planned**
> Implement only through a separately reviewed OpenSpec change.

## Purpose

Extend the engine with conditional volatility and dependence models that are
indispensable for crypto: GARCH-family univariate volatility, copula-based
joint distributions, and dynamic conditional correlation (DCC).

## Planned Scope

- Univariate GARCH(1,1), EGARCH, GJR-GARCH volatility forecasts.
- VaR / CVaR conditional on the forecast volatility.
- Copula models (Gaussian, Student-t, Clayton, Gumbel) for joint tail
  dependence between assets.
- DCC-GARCH for time-varying correlation structures.
- Stress testing scaffold: historical replay, hypothetical shocks, copula
  scenario generation.
- Risk contribution / decomposition (Euler allocation) per asset.
- Tail-dependence diagnostics and visual reports.

## Dependencies

Requires the implemented core risk, covariance-governance, scenario, and
validation contracts.
