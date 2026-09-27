# Update Phase 8 to manual portfolio monitoring

## Why

The monitoring creation UI currently rebuilds an optimizer. The requested Phase
8 update is a manual-only creation workflow: users declare held assets and
initial weights; monitoring must not choose or optimize an allocation.

## What changes

- Accept explicit long-only weights summing to one without silent normalization.
- Freeze a manual allocation snapshot and derive quantities from launch prices.
- Remove optimizer, expected-return, covariance and scenario controls from the
  monitoring creation UI; retain risk settings, prices and date boundaries.
- Reuse fixed-holdings valuation, historical replay, live updates and persistence.
- Preserve existing optimized experiments without rewriting their history.
- Leave the separate Risk Lab optimizer unchanged.

## Model risk

Historical manual allocations may have been selected with hindsight. A stated
decision date does not prove the weights were known then. Risk forecasts remain
origin-safe, but replay cannot certify allocation-selection independence. No
trading, fees, rebalancing or performance guarantee is introduced.

## Authority

The user explicitly requested implementing manual-only monitoring as a Phase 8
update on 2026-09-18. No release, tag, push or deployment is part of this update.
