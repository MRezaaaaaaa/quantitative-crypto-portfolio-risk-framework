# Proposal: Add robust expected-return dumbbell audit

## Why

The Robust Assumptions table exposes every estimator, but the original chart
shows only the raw mean and one selected endpoint. Repeated selector changes do
not reveal the full cross-estimator range or estimator dependence at a glance.

## What changes

- Make All Estimators the default view, with one stable marker per completed
  estimator and a min-to-max range connector for every asset.
- Retain the Raw Historical Mean-versus-one-estimator chart as Pairwise
  Comparison.
- Resolve coincident estimates through deterministic y-offsets without changing
  the x-values supplied by the assumptions table.
- Preserve portfolio order by default and offer full estimator-dispersion sorting.
- Store the complete table values, Final E[r], configuration, range and marker
  contract in Plotly metadata and hover details.

## Out of scope

- Re-estimating expected returns inside the chart layer.
- Changing the assumptions engine, scenario matrix, optimizer input, or
  expected-return methodology.
- Describing any displayed estimate as a forecast.
