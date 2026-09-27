# Rebuild retrospective Risk Summary and migrate Risk Lab charts to Plotly

## Why

Risk Summary mixed historical outcomes with annualized and risk-estimate labels
that could be read as forecasts, while Risk Lab maintained two presentation
stacks.  A publication-facing application needs one audited historical path,
explicit data-quality disclosure, and portable interactive figures without
duplicating financial calculations in the chart layer.

## What changes

- Make the finalized selected portfolio path the sole source for Risk Summary
  headline, descriptive and tail-distribution rows.
- Expose calendar gaps, incomplete prices, mixed return intervals, duplicate or
  unsorted source dates, and a partial current UTC day.
- Remove annualized return, annualized volatility and Sharpe from the visible
  retrospective summary and its CSV.
- Migrate every user-facing Risk Lab Matplotlib figure to pure Plotly builders
  with a shared theme, provenance metadata and standalone HTML export.
- Make the selected Pearson or Spearman method govern rolling correlation as
  well as the static correlation diagnostics.

## Boundaries

This change does not modify portfolio accounting, transaction-cost semantics,
VaR/CVaR formulas, backtesting tests, simulation or optimization behavior.  It
does not make historical statistics predictive.
