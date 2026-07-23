# Phase 3 Proposal — VaR Backtesting & Model Validation

## Problem

Phase 1 and Phase 2 give the user a *number* (Historical, Gaussian, or
Cornish-Fisher VaR / CVaR) but no evidence that the number is right. A
4.2% one-day VaR at 95% confidence is meaningful only if, over a long
historical sample, the realised loss exceeds 4.2% on roughly 5% of days
**and** those breaches are not all bunched together in a few volatile
months. Without that check, the engine reports risk it cannot defend —
a serious omission given that crypto markets switch regime far more often
than equity markets.

## Why VaR Must Be Validated

The Basel III Market Risk framework (BCBS 2019) requires every internal
VaR model to undergo daily backtesting against a 250-trading-day window
of clean P&L. Models that produce too many or too few breaches are
penalised through the supervisory multiplier `m`. The most widely used
academic tools to formalise that judgement are:

1. **Kupiec POF** — does the *frequency* of breaches match the model?
2. **Christoffersen Independence** — are the breaches *independent* in time?
3. **Christoffersen Conditional Coverage** — both at once.

These three tests, plus the Basel-III three-zone traffic light, are the
minimum viable backtesting suite for any market-risk model. Phase 3
ports that suite to crypto.

## What Phase 3 Implements

- A no-look-ahead `rolling_var_forecast()` engine.
- Breach detection (`actual_return < -var_forecast`) and breach analytics.
- The three statistical tests (Kupiec POF, Christoffersen Independence,
  Christoffersen Conditional Coverage).
- Two-tier traffic light: Basel III absolute counts + a generalised
  rate-based variant for non-standard windows.
- Multi-method comparison so the user can see which of Historical /
  Gaussian / Cornish-Fisher actually validates against their portfolio.
- Three new matplotlib charts — backtest overlay, breach timeline, and
  model comparison.
- A new Streamlit "Backtesting & Model Validation" tab.
- A 50-test pytest suite covering every public function and edge case.

## What Phase 3 Explicitly Excludes

- Monte Carlo simulation (Phase 4).
- CVaR / Expected Shortfall backtesting under FRTB (Phase 5).
- GARCH / Filtered Historical Simulation / DCC / copula models (Phase 6).
- Stress testing, risk contributions, optimisation under backtested
  constraints.

## Expected Outcomes

After Phase 3 the user can:

1. Choose any of the three VaR methods and see whether it actually holds
   up on their realised portfolio history.
2. Compare the three methods side by side and pick the one with the best
   coverage for their data.
3. Spot breach clustering visually (timeline) and statistically
   (independence test).
4. Read a Basel-aligned traffic-light verdict on each model.
5. Export the forecast series, the result JSON, and the comparison CSV
   for portfolio review or résumé portfolios.

The architecture of the engine remains modular: `backtesting.py` is a
pure-function module, plotting stays in `plotting.py`, the Streamlit tab
imports both. No existing Phase 1/2 code is modified beyond additive
edits in the import block, the tab list, and the appended tab body.
