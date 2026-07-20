# Model Risk

## Purpose

This document defines how results should be interpreted and which claims the
project does not support. Model transparency is part of the product, not a
disclaimer added after analysis.

## Data risk

- Free vendor data can contain revisions, gaps, inconsistent timestamps, or
  source-specific conventions.
- Inner joins can shorten the common sample and change the observed regime.
- Limited forward filling can create synthetic unchanged prices.
- A current asset list is not a historical point-in-time universe; selection
  and survivorship bias remain.
- Mixed crypto and traditional assets have different calendars and market-close
  conventions.

## Estimation risk

Sample means of volatile assets have low signal-to-noise ratios. Small changes
in the estimation window can reverse expected-return rankings and therefore
optimized allocations.

Sample covariance matrices are noisy; EWMA is more responsive but can overreact
to a short regime; linear shrinkage is more stable but depends on the selected
target and intensity. None is universally correct.

## Distribution and tail risk

- Gaussian models underrepresent skewness, jumps, and heavy tails.
- Student-t models add heavy tails but retain a fixed parametric structure.
- Historical simulation cannot generate events absent from the sample.
- Cornish-Fisher depends on unstable higher moments and can generate poor
  approximations.
- VaR is a quantile and says nothing about the size of losses beyond it.
- CVaR describes the modeled tail; it is not a maximum loss.

## Time and regime risk

Square-root/time-linear scaling assumes independent, identically distributed
increments and stable moments. Crypto returns exhibit volatility clustering,
jumps, changing correlations, and structural breaks.

EWMA reflects recent observations through weighting; it does not formally
identify economic regimes.

## Backtesting risk

- Passing a statistical test is failure to reject a null, not proof that a model
  is correct.
- Small samples have low power.
- Overlapping horizons violate simple independence assumptions.
- Selecting a method after comparing many backtests introduces selection bias.
- An all-breach or no-breach sequence cannot identify both transition
  probabilities required by the Christoffersen independence test; the project
  reports independence and conditional coverage as inconclusive in this case.
- A green traffic light summarizes breach frequency only. It does not override
  an inconclusive independence test or establish full model validity.

## Optimization risk

- Optimizers amplify input error.
- Constraints can determine allocations more strongly than the objective.
- A high expected-return estimate can dominate a tail-risk objective unless
  explicitly constrained.
- `optimal_inaccurate` is a numerical warning, not a clean economic optimum.
- Current results omit transaction costs, turnover, liquidity, capacity, taxes,
  custody risk, and execution slippage.
- Current comparisons are not walk-forward portfolio backtests.

## Communication rules

Do not claim that the project:

- predicts future crypto losses accurately;
- finds the best portfolio;
- demonstrates outperformance;
- is Basel/FRTB compliant;
- validates investment suitability;
- replaces licensed data or independent model validation.

Acceptable language describes the system as a research platform for comparing
how data, horizon, distribution, and estimator assumptions affect risk and
portfolio construction.
