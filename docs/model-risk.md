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
- Monitoring avoids silent forward fill, but provider completeness, UTC cutoff,
  stale observations, symbol mapping, and later vendor corrections remain model
  and operational risks.

## Estimation risk

Sample means of volatile assets have low signal-to-noise ratios. Small changes
in the estimation window can reverse expected-return rankings and therefore
optimized allocations.

Sample covariance matrices are noisy; EWMA is more responsive but can overreact
to a short regime; linear shrinkage is more stable but depends on the selected
target and intensity. None is universally correct.

A covariance repair can make an estimate numerically usable without making it
economically credible. A material eigenvalue or symmetry adjustment is a model-
risk signal, not a quality certificate. The adjustment and diagnostics must be
retained with the result.

## Distribution and tail risk

- VaR and CVaR use signed loss space. A negative value represents a tail gain,
  not a calculation error; forcing every result positive would misstate the
  modeled return threshold.
- Multiplying a log-return risk value by capital is a linearized monetary
  equivalent, not exact scenario-level tail P&L.
- The application avoids that ambiguity in core results by using Simple
  returns for monetary risk, wealth paths, scenarios, and optimization. Log is
  an Advanced-mode distribution diagnostic only.
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
- Raw solver success is rejected when independent constraint residuals exceed
  tolerance. Passing residual checks establishes numerical consistency only;
  it does not validate the objective, inputs, or out-of-sample portfolio.
- Current results omit transaction costs, turnover, liquidity, capacity, taxes,
  custody risk, and execution slippage.
- A frozen optimizer snapshot can now be evaluated by Historical OOS replay or
  Live Forward monitoring, but this does not turn input estimates into truth or
  eliminate research selection and multiple-testing bias.

## Forward-testing and monitoring risk

- Historical Out-of-Sample Replay is evaluated after the historical outcomes
  already exist. It is not a genuine Live Forward Test, and choices influenced
  by the evaluation period can leak knowledge into the research design.
- Live Forward evidence begins only after a frozen launch. A short live record
  has low power and can be dominated by one regime.
- The asset universe is frozen from current inputs rather than reconstructed
  from a historical eligibility database. Historical and Hybrid experiments may
  therefore contain survivorship or selection bias.
- Holdings remain fixed after launch. Weight drift is measured, but no
  re-optimization or rebalancing is performed or recommended.
- Launch and valuation use complete daily closes. These are research marks, not
  evidence of executable fills.
- Fees, slippage, liquidity, capacity, taxes, custody, market impact, funding,
  and transaction constraints are absent. Reported NAV is gross of all of them.
- Current-exposure risk forecasts use drifted weights at the origin, but their
  expected-return, covariance, tail, and horizon assumptions remain uncertain.
- Overlapping horizon forecasts share returns. Their exception observations are
  not independent and must not be counted as equivalent to non-overlapping
  evidence.
- A missing observation produces an auditable gap rather than a fabricated
  price. The next complete return spans multiple calendar days and is excluded
  from the realized one-day volatility statistic.
- Comparing many experiment dashboards and reporting only the strongest path is
  backtest overfitting unless the search universe and rejected experiments are
  retained.
- Common-calendar and days-since-launch alignments answer different questions.
  Neither controls for different market regimes, recipes, assets, or constraints.
- Database persistence and provenance improve auditability; they do not validate
  performance or ensure the absence of data/vendor errors.

## Communication rules

Do not claim that the project:

- predicts future crypto losses accurately;
- finds the best portfolio;
- demonstrates outperformance;
- calls a Historical OOS replay a live forward test;
- treats drift as a rebalancing recommendation;
- is Basel/FRTB compliant;
- validates investment suitability;
- replaces licensed data or independent model validation.

Acceptable language describes the system as a research platform for comparing
how data, horizon, distribution, and estimator assumptions affect risk and
portfolio construction.

See [Forward testing](forward-testing.md) and
[Portfolio monitoring](portfolio-monitoring.md) for the experiment-specific
interpretation boundary.
