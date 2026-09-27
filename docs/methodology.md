# Methodology and Horizon Conventions

## Return conventions

For price (P_t):

- Simple return: `P_t / P_(t-1) - 1`
- Log return: `log(P_t / P_(t-1))`
- Simple h-period return: `product(1 + r_i) - 1`
- Log h-period return: `sum(r_i)`

The return convention must accompany every stored or exported result. Simple
and log returns are not interchangeable when aggregating wealth or horizons.

The application uses Simple returns for portfolio construction, NAV,
headline risk, backtesting, scenarios, Monte Carlo, robust-risk inputs, and
optimization. Automatic mode uses Simple returns everywhere. Advanced mode can
select Log returns for distribution diagnostics only. Portfolio diagnostic Log
returns are derived exactly from reconstructed asset gross returns, not from a
weighted average of asset Log returns. See [Return conventions](return-conventions.md).

## Retrospective Risk Summary

Risk Summary answers only what happened historically under the selected
portfolio-evolution policy. Ending net NAV, net cumulative return, maximum
drawdown and transaction costs come from the finalized Portfolio Path V1 net
history. Maximum drawdown includes initial capital as the launch peak. The page
does not rebuild a constant-weight return series and does not present sample
statistics as forecasts.

Before the path is built, a source row dated on the current UTC calendar day is
classified as provisional. It remains available for audit but is excluded from
finalized NAV, returns, drawdown, tail observations, and sample counts. The same
rule applies to later-dated source rows. If fewer than two finalized observations
remain, analysis stops with an explicit insufficient-history error; the engine
does not substitute the provisional mark.

Mean, volatility, minimum and maximum are labeled daily only when every included
finalized return spans exactly one calendar day and source-quality checks permit
that label. Missing prices, calendar gaps, source duplicate/order defects and
excluded provisional rows are disclosed. Missing finalized observations are not
silently forward-filled or removed from cumulative NAV merely to make daily
statistics available. Skewness and excess kurtosis are secondary descriptive
moments and show `N/A` when the daily sample is invalid or insufficient.

The separate Historical Tail Distribution retains the existing VaR/CVaR models.
Its values describe the finalized historical return sample, not future losses.
A monetary equivalent multiplies the return-space statistic by ending net NAV;
it is not a realized historical loss.

## Portfolio evolution in Risk Lab

Risk Lab no longer treats `asset_returns @ constant_weights` as a generic
portfolio history. That arithmetic is retained only as the explicitly labeled
zero-cost **Daily Rebalanced — legacy constant-weight** policy. The selected
policy is applied by the pure Portfolio Path V1 engine:

- **Buy & Hold:** quantities established at launch remain fixed; weights drift.
- **Daily rebalance:** every later complete observation resets post-cost weights
  to target.
- **Weekly:** theoretical UTC `W-SUN` boundary.
- **Monthly/quarterly:** theoretical UTC calendar month-end/quarter-end.

At launch, `quantity_i = initial_capital * weight_i / launch_price_i`. The
allocation is treated as already established, so V1 charges no setup cost. On a
later date, old quantities are marked at that complete close before any trade.
If scheduled, a rebalance executes at that close; its new quantities affect the
next return interval, never the interval that just ended. Close execution is an
idealized research mark, not evidence of an executable fill.

The event calendar is generated independently of observed price dates. A
missing scheduled date can therefore never be replaced by an earlier observed
date. Missing prices are not forward-filled for the path. An absent boundary or
a scheduled row with an incomplete cross-asset close is retained as pending,
then executed on the first later complete observation with both scheduled and
effective dates recorded. When several scheduled boundaries are pending before
that observation, the engine records every boundary but coalesces them into one
target-reset trade and one transaction-cost charge. V1 stateful paths are
long-only and unlevered because margin, borrow, funding and short-sale cash flows
are not modeled.

Asset-level drawdown uses initial wealth `1.0` as the launch peak. Therefore a
first-period loss is visible immediately: returns `[-20%, +12.5%]` produce
drawdowns `[-20%, -10%]`, not `[0%, 0%]`. Stateful NAV-path drawdown uses the
same launch-peak convention and is not reconstructed from a separate return
series.

For combined commission and slippage rate `k`:

```text
k = (commission_bps + slippage_bps) / 10_000
gross_traded_notional = sum(abs(trade_value_i))
transaction_cost = k * gross_traded_notional
one_way_turnover = 0.5 * gross_turnover
```

Post-cost NAV is solved jointly with the target allocation so reported
post-trade weights still sum to the requested target. Portfolio and asset
exports carry policy, target/initial weights, event convention, costs, missing-
price rule and `portfolio-path-v1` provenance.

Risk Summary CSV is a long-form contract with columns `Record Type`, `Section`,
`Name`, `Value`, `Display Value`, `Unit`, and `Sample Size`. `Value` preserves
the raw machine-readable number when one exists; `Display Value` is the exact
human-facing representation. Context, visible metrics, data-quality counts and
details, excluded provisional dates, rebalance audit, methodology, target
weights, costs, source, and date bounds are exported from the same immutable
summary object rendered in the application.

## VaR and CVaR

At confidence level `c`, the implementation evaluates the left tail at
`alpha = 1 - c` and reports the negative return quantile in signed loss space.
Outputs are decimal values: positive means loss, zero means break-even, and
negative means the selected tail threshold remains profitable.

- Historical VaR uses the empirical quantile.
- Gaussian VaR uses the sample mean and standard deviation under Normality.
- Cornish-Fisher VaR adjusts the Normal quantile with sample skewness and excess
  kurtosis.
- Historical CVaR averages observations at or below the empirical VaR return
  threshold.
- Gaussian CVaR uses the analytical Normal Expected Shortfall expression.

Cornish-Fisher is a truncated moment expansion. Extreme or unstable moment
estimates can produce distorted or non-monotone quantiles; it must be compared
with other methods rather than treated as automatically superior.

Negative VaR or CVaR is valid in an all-gain sample and must not be converted
with `abs()` or silently clamped to zero. Monetary values preserve the same sign.
See the [VaR and CVaR output contract](risk-measure-contract.md) for the exact
sign, unit, horizon, backtesting, and optimization conventions.

## Horizon map

Different components answer different questions and therefore use different
horizon constructions.

| Component | Construction | Main limitation |
|---|---|---|
| Portfolio path | Stateful quantities under the selected rebalance calendar | Idealized close execution; proportional costs only |
| Risk Summary | Finalized selected-policy net path and return sample | Retrospective only; source/calendar quality limits daily labels |
| Distribution diagnostics | Realized rolling h-period returns | Overlapping observations |
| VaR backtesting | Rolling historical estimate versus forward realized h-period return | Independence depends on stepping mode |
| Historical optimization scenarios | Rolling h-period asset returns with equal scenario probability | Overlapping observations |
| Parametric Monte Carlo | Mean multiplied by `h`, covariance multiplied by `h` | Constant moments and i.i.d. increments |
| Robust expected returns | Estimated from the selected scenario/observation horizon | High estimation error |
| Robust covariance | Estimated from daily returns; volatility may be displayed at `sqrt(h)` | Daily dependence may not persist |
| Monitoring realized volatility | Expanding sample standard deviation of eligible one-calendar-day post-launch Simple returns, annualized with `sqrt(365)` | Short sample; excludes multi-day gaps |
| Monitoring risk forecast | Origin-safe estimate for a stored future target using current drifted weights | Overlap and estimation uncertainty |

The risk horizon never determines the rebalance calendar. Headline Money
VaR/CVaR described as current risk multiplies return-space risk by the latest
net policy NAV and records that NAV's date/type; initial capital remains a
separate launch notional.

The Robust Assumptions comparison reads the completed transparency table; it
does not rebuild any estimator. By default, every available estimator and the
exact Final E[r] passed downstream appear in one asset band. The connector spans
`max(available estimates) - min(available estimates)`, and optional dispersion
sorting uses that full range. Deterministic vertical marker offsets resolve
equal x-values without modifying the assumptions. Pairwise mode retains Raw
Historical Mean as its fixed first endpoint. Displayed percentage points are the
table's decimal returns multiplied by 100; signed basis-point differences are
`(estimate - mean) * 10,000`. Missing Manual Views remain unavailable rather
than becoming zero. The figure is an assumption-sensitivity audit, not an
expected-performance forecast.

## Backtesting

For forecast position `t`, the estimator uses only observations in
`[t-window, t)`. The realized return begins at `t`, preventing direct
look-ahead.

Two stepping modes exist:

- `overlapping`: advances one observation at a time;
- `non_overlapping`: advances by the full horizon.

Overlapping horizons share underlying returns. Kupiec frequency results may
still be descriptive, but Christoffersen independence claims require special
caution. Non-overlapping evaluation is the more defensible option for an
independence test, at the cost of a smaller sample.

The project implements:

- Kupiec Proportion of Failures;
- Christoffersen independence;
- Christoffersen conditional coverage;
- Basel-inspired and rate-based traffic-light summaries.

These components do not constitute a complete regulatory validation framework.
For an all-breach or no-breach hit sequence, one previous-state transition row
is never observed. Both Markov transition probabilities therefore cannot be
identified: the independence statistic, p-value, and conditional-coverage test
are reported as `inconclusive`, not `pass`. The breach-frequency traffic light
remains a separate descriptive result and must not be interpreted as proof that
independence or full model validity has been established.

## Monte Carlo

Normal and Student-t scenarios use historical mean and covariance inputs unless
the robust assumptions engine supplies an alternative covariance. Multi-period
moments use linear time scaling. A seed makes a given configuration repeatable,
but it does not remove simulation or parameter uncertainty.

The Student-t implementation rescales its dispersion so theoretical covariance
matches the supplied covariance when degrees of freedom exceed two. A heavier
tail is not evidence that the chosen degrees of freedom or dependence structure
is correct.

All scenario returns and simulated wealth-path innovations are represented as
Simple returns. The public scenario and optimization boundaries reject a Log
input declaration rather than applying incompatible arithmetic.

Monte Carlo current-exposure aggregation and the optimizer's `Current`
comparator use end-of-path weights available at the Risk Lab cutoff. Historical
VaR backtesting consumes the sequential realized returns produced by the
selected policy; it does not apply final-period weights retrospectively.

Before Normal or Student-t simulation, the covariance input is checked for
labels, finite values, symmetry, eigenvalues, and conditioning. The default
policy repairs a numerically invalid matrix in correlation space while
preserving marginal variances; strict mode rejects it. Repair stabilizes the
linear algebra but does not validate the economic covariance assumption. See
[Covariance and solver governance](covariance-and-solver-governance.md).

## Robust assumptions

Expected-return choices include mean, median, trimmed mean, winsorized mean,
shrinkage toward zero, zero, and manual views. Risk choices include sample,
EWMA, and manually weighted linear covariance shrinkage.

EWMA uses exponentially decaying daily weights and a zero-mean RiskMetrics-style
convention. Linear shrinkage intensity is user-selected; it is not an estimated
Ledoit-Wolf optimum.

## CVaR optimization

The optimizer uses scenario returns `R` and weights `w`, with loss `-R @ w`.
The Rockafellar-Uryasev auxiliary-variable formulation minimizes empirical CVaR
subject to the selected budget, box, cash, target-return, or CVaR-cap
constraints.

Solver success is provisional. Returned weights and available auxiliary
variables are independently checked against the budget, box, target-return,
CVaR-cap, and Rockafellar-Uryasev constraints. A result that exceeds the
configured tolerance is labeled `validation_failed` even if the raw solver
status was successful.

The same scenarios are generally used to estimate inputs and evaluate the
result. Metrics are therefore in-sample estimates. Portfolio weights should not
be described as out-of-sample performance or as the unique best portfolio.

## Portfolio experiment monitoring

New experiments freeze manually entered positive long-only weights totaling 100%
without optimization, expected-return fitting, scenario generation or silent
normalization. The user-selected completed launch day requires complete prices.
Legacy internal `optimization_as_of` means the allocation decision date in this
manual workflow; historical weight selection may still contain hindsight bias.
Launch NAV equals
initial capital, and launch return is zero. Post-launch quantities are fixed and
Simple-return wealth arithmetic is used. Price moves create current-weight
drift; no rebalancing is performed.

Monitoring forecasts end their information set at the origin and use current
drifted weights by default. Realized loss is attached only when the target
matures. A VaR exception is `realized_loss > forecast_var`; CVaR is tail severity
and is not an exception threshold. Historical replay reveals evaluation dates
sequentially but remains retrospective research rather than a Live Forward Test.

See [Forward testing](forward-testing.md) and
[Portfolio monitoring](portfolio-monitoring.md).

## References

- Rockafellar, R. T. and Uryasev, S. (2000), *Optimization of Conditional
  Value-at-Risk*.
- Kupiec, P. H. (1995), *Techniques for Verifying the Accuracy of Risk
  Measurement Models*.
- Christoffersen, P. F. (1998), *Evaluating Interval Forecasts*.
- J.P. Morgan/Reuters (1996), *RiskMetrics Technical Document*.
