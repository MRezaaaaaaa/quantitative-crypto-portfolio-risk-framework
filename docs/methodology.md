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
| Headline scaled risk | Daily estimate scaled by `sqrt(h)` | i.i.d. approximation |
| Distribution diagnostics | Realized rolling h-period returns | Overlapping observations |
| VaR backtesting | Rolling historical estimate versus forward realized h-period return | Independence depends on stepping mode |
| Historical optimization scenarios | Rolling h-period asset returns with equal scenario probability | Overlapping observations |
| Parametric Monte Carlo | Mean multiplied by `h`, covariance multiplied by `h` | Constant moments and i.i.d. increments |
| Robust expected returns | Estimated from the selected scenario/observation horizon | High estimation error |
| Robust covariance | Estimated from daily returns; volatility may be displayed at `sqrt(h)` | Daily dependence may not persist |
| Monitoring realized volatility | Expanding sample standard deviation of eligible one-calendar-day post-launch Simple returns, annualized with `sqrt(365)` | Short sample; excludes multi-day gaps |
| Monitoring risk forecast | Origin-safe estimate for a stored future target using current drifted weights | Overlap and estimation uncertainty |

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

Historical OOS and Hybrid experiments rebuild optimization from observations no
later than `optimization_as_of`; Live Forward freezes its snapshot at creation.
Launch occurs on the declared next complete observation, launch NAV equals
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
