# VaR and CVaR Output Contract

This document defines the public sign, unit, and conversion contract for every
VaR and CVaR value reported by the package, application, exports, backtests,
scenario engine, and optimizer.

## Canonical representation

VaR and CVaR are decimal values in the input return convention and in
**signed loss space**:

| Value | Meaning |
|---:|---|
| `0.04` | 4% loss |
| `0.00` | break-even |
| `-0.02` | 2% gain at the measured tail threshold |

Negative VaR or CVaR is therefore valid. It can occur when even the selected
left tail of the return distribution remains profitable. Public calculations
must not apply `abs()`, clamp the result to zero, or silently relabel a negative
loss value as a positive loss.

## Return and loss relationship

For a portfolio return random variable `R`, confidence level `c`, and
`alpha = 1 - c`:

```text
return_threshold = quantile(R, alpha)
VaR_loss = -return_threshold
```

The displayed return threshold is therefore always `-VaR_loss`. Historical
CVaR similarly negates the mean return in the empirical tail at or below the
return threshold. Gaussian and Cornish-Fisher methods follow the same sign
conversion after estimating their return-space tail value.

For matching methods, samples, confidence levels, and horizons, CVaR is
expected to satisfy `CVaR >= VaR` in loss space. This ordering still applies
when both values are negative.

## Units

- Core functions return decimal values, not percentages.
- Presentation code may multiply the decimal by `100` and append `%`.
- Linear monetary equivalents use `loss_value * portfolio_value`.
- Monetary conversion preserves the sign and uses the same currency as the
  supplied portfolio value.
- Portfolio value must be finite and non-negative.

The monetary multiplication is exact when the metric is a simple-return loss
fraction. For a metric estimated from log returns, it is only a first-order
linearized equivalent. In particular, transforming an average log-tail loss is
not the same as averaging scenario-level monetary losses; reports must label
the log-return result as linearized rather than exact tail P&L.

For example, a signed loss value of `-0.02` on a portfolio value of `100,000`
produces `-2,000`, meaning a 2,000-unit gain at the measured tail threshold.

## Horizon

Core estimators measure the observation horizon represented by their input
series; they are not inherently daily. A daily input produces a daily metric,
while an already aggregated seven-day input produces a seven-day metric.

The separate square-root-of-time helper preserves the sign while scaling the
magnitude. It is an i.i.d. approximation and is not equivalent to estimating
risk directly from realized or simulated multi-period returns.

## Backtesting

Backtesting compares realized return with the return-space threshold:

```text
breach = realized_return < -var_forecast
```

This remains correct when `var_forecast` is negative. In that case the model
forecast a positive return threshold, and falling below that forecast counts as
a breach even if the realized return itself is still positive.

## Optimization boundary

Optimization losses are defined as `-scenario_return`, so optimized VaR and
CVaR use the same signed loss-space contract. The current user-facing CVaR-cap
input is intentionally restricted to a positive loss budget; that input policy
does not change the signed representation of reported risk metrics.

## Public conversion helpers

The authoritative conversions are implemented in
`src/var_cvar_crypto_risk/risk_conventions.py`:

- `return_threshold_to_loss_value`
- `loss_value_to_return_threshold`
- `loss_value_to_money`

New risk models and presentation paths must use these helpers instead of
introducing independent sign or money-conversion rules.
