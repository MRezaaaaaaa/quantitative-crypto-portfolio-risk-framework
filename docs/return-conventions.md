# Return Conventions and Calculation Boundaries

## Policy

The application does not use one global return switch for unrelated
calculations. It resolves an explicit policy:

| Calculation | Required convention |
|---|---|
| Portfolio construction | Simple |
| NAV, cumulative return, and drawdown | Simple |
| VaR/CVaR monitoring and backtesting | Simple |
| Historical and Monte Carlo scenarios | Simple |
| Portfolio optimization | Simple |
| Distribution diagnostics | Simple by default; Log in Advanced mode |

`Automatic` mode uses simple returns everywhere. `Advanced` mode exposes a
Simple/Log choice only for distribution diagnostics. Selecting Log does not
change portfolio wealth, headline risk, backtesting, Monte Carlo, robust-risk
inputs, or optimization.

## Why portfolio arithmetic uses simple returns

For asset simple returns `r_i` and the actual beginning-of-period weights `w_i`,
the one-period portfolio return before a close-time transaction cost is:

```text
r_p = sum(w_i * r_i)
```

Asset log returns do not aggregate across assets by weighted addition. If
`g_i = log(1 + r_i)`, the exact portfolio log return is:

```text
g_p = log(1 + sum(w_i * (exp(g_i) - 1)))
```

Therefore `sum(w_i * g_i)` is not treated as the portfolio log return. The
package derives the realized simple return from the explicit portfolio path,
then uses `log(1 + r_p)` only for the optional diagnostic portfolio Log series.

Keeping `w_i` constant through every row implies a frictionless rebalance at
every close. That behavior exists only under the labeled legacy daily-
rebalanced policy. Buy & Hold carries quantities, so beginning weights drift;
periodic policies reset them only on audited effective dates. Risk horizon is
an aggregation/scaling choice and does not change those dates.

## Time aggregation

- Simple returns compound as `product(1 + r_t) - 1`.
- Log returns aggregate through time as `sum(g_t)`.
- A simple-return NAV is `V_0 * product(1 + r_t)`.
- A log-return diagnostic wealth equivalent is `V_0 * exp(sum(g_t))`.

Multi-day square-root scaling is a separate approximation for volatility or
risk magnitude. It does not convert Simple returns into Log returns and is not
equivalent to realized multi-day compounding.

## API enforcement

Scenario construction, scenario portfolio aggregation, Monte Carlo wealth
paths, cross-method scenario comparison, and optimization reject a declared Log
input convention. This fails loudly instead of silently mixing arithmetic and
log-return formulas.

The return-policy object is implemented in
`src/var_cvar_crypto_risk/return_conventions.py`. Stateful wealth and realized
portfolio returns are implemented in `portfolio_path.py`; legacy stateless
aggregation remains in `portfolio.py`, and horizon compounding is in
`returns.py`.
