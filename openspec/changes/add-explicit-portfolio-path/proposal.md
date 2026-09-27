# Add explicit Risk Lab portfolio paths

## Why

Risk Lab historically calculated `asset_returns @ constant_weights` and
presented the result as a generic portfolio return. That arithmetic embeds a
frictionless daily rebalance and is not a Buy & Hold history. Portfolio
evolution must be explicit, selectable and auditable independently from risk
horizon.

## What changes

- Add a pure versioned engine for fixed-quantity Buy & Hold and explicit UTC
  daily, weekly, monthly and quarterly rebalancing.
- Model user-entered proportional commission and slippage on gross traded
  notional, with internally consistent post-cost target weights.
- Preserve incomplete closes, defer scheduled trades and record scheduled and
  effective dates.
- Route Risk Lab realized analytics, backtesting, current exposure and exports
  through the selected path.
- Use latest net NAV for monetary measures labeled as current risk.
- Preserve Portfolio Monitor as manual fixed holdings with no rebalancing.

## Model risk

Close execution is idealized. The cost model excludes liquidity, market impact,
partial fills, taxes, funding, borrow and custody costs. Selecting a rebalance
frequency after reviewing the same history is model-selection overfitting, not
evidence of an optimal rule.

## Authority

The user explicitly requested this Risk Lab methodology correction. No commit,
push, tag, release or deployment is part of the change.
