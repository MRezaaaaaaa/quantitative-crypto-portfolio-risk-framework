# Portfolio Monitoring

## Purpose

The Portfolio Experiment Monitor turns one validated optimizer result into a
persistent research experiment. It preserves the construction recipe, target
allocation, launch convention, daily valuations, risk forecasts, data-quality
records, and update history. It is not an order-management, advisory, or
rebalancing system.

Every experiment has a required human-readable name and an authoritative UUID.
Names may repeat; the UUID is the durable identity used by the database,
downloads, and dashboard.

## Frozen construction snapshot

Creation rebuilds the optimizer from the declared point-in-time data and recipe.
It does not reuse an optimizer result held in Streamlit session state. Activation
requires an accepted solver status and passing independent residual validation.
The activated snapshot then freezes:

- universe, target weights, cash policy, capital, and base currency;
- expected-return, covariance, scenario, optimizer, and risk settings;
- information-set dates, package/code versions, and source/recipe hashes;
- solver status, numerical validation, launch forecast, prices, values, and
  quantities.

The framework supports long-only fixed holdings in Phase 8. For a non-cash asset
`i`, its launch quantity is conceptually:

```text
quantity_i = initial_capital * target_weight_i / launch_price_i
```

After launch, quantities do not change. Current weights drift as prices change:

```text
current_weight_i,t = market_value_i,t / portfolio_NAV_t
```

Total drift is `0.5 * sum(abs(current_weight - target_weight))`. Cash is an
explicit allocation and follows either the configured zero-return or annual-rate
policy. No trade, turnover, or rebalance is inferred from drift.

## Daily records

For each eligible date, the monitor can persist:

- portfolio and benchmark NAV, Base-100 NAV, daily and cumulative return;
- current allocation, target/current difference, and total drift;
- running peak, drawdown, and maximum drawdown;
- origin-safe VaR/CVaR forecasts and matured realized loss;
- source, cutoff, completeness, missing-asset, run, and event metadata.

Monitoring never silently forward-fills a missing price. An incomplete date is
identified explicitly and a fabricated NAV is not finalized. A return after a
gap retains its actual calendar interval.

Realized volatility is an expanding sample standard deviation of eligible
post-launch one-calendar-day Simple returns, annualized with `sqrt(365)`. The
launch zero and multi-day post-gap returns are excluded. At least two eligible
returns are required. It is descriptive realized volatility, not a forecast.

## Risk forecast interpretation

Each risk record preserves origin date, target date, horizon, estimation window,
confidence, methods, portfolio definition, input maximum date, and model version.
The default forecast uses current drifted weights at the origin. Inputs end at
the origin, and an outcome is evaluated only after its target matures.

A VaR exception occurs only when realized horizon loss is greater than the VaR
threshold. CVaR / Expected Shortfall describes modeled tail severity; it is not
an exception threshold. A small exception count is not enough to establish
calibration, independence, or model validity.

## Dashboard views and chart semantics

The Streamlit workspace provides:

| View | Presentation | Interpretation |
|---|---|---|
| NAV and benchmark | Plotly line chart in currency or Base-100 units | Persisted fixed-holdings wealth path |
| Asset allocation | Plotly 100% stacked area chart | Current weights through time; colors are stable by symbol |
| Target versus current | Grouped horizontal bars | Latest immutable target and drifted current weights |
| Weight drift | Asset/date heatmap plus total-drift line | Percentage-point asset drift and portfolio-level drift |
| Drawdown | Underwater area chart | Decline from the persisted running NAV peak |
| VaR/CVaR history | Multi-line loss-space chart | Forecast VaR, forecast CVaR, and matured realized loss |
| Breach timeline | Lines and exception markers | Realized loss versus VaR; CVaR is not a breach rule |
| Forecast versus realized | Grouped bars | Matured point forecasts only; no synthetic fan chart |
| Experiment comparison | Base-100 lines and risk/return scatter | Metrics computed on one explicit shared sample |

Hybrid charts mark the boundary between historical replay and live append.
Comparison requires one of two explicit policies:

- **Common calendar intersection:** experiments are compared only on shared
  dates. Annualized realized volatility uses consecutive one-day intervals.
- **Days since launch:** experiments are compared at shared calendar ages after
  launch, which are not necessarily the same market dates.

All comparison paths are rebased to 100 at the shared start, and summary metrics
use that same intersection. This reduces inconsistent-sample comparisons but
does not remove selection bias or make unlike recipes economically comparable.

## Lifecycle, archive, and downloads

Experiments move through validated states such as draft, backfilling, active,
completed, failed, and archived. Archive is an irreversible action in the
current UI, but it is not deletion: snapshot, valuation, forecast, run, and event
history remain queryable.

The dashboard offers CSV tables and a JSON methodology manifest. The package
also exposes a fuller CSV/JSON experiment bundle with hashes. Both are
**private by default** because they may contain holdings, quantities, prices,
and realized performance. Review every file before deliberate publication.

## Limitations

- No rebalancing, transaction ingestion, orders, or broker integration.
- No fees, slippage, liquidity, market impact, taxes, custody, or execution
  model.
- No intraday valuation, internal scheduler, alerting, or user authentication.
- Fixed holdings are a research convention, not proof that they were tradable at
  the displayed close.
- A current asset universe is not point-in-time and may embed selection or
  survivorship bias.
- Realized performance and risk exceptions do not guarantee future performance
  or validate investment suitability.

See [Forward testing](forward-testing.md),
[Monitoring database and operations](monitoring-database.md), and
[Model risk](model-risk.md).
