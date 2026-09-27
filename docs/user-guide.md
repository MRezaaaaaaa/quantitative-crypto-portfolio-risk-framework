# User Guide

## 1. Install the reviewed environment

Requirements are Python 3.10 through 3.13 and uv 0.11.16 for the exact locked
environment.

```bash
uv sync --locked --extra app --extra dev
uv run --locked --no-sync alembic upgrade head
uv run --locked --no-sync streamlit run app.py
```

The first command installs the application and developer extras from `uv.lock`.
The second initializes or upgrades the private local monitoring database. The
third opens the Streamlit application.

## 2. Choose a workspace

- **Risk Lab** is the existing interactive VaR/CVaR, backtesting, simulation,
  assumptions, and optimization workspace.
- **Portfolio Monitor** is the persistent experiment registry and forward-
  testing workspace.

Risk Lab session results are not silently converted into monitored portfolios.
New monitoring experiments use manually entered assets and weights only. No
optimizer, expected-return fitting or scenario generation runs during creation.
The separate Risk Lab optimizer is unchanged. Existing optimized experiments
remain readable and updatable; their old snapshots are not rewritten.

## 3. Select the Risk Lab portfolio path

Under **Returns & portfolio**, choose one explicit policy:

- **Buy & Hold — fixed quantities:** entered weights are initial launch weights;
- **Periodic rebalance to target weights:** choose weekly, monthly or quarterly;
- **Daily rebalanced — legacy constant-weight model:** preserves the prior
  zero-cost constant-weight interpretation when costs are zero.

For rebalanced paths, enter non-negative commission and slippage in basis
points. The first allocation has no setup cost. Weekly, monthly and quarterly
events are scheduled on theoretical UTC calendar boundaries, independent of
the observed price index. A missing boundary or incomplete scheduled close is
never moved backward: it is deferred to the first later complete close. If
several boundaries are pending, all remain in the audit but only one target-reset
trade and one cost are applied. The VaR time horizon does not alter the rebalance
schedule.

After running Risk Lab, review **Cumulative Return** for Hold-versus-policy NAV,
weights/drift/event markers, turnover/costs and gross-versus-net NAV. Review
**Drawdown** for the policy comparison and **Data** for portfolio-, asset- and
JSON-provenance exports. Current Money VaR/CVaR use the displayed latest net NAV
and date. Stateful policies reject negative/leverage weights because financing,
margin and borrow mechanics are outside V1.

Start with **Risk Summary**. Its Historical Analysis Context, headline cards and
CSV all describe the finalized selected-policy path after recorded proportional
costs. Annualized return, annualized volatility and Sharpe are intentionally not
shown there. Historical Tail Distribution is not a loss forecast; a money value
is only the historical percentage statistic scaled by ending net NAV. Review the
data-quality audit before interpreting any row labeled daily. A source row on
the current UTC day is retained in the audit but excluded from finalized NAV,
drawdown, tail metrics, and sample counts. Calendar gaps, incomplete prices and
duplicate/order defects remain visible and can make daily moments unavailable.

The Risk Summary CSV is a long table with `Record Type`, `Section`, `Name`,
`Value`, `Display Value`, `Unit`, and `Sample Size`. Use `Value` for analysis and
`Display Value` for presentation. Context, policy, weights, costs, data-quality
details, excluded provisional dates, methodology, and every visible metric come
from the same summary object shown in the page.

Risk Lab figures are interactive Plotly charts. Use the chart modebar to export
PNG, or the adjacent download control for a self-contained interactive HTML
file. Hover labels carry exact dates and units; chart calculations still come
from the underlying finance modules rather than the browser renderer.

In **Robust Assumptions**, the default **All Estimators** view displays Raw
Historical Mean, Median, Trimmed Mean, Winsorized Mean, Shrinkage Estimate, each
available Manual View, and the exact **Final E[r]** received downstream. The
horizontal range shows estimator sensitivity, while small vertical offsets keep
equal x-values inspectable without changing their values. Switch to **Pairwise
Comparison** to retain the Raw Mean-versus-one-estimator audit. Portfolio order
is preserved unless **Largest estimator dispersion** is selected; dispersion is
the full available estimator range, not only Raw Mean-to-Final distance. Missing
Manual Views remain `N/A`, and the title follows the actual assumption horizon.
Neither a narrow nor a wide range validates a future-return estimate.

## 4. Create an experiment

In **Portfolio Monitor → Create Forward Test**:

1. Assign a descriptive name. The generated UUID remains authoritative.
2. Choose Historical OOS, Live Forward, or Hybrid and read the exact label.
3. Select CoinGecko, yfinance, or a wide daily-price CSV.
4. Provide and review the frozen symbol mapping and optional benchmark.
5. Enter each held asset and its initial **Weight (%)** in the Manual portfolio
   table. Use strictly positive weights totaling 100%; there is no automatic
   normalization, shorting or leverage. Add a `CASH` row only for explicit cash.
6. Set risk-history start, allocation decision/risk-history cutoff, launch and
   evaluation/live boundaries, capital, horizon, confidence and estimation window.
7. Read the preview and choose **Validate and create manual portfolio**. The
   selected launch date must have complete prices and be a completed UTC day.
   Quantity is `capital * entered_weight / launch_price`; it stays fixed afterward.

The decision date precedes launch and does not need to be immediately adjacent
to it. Historical weights entered today may contain hindsight bias: declaring
an old decision date does not prove those weights were known then. Prospective
evidence requires observations arriving after the actual snapshot was frozen.

### Offline manual acceptance example

Upload `tests/fixtures/synthetic_daily_prices.csv` (synthetic, not market evidence).
Use `{"BTC":"BTC","ETH":"ETH","SOL":"SOL"}` as the mapping and enter
BTC 50%, ETH 30%, SOL 20%. Set history start `2024-01-01`, decision/cutoff
`2024-03-01`, launch `2024-03-02`, historical end `2024-04-05`, capital 100000,
horizon 1 and estimation window 30. After creation, check initial NAV 100000,
allocation totaling 100%, constant quantities, drift, risk history and persistence
after refreshing the browser. A second manual weight mix can test Comparison.

CSV input must contain exactly one `Date` column and one column per required
asset. It is permitted only for Historical OOS because a static upload is not a
refreshable future feed. Live and Hybrid modes require a recorded provider
mapping. Provider fallback cannot silently define the construction snapshot.

Begin with synthetic or publication-safe data. Creation can fail when dates are
reversed, launch prices are incomplete, weights are invalid or data are invalid.
Insufficient risk history is shown as an unavailable/insufficient-window forecast,
not a reason to fabricate a risk estimate or optimize different weights.

## 5. Inspect the monitor

The **Experiments** view lists name, UUID, mode, status, date boundaries, latest
state, and quality. Archive retains all history and is irreversible in the
current UI.

The **Portfolio Monitor** view contains:

- Overview: NAV/benchmark and drawdown;
- Allocation & Drift: 100% stacked current allocation, latest target/current
  weights, and drift;
- Risk & Breaches: VaR/CVaR forecasts, matured losses, and VaR exceptions;
- Forecast vs Realized: matured point forecasts without a fabricated fan;
- Snapshot & Provenance: fixed allocations, hashes, methods, dates, and events;
- private-by-default table and manifest downloads.

An empty or pending chart is not automatically an error. New live experiments
need future complete observations, and a horizon forecast cannot be evaluated
before its target matures. Realized volatility needs at least two eligible
one-day post-launch returns.

## 6. Update active experiments

**Data Quality → Update Now** performs one bounded refresh for an active Live or
Hybrid experiment. Review actual source, cutoff, inserted/skipped counts,
warnings, incomplete dates, and missing assets.

For unattended cadence, configure the one-shot command in an external scheduler:

```bash
uv run --locked --no-sync qcprf-monitor --all-active
```

The repository does not install or manage cron/launchd for you. Test the command
manually, capture its JSON exit result in private operational logs, and avoid
placing database credentials in the command line.

## 7. Compare experiments

Select at least two experiments and choose the alignment before interpreting the
chart:

- **Common calendar intersection** compares identical dates.
- **Days since launch** compares identical calendar age, not identical market
  conditions.

Paths are rebased to 100 at the shared start. Return, drawdown, volatility, and
exception summaries use the same intersection. A visually better path is not
proof that its allocation or asset universe is superior; repeated
experiment selection can overfit the comparison.

## 8. Export responsibly

Downloads may contain portfolio holdings, quantities, prices, and realized
performance. Store them privately unless their inputs and publication rights
have been reviewed. Do not commit the local monitoring database, vendor cache,
or generated exports.

For a public article, use the separately documented deterministic synthetic
[publication workflow](../publication/README.md), or build a reviewed equivalent
with legally usable pinned data. A live dashboard screenshot is not exactly
reproducible unless its cutoff, source, recipe, code version, and underlying data
are preserved.

## 9. Interpretation boundary

The monitor uses fixed quantities and does not recommend rebalancing. It omits
fees, slippage, liquidity, market impact, taxes, custody, and execution. A
historical replay is not live; a live test is still one noisy sample; and neither
guarantees performance, suitability, or model validity.

Read [Forward testing](forward-testing.md),
[Portfolio monitoring](portfolio-monitoring.md),
[Monitoring database and operations](monitoring-database.md), and
[Model risk](model-risk.md) before publishing results.
