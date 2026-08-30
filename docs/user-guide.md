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
Experiment creation rebuilds the optimizer from its own declared information
set and serialized recipe.

## 3. Create an experiment

In **Portfolio Monitor → Create Forward Test**:

1. Assign a descriptive name. The generated UUID remains authoritative.
2. Choose Historical OOS, Live Forward, or Hybrid and read the exact label.
3. Select CoinGecko, yfinance, or a wide daily-price CSV.
4. Provide and review the frozen symbol mapping and optional benchmark.
5. Set training, optimization-as-of, launch, and evaluation/live boundaries.
6. Set capital, risk horizon, confidence, estimation window, expected-return,
   covariance, scenario, objective, cash, and maximum-weight assumptions.
7. Read the methodology preview, then validate, rebuild, and create.

CSV input must contain exactly one `Date` column and one column per required
asset. It is permitted only for Historical OOS because a static upload is not a
refreshable future feed. Live and Hybrid modes require a recorded provider
mapping. Provider fallback cannot silently define the construction snapshot.

Begin with synthetic or publication-safe data. Creation can fail when dates are
reversed, the launch observation is incomplete, the training sample is
insufficient, data are invalid, the optimizer fails, or residual validation does
not pass. Do not weaken these gates merely to obtain weights.

## 4. Inspect the monitor

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

## 5. Update active experiments

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

## 6. Compare experiments

Select at least two experiments and choose the alignment before interpreting the
chart:

- **Common calendar intersection** compares identical dates.
- **Days since launch** compares identical calendar age, not identical market
  conditions.

Paths are rebased to 100 at the shared start. Return, drawdown, volatility, and
exception summaries use the same intersection. A visually better path is not
proof that its optimizer, assumptions, or asset universe are superior; repeated
experiment selection can overfit the comparison.

## 7. Export responsibly

Downloads may contain portfolio holdings, quantities, prices, and realized
performance. Store them privately unless their inputs and publication rights
have been reviewed. Do not commit the local monitoring database, vendor cache,
or generated exports.

For a public article, use the separately documented deterministic synthetic
[publication workflow](../publication/README.md), or build a reviewed equivalent
with legally usable pinned data. A live dashboard screenshot is not exactly
reproducible unless its cutoff, source, recipe, code version, and underlying data
are preserved.

## 8. Interpretation boundary

The monitor uses fixed quantities and does not recommend rebalancing. It omits
fees, slippage, liquidity, market impact, taxes, custody, and execution. A
historical replay is not live; a live test is still one noisy sample; and neither
guarantees performance, suitability, or model validity.

Read [Forward testing](forward-testing.md),
[Portfolio monitoring](portfolio-monitoring.md),
[Monitoring database and operations](monitoring-database.md), and
[Model risk](model-risk.md) before publishing results.
