# Manual Review Findings

This backlog records issues identified while reviewing the application manually.
Entries are observations and proposed acceptance criteria; they are not implemented
changes. Financial methodology must not be changed without a separate specification,
tests, documentation review, and numerical-regression review.

## MR-001 — Separate realized CAGR from annualized mean-return estimates

- **Status:** Planned for the next methodology release
- **Priority:** P1
- **Observed location:** Risk Lab → Risk summary → `Annualized Return`
- **Current behavior:** The displayed value is calculated as
  `(1 + arithmetic mean daily return) ** 365 - 1`.
- **Why this is misleading:** This compounds the same arithmetic sample mean for a
  hypothetical 365-day year. It is not the realized annual growth rate of the
  observed wealth path. The difference can be material for volatile returns because
  the arithmetic mean does not capture volatility drag.
- **Manual-review example:** The reviewed portfolio displayed a mean daily return of
  approximately `0.1597%`, cumulative return of `688.91%`, and 2,068 daily return
  observations. The current calculation produces approximately `79.07%`. The
  realized wealth multiple is `7.8891`, which implies an approximate CAGR of `44.0%`
  over `2,068 / 365` years.
- **Required methodology decision:**
  - report realized CAGR using terminal wealth and actual elapsed time; and
  - if an annualized arithmetic-mean estimate is retained, label it separately and
    do not present it as realized performance. A standard annualized arithmetic mean
    would be `mean_daily_return * periods_per_year`, not a realized CAGR.
- **Target CAGR formula:**
  `CAGR = (ending_value / starting_value) ** (1 / elapsed_years) - 1`, where elapsed
  years are derived from the actual start and end dates under a documented day-count
  convention.
- **Acceptance criteria:**
  1. The UI and exported table distinguish realized CAGR from any expected or
     arithmetic annualized-return estimate.
  2. Metric labels, help text, formulas, README, and methodology documentation agree.
  3. CAGR uses the observed wealth path endpoints and actual elapsed dates.
  4. Missing dates do not silently shorten the elapsed-time denominator.
  5. Unit tests cover constant returns, volatile paths with identical terminal
     wealth, leap-year/date handling, and insufficient samples.
  6. Existing publication baselines change only through an explicit methodology
     review.

## MR-002 — Replace the displayed return-to-volatility ratio with a standard Sharpe ratio

- **Status:** Planned for the next methodology release
- **Priority:** P1
- **Observed location:** Risk Lab → Risk summary → `Sharpe Ratio (annualized)`
- **Current behavior:** The displayed ratio divides the compounded mean-daily
  annualized value from MR-001 by annualized volatility and does not subtract an
  explicit risk-free return. In the reviewed example, `79.0697% / 65.9375%` produces
  approximately `1.1992`.
- **Why this is misleading:** A standard ex-post Sharpe ratio uses the arithmetic mean
  of periodic excess returns divided by their periodic standard deviation, with
  consistent annualization. CAGR should not replace the arithmetic excess-return
  numerator.
- **Target formula:**
  `Sharpe = mean(r_daily - rf_daily) / std(r_daily - rf_daily) * sqrt(365)` for daily
  crypto observations, with the risk-free convention stated explicitly. If a
  constant annual effective rate is selected, its daily equivalent is
  `(1 + rf_annual) ** (1 / 365) - 1`.
- **Manual-review example:** Using the rounded displayed inputs, the standard ratio is
  approximately `0.884` with a zero risk-free rate and approximately `0.810` with a
  5% effective annual risk-free rate, rather than `1.1992`.
- **Acceptance criteria:**
  1. The risk-free assumption, annualization basis, return frequency, and missing-data
     policy are visible beside the result and included in exports.
  2. Risk Lab and optimizer use a shared, explicitly versioned risk-free-rate
     convention where their purposes overlap.
  3. The ratio is calculated from periodic excess returns; it does not use CAGR or
     compounded arithmetic-mean return in the numerator.
  4. Tests cover zero and nonzero risk-free rates, zero volatility, missing values,
     crypto 365-day annualization, and deterministic examples.
  5. The UI distinguishes an ex-post sample Sharpe ratio from any forecast Sharpe used
     by optimization.
  6. Documentation warns that a historical Sharpe ratio is sample-dependent and is
     not evidence of future risk-adjusted performance.
