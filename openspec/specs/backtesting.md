# Backtesting & Model Validation — OpenSpec

> **Status: Implemented**

## 1. Purpose

Provide statistical and visual diagnostics for evaluating VaR forecasts
against realized portfolio returns. Kupiec and Christoffersen tests examine
coverage and breach dependence; they do not by themselves establish complete
model validity or regulatory compliance.

## 2. Scope

### In scope
- Rolling one-step-ahead VaR forecasts with strict no-look-ahead bias.
- Breach detection and breach statistics (counts, rates, ratios).
- Kupiec Proportion of Failures test (unconditional coverage).
- Christoffersen Independence test (serial correlation of breaches).
- Christoffersen Conditional Coverage test (combined POF + Independence).
- Two-tier traffic light system: Basel III absolute counts (250-day standard
  window) and a generalised ratio-based variant for non-standard windows.
- Multi-model comparison across Historical, Gaussian, and Cornish-Fisher VaR.
- Backtesting CSV exports and PNG charts.
- A new Streamlit tab ("Backtesting & Model Validation") wrapping the
  pipeline behind interactive controls.
- Unit tests for every public function and edge case.

### Out of scope
- Monte Carlo simulation as a VaR forecast method.
- CVaR backtesting / Expected Shortfall validation.
- GARCH-based conditional coverage tests.
- Filtered Historical Simulation, DCC-GARCH, copula-based dependence
  modelling.
- Stress testing or risk-contribution decomposition.
- Optimisation under backtested constraints.

## 3. Functional Requirements

1. `rolling_var_forecast(returns, method, confidence_level, window)` SHALL
   produce a `pd.DataFrame` with columns `actual_return`, `var_forecast`,
   `breach` indexed on the forecast dates.
2. The forecast at date `t` MUST be computed from `returns[t-window:t]`
   strictly excluding `t` itself. Look-ahead bias is forbidden.
3. The output length MUST equal `len(returns) - window`.
4. The supported methods are exactly `historical`, `gaussian`,
   `cornish_fisher` — all delegated to the existing `var_models.calculate_var`.
5. `calculate_var_breaches(backtest_df, confidence_level)` SHALL return a
   dict with `observations`, `actual_breaches`, `expected_breaches`,
   `expected_breach_rate`, `actual_breach_rate`, `breach_ratio`, and
   `confidence_level`.
6. `kupiec_pof_test(breaches, confidence_level)` SHALL implement the
   likelihood-ratio test with `df=1` and gracefully handle the boundary cases
   `x = 0` and `x = n` without returning NaN.
7. `christoffersen_independence_test(breaches)` SHALL build the 2 × 2
   transition matrix `(n00, n01, n10, n11)` and perform the LR test with
   `df=1`. All-same and length-1 inputs SHALL NOT raise.
8. `christoffersen_cc_test(breaches, confidence_level)` SHALL combine the
   POF and Independence statistics: `LR_cc = LR_pof + LR_ind` evaluated on
   `chi2(df=2)`.
9. `assign_traffic_light_status(...)` SHALL support modes `basel3`,
   `rate_based`, and `auto` — with `auto` selecting `basel3` for
   `240 ≤ n ≤ 260` observations and `rate_based` otherwise.
10. `interpret_traffic_light_status(status)` SHALL return the canonical
    one-sentence description for `Green` / `Yellow` / `Red`, raising for
    unknown statuses.
11. `backtest_var_model(...)` SHALL orchestrate the full pipeline and return
    `(forecast_df, result_dict)` with all 19+ documented keys including
    `traffic_light_mode_used`.
12. `compare_var_models_backtest(...)` SHALL run multiple methods and MUST
    NOT abort the comparison if one method raises — the failed method's row
    SHALL include the error message in an `error` column.
13. `create_backtesting_report_table(comparison_df)` SHALL return the
    canonical 14-column reporting DataFrame.
14. The Streamlit "Backtesting & Model Validation" tab SHALL expose method,
    confidence, window, and test-period controls and SHALL render KPI
    cards, the backtest chart, the breach timeline, and (in compare-all
    mode) the model-comparison chart.
15. The tab SHALL provide download buttons for the forecast CSV, the
    backtest results JSON, the model-comparison CSV, and chart PNGs.

## 4. Non-Functional Requirements

- **Performance.** `rolling_var_forecast` for a 1 000-observation series
  with a 252-day window SHALL complete in under five seconds on a
  standard developer laptop.
- **Maintainability.** Each statistical test SHALL be isolated in its own
  function in `backtesting.py`. No plotting, Streamlit, or data-loading
  code is permitted in `backtesting.py`.
- **Testability.** All core functions SHALL be pure — no global state, no
  side effects, no network access — so they can be unit-tested with
  synthetic data only.
- **Determinism.** Tests SHALL use seeded RNGs and synthetic series; no
  live API calls.

## 5. Inputs

| Input                  | Type             | Notes                                          |
| ---                    | ---              | ---                                            |
| `portfolio_returns`    | `pd.Series`      | DatetimeIndex, no NaN, decimal returns         |
| `method`               | `str`            | `historical` \| `gaussian` \| `cornish_fisher` |
| `confidence_level`     | `float`          | strictly in (0, 1)                             |
| `window`               | `int`            | ≥ 30, < `len(returns)`                         |
| `traffic_light_mode`   | `str`            | `auto` \| `basel3` \| `rate_based`             |

## 6. Outputs

| File                                                    | Format | Schema / Notes                                                                |
| ---                                                     | ---    | ---                                                                          |
| `outputs/tables/var_forecasts_<method>.csv`             | CSV    | columns: `date`, `actual_return`, `var_forecast`, `breach`                    |
| `outputs/tables/backtesting_results.csv`                | CSV    | one row, all `result_dict` keys                                               |
| `outputs/tables/model_comparison.csv`                   | CSV    | one row per method, includes `error`                                          |
| `outputs/charts/var_backtesting_exceptions_<method>.png`| PNG    | from `plot_var_backtest`                                                      |
| `outputs/charts/breach_timeline_<method>.png`           | PNG    | from `plot_breach_timeline`                                                   |
| `outputs/charts/model_comparison_backtest.png`          | PNG    | from `plot_model_comparison_backtest`                                         |

## 7. Statistical Tests

### 7.1 Kupiec POF Test (Unconditional Coverage)

```
LR_pof = -2 · ln( [(1-p)^(n-x) · p^x] / [(1-x/n)^(n-x) · (x/n)^x] )
p_value = 1 - chi2.cdf(LR_pof, df=1)
```

with `p = 1 - confidence_level`, `n = observations`, `x = breaches`.
Boundary cases `x = 0` and `x = n` are handled via the convention
`0 · log(0) = 0`, so the statistic is always finite.

**Interpretation.** `pass = (p_value >= 0.05)`. A pass means the breach
*frequency* is consistent with the model; a fail means the model is
*either* too conservative or too aggressive.

### 7.2 Christoffersen Independence Test

Build the 2 × 2 first-order transition matrix:

| prev \ curr | 0   | 1   |
| ---         | --- | --- |
| 0           | n00 | n01 |
| 1           | n10 | n11 |

Then:
```
pi01 = n01 / (n00 + n01)
pi11 = n11 / (n10 + n11)
pi   = (n01 + n11) / (n00 + n01 + n10 + n11)

ll_r = (n00 + n10)·ln(1-pi) + (n01 + n11)·ln(pi)
ll_u = n00·ln(1-pi01) + n01·ln(pi01) + n10·ln(1-pi11) + n11·ln(pi11)

LR_ind  = -2 · (ll_r - ll_u)
p_value = 1 - chi2.cdf(LR_ind, df=1)
```

**Interpretation.** A pass means breaches are not serially correlated in
the first-order Markov sense (`pi01 ≈ pi11`); a fail signals breach
clustering and is the classic symptom of a model that ignores volatility
regimes.

### 7.3 Christoffersen Conditional Coverage Test

```
LR_cc   = LR_pof + LR_ind
p_value = 1 - chi2.cdf(LR_cc, df=2)
```

This is the combined joint test — `df = 2` because two restrictions are
being tested simultaneously. Frequently the most informative single
diagnostic in published academic backtests.

## 8. Traffic Light System

### 8.1 Basel III mode (BCBS 2019 — standard 250-day window)

| Zone     | Breach count       | Action                            |
| ---      | ---                | ---                               |
| 🟢 Green  | 0 – 4              | No supervisory action             |
| 🟡 Yellow | 5 – 9              | Capital multiplier escalation     |
| 🔴 Red    | ≥ 10               | Model presumptively inadequate    |

### 8.2 Rate-based mode

`breach_ratio = actual_breach_rate / expected_breach_rate`

| Zone     | Threshold               |
| ---      | ---                     |
| 🟢 Green  | 0.75 ≤ ratio ≤ 1.25     |
| 🟡 Yellow | 0.50 ≤ ratio ≤ 1.75 (and not Green) |
| 🔴 Red    | otherwise               |

### 8.3 Auto-selection logic

```
if 240 <= n_observations <= 260:
    use Basel III thresholds
else:
    use rate-based thresholds
```

The Basel III thresholds are calibrated for ~250 trading days. Applying
them outside that range can be misleading (e.g. 10 breaches in 5 000
observations is *less* than expected at 95%, not "Red"). The rate-based
mode normalises by the expected count and remains meaningful at any window
length.

## 9. Edge Cases

| Case                                | Behaviour                                                                                |
| ---                                 | ---                                                                                      |
| Zero breaches                       | All tests return finite stats; Kupiec pass depends on `n` and `p`                        |
| All breaches                        | All tests return finite stats; Independence test is degenerate-but-defined               |
| `window > len(returns)`             | `rolling_var_forecast` raises `ValueError`                                               |
| `window < 30`                       | `rolling_var_forecast` raises `ValueError`                                               |
| Length-1 breach series              | Independence and CC return `pass_test = None` and `p_value = NaN`; no exception          |
| One method fails inside `compare_*` | That method's row carries `error = str(exc)`; other methods proceed                      |
| Empty `comparison_df`               | `create_backtesting_report_table` returns an empty DataFrame with the canonical columns  |

## 10. Acceptance Criteria

1. `rolling_var_forecast` works for `historical`, `gaussian`, and
   `cornish_fisher`.
2. No look-ahead bias: forecast at `t` uses only `returns[t-window:t]`.
3. Breach detection equals `actual_return < -var_forecast`.
4. Expected vs actual breach statistics are computed correctly.
5. Kupiec POF test (`df = 1`) is implemented and numerically robust at
   `x = 0` and `x = n`.
6. Christoffersen Independence test (`df = 1`) is implemented with explicit
   transition counts.
7. Christoffersen CC test (`df = 2`) equals the sum of POF and Independence
   LR statistics.
8. Two-tier traffic light system supports `basel3`, `rate_based`, and
   `auto`.
9. Three new backtesting plot functions are added to `plotting.py` without
   modifying the existing three.
10. Output CSVs follow the naming convention
    `var_forecasts_<method>.csv` / `model_comparison.csv`.
11. The Streamlit Backtesting tab works end-to-end without modifying any
    existing tab logic.
12. The pytest suite (50 backtesting tests) all pass; the prior 42 tests
    still pass.
13. Public methodology documentation explains backtesting scope and limits.
14. This OpenSpec document is fully written.
15. The backtesting module does not silently claim support for CVaR
    backtesting, GARCH conditional tests, or regulatory validation.

## 11. Future Extension Points

- **Monte Carlo validation:** generate synthetic breach paths to assess the
  sampling distribution of breach counts.
- **CVaR / ES backtesting:** evaluate appropriate Expected Shortfall tests
  through a separately reviewed methodology change.
- **Conditional coverage:** consider dynamic quantile tests and conditional
  volatility models without presenting them as regulatory approval.

## Current Enhancements

- **Backtesting mode.** `rolling_var_forecast`, `backtest_var_model`, and
  `compare_var_models_backtest` accept `backtest_mode ∈ {"overlapping",
  "non_overlapping"}`. `overlapping` (default, step 1) preserves the original
  behaviour; `non_overlapping` steps by `horizon_days` so realised h-day returns
  are disjoint (the correct basis for the Christoffersen independence tests). The
  no-look-ahead invariant is unchanged; outputs carry `backtest_mode` /
  `step_size`, and the report table gains a `Mode` column. For `horizon_days == 1`
  the two modes are identical.
- **Breach-rate evolution.** `calculate_rolling_breach_rate(backtest_df, window)`
  → rolling mean of the breach indicator.
- **Worst losses.** `get_worst_realized_losses(backtest_df, n)` → the n most
  negative realised returns (Date / Actual Return / VaR Forecast / Breach / Loss).
- **Sub-period summary.** `summarize_backtest_by_period(backtest_df,
  confidence_level, freq)` → per-period observations / actual vs expected breaches.
