# Phase 3 Design Document — VaR Backtesting & Model Validation

## System Architecture

```
                ┌──────────────────────────────────────────────┐
                │  Streamlit app.py — "Backtesting" tab        │
                │   (controls, KPI cards, charts, downloads)   │
                └───────────────┬──────────────────────────────┘
                                │
                                ▼
                ┌──────────────────────────────────────────────┐
                │  backtesting.py  (pure functions, no I/O)    │
                │                                              │
                │   rolling_var_forecast                       │
                │   calculate_var_breaches                     │
                │   kupiec_pof_test                            │
                │   christoffersen_independence_test           │
                │   christoffersen_cc_test                     │
                │   assign_traffic_light_status                │
                │   interpret_traffic_light_status             │
                │   backtest_var_model                         │
                │   compare_var_models_backtest                │
                │   create_backtesting_report_table            │
                └────────────────┬─────────────────────────────┘
                                 │
                                 ▼
                ┌──────────────────────────────────────────────┐
                │  var_models.py  (Phase 1 — unchanged)        │
                │   calculate_var(returns, method, confidence) │
                └──────────────────────────────────────────────┘

                                 ▲
                                 │
                ┌──────────────────────────────────────────────┐
                │  plotting.py  (3 new chart functions)        │
                │   plot_var_backtest                          │
                │   plot_breach_timeline                       │
                │   plot_model_comparison_backtest             │
                └──────────────────────────────────────────────┘
```

## Module Boundaries

| Module           | Responsibility                                                | Forbidden        |
| ---              | ---                                                           | ---              |
| `backtesting.py` | Pure compute (forecasts, statistics, traffic light, reports)  | matplotlib, streamlit, requests, file I/O |
| `plotting.py`    | Convert backtest DataFrames into matplotlib figures           | calculation logic, statistical tests       |
| `app.py`         | Bind UI controls to compute and render results                | calculation logic, custom chart logic      |
| `var_models.py`  | VaR calculation primitives (Phase 1, untouched)               | —                |

## Data Flow

```
raw price data (Phase 1)
   ↓
asset returns (Phase 1)
   ↓
portfolio_returns: pd.Series (Phase 1 / 2)
   ↓
rolling_var_forecast(returns, method, conf, window)   ← no look-ahead
   ↓
forecast_df: actual_return | var_forecast | breach
   ↓
calculate_var_breaches  ─►  kupiec_pof_test
                       └─►  christoffersen_independence_test
                       └─►  christoffersen_cc_test
   ↓
assign_traffic_light_status  ─►  interpret_traffic_light_status
   ↓
backtest_var_model returns (forecast_df, result_dict)
   ↓
compare_var_models_backtest aggregates across methods → comparison_df
   ↓
plotting.py renders 3 charts                       (visual)
create_backtesting_report_table renders the table  (tabular)
   ↓
app.py displays + offers downloads (CSV / PNG / JSON)
```

## Look-Ahead Bias Prevention

The single most important correctness invariant in this module. We
implement it by:

1. Iterating with an explicit integer index `i`, computing `t = window + i`.
2. Using exactly `returns.iloc[t-window : t]` (Pythonic half-open slice
   excluding `t`).
3. Recording `actual_return = returns.iloc[t]` *after* the forecast is
   already stored.
4. The unit test `test_rolling_var_forecast_no_lookahead` asserts that
   for every forecast date the historical-VaR value computed by the
   function is reproducible from `returns[t-window:t]` alone — the
   realised value at `t` must not appear in that window.

Because every test reduces to an exact numeric comparison against a
window slice that excludes `t`, any future refactor that accidentally
introduces look-ahead bias will fail this test deterministically.

## Statistical Test Design Decisions

### Why `df = 1` for Kupiec POF

The Kupiec test imposes a single restriction: `p̂ = p`. The likelihood
ratio is therefore distributed as `chi²(df=1)` under the null. The
test is asymptotic — for very small samples (n < ~50) the chi² approximation
is loose, but Phase 3 enforces `window ≥ 30` as a practical floor.

### Why `df = 1` for Christoffersen Independence

The Independence test imposes a single restriction on the 2-state Markov
chain: `pi01 = pi11`. Hence `df = 1` again.

### Why `df = 2` for the Conditional Coverage test

CC tests two restrictions simultaneously (the POF restriction *and* the
Independence restriction), so the combined LR statistic is asymptotically
`chi²(df=2)`. This is the standard joint test reported in academic
backtest tables and the most parsimonious single number that says "model
is consistent with reality".

### Numerical robustness — `0 · log(0) = 0`

The standard convention for terms like `0 · log(p)` when `p = 0` (or
equivalently `0 · log(0)`) is to assign the value `0`, since the limit
`x · log(x) → 0` as `x → 0⁺`. We implement this via the `_xlogx` helper
so all four edge cases (`x = 0`, `x = n`, `pi01 = 0`, `pi11 = 0`) yield
finite statistics rather than NaN.

## Two-Tier Traffic Light Design Rationale

The Basel III thresholds (Green ≤ 4, Yellow 5–9, Red ≥ 10) are calibrated
for a 250-trading-day window and a 99% one-day VaR. They are the
*regulatory* benchmark and we want to expose them verbatim. But they
break down off-window:

- 1 000 observations at 95% expects 50 breaches; 10 breaches there is
  spectacularly *low*, not Red.
- 5 000 observations at 95% expects 250 breaches; the basel3 thresholds
  are nonsensical at that scale.

We therefore added the `rate_based` mode (Green: 0.75 ≤ ratio ≤ 1.25,
Yellow: 0.50 ≤ ratio ≤ 1.75, Red: outside). It generalises the basel3
intuition to any window length while preserving the same three-zone
shape. The `auto` mode chooses basel3 when `n ∈ [240, 260]` and
rate-based otherwise — so the user sees the regulatory verdict whenever
their window is close to the regulatory window, and a sensible
generalisation otherwise.

## Dashboard Integration Pattern

`app.py` only contains UI wiring and orchestration. All math is delegated
to `backtesting.py` and all rendering to `plotting.py`. The new tab is
appended; the five Phase 1/2 tabs and the analysis pipeline above them
are not modified. The tab is gated behind the same `st.stop()` guard
that exists for the global "Run risk analysis" button — by the time the
user can click "Run Backtest" we already have a valid `portfolio_returns`
in scope.

## Stabilization Pass (post-implementation)

Some Phase 3 design choices were tightened after the initial
implementation:

- **Reproducible artifacts on disk.** Streamlit download buttons are
  convenient inside the app but not suitable as deliverables for a
  portfolio, a GitHub README, or a LinkedIn writeup. The new
  `run_phase3_backtest_demo.py` CLI script orchestrates the existing
  package modules and writes the full Phase 3 output set (forecast CSVs,
  model comparison table, backtesting results CSV/JSON, three plot
  families) directly under `outputs/`. The script contains no
  calculation logic — it only loads config, calls the package functions,
  and saves files.
- **Lazy data-source imports.** Eagerly importing `yfinance` (and the
  CoinGecko HTTP client) from the package `__init__` made it impossible
  to install / test the core analytics without those optional
  dependencies, and made `import var_cvar_crypto_risk.backtesting`
  pay a network-stack import cost it didn't need. We moved both imports
  inside the call sites of `yfinance_client.fetch_yfinance_prices` and
  `data_loader.load_price_data`, raising a clear `ImportError` with an
  install hint if yfinance is missing at the point of use.
- **Lightweight package surface.** The package `__init__.py` now exposes
  only `__version__` and `__project__`. Users import from submodules
  explicitly (`from var_cvar_crypto_risk.var_models import calculate_var`),
  which keeps the dependency graph honest and means future analytic
  modules can be added without re-touching the package init.
- **Repo hygiene.** `.gitignore` excludes `outputs/`, `data/cache/`,
  `__MACOSX/`, `*.egg-info/`, `.DS_Store`, `__pycache__/`, build / dist
  / coverage / Jupyter checkpoint dirs, the virtualenv. `.gitkeep`
  markers preserve the empty directory structure for `data/{raw,processed,cache}`
  and `outputs/{charts,tables,reports}`.
- **Stabilization tests.** Four new tests in `tests/test_stabilization.py`
  pin the invariants: (a) `import var_cvar_crypto_risk` does not load
  yfinance/coingecko/data_loader, (b) `import var_cvar_crypto_risk.backtesting`
  does not load yfinance, (c) the Phase 3 demo script imports cleanly,
  (d) `export.save_dataframe` / `save_series` / `save_json` /
  `ensure_output_dirs` all create missing parent directories. Plus a
  fifth test that runs `backtest_var_model` with `yfinance` stubbed out
  in `sys.modules`.

## Future Upgrade Path

- **Phase 4 — Monte Carlo:** generate synthetic breach paths to bound the
  sampling distribution of breach counts; supports super-Bonferroni
  multi-period coverage tests.
- **Phase 5 — ES backtesting (FRTB):** add Acerbi-Székely Z2/Z3 ES tests,
  McNeil-Frey ES residual test. Reuses the rolling forecast harness with
  a new "rolling_cvar_forecast" sibling.
- **Phase 6 — GARCH-conditional tests:** Engle-Manganelli dynamic
  conditional quantile (DQ) test, DCC-GARCH for multi-asset breach
  dependency. Plug a `GARCHVaR` model into `var_models.py` and the rest
  of this pipeline reuses unchanged.
