# Phase 5.5 — Design Notes

## Non-overlapping backtesting

`rolling_var_forecast` previously stepped one day at a time
(`for t in range(window, n - h + 1)`). Phase 5.5 generalises the step:

```
step_size = 1 if backtest_mode == "overlapping" else horizon_days
for t in range(window, n - h + 1, step_size):
    ...
```

`overlapping` reproduces the original behaviour exactly; `non_overlapping`
spaces forecast dates one full horizon apart so the realised *h*-day returns are
disjoint. The lookback-window VaR estimation and the strict
`[t-window, t)` / `[t, t+h)` separation are unchanged, so the no-look-ahead
invariant still holds. Output rows carry `backtest_mode` and `step_size`, and the
values propagate through `backtest_var_model`, `compare_var_models_backtest`, and
the report table (`Mode` column). For `horizon_days == 1` the two modes are
identical (`step_size == 1`).

## Horizon-matched distribution

`returns.calculate_horizon_returns(returns, horizon_days, method, overlapping)`
aggregates daily returns into *h*-day returns (`prod(1+r)-1` for simple,
`sum(r)` for log). The Distribution tab feeds the horizon-matched series into
both the histogram and the VaR/CVaR computation, and labels the x-axis / title
accordingly. The headline VaR/CVaR cards keep √t scaling (out of scope to change),
and an inline caption flags the difference. The "show all VaR lines" overlay
computes each method's VaR on the *same* horizon-matched series.

## Correlation tab

`correlation.py` is a new pure-analytics module (matching the one-concern-per-
module layout). `calculate_correlation_matrix` wraps `DataFrame.corr` (pearson /
spearman). `calculate_rolling_average_correlation` slides a window, takes the
full pairwise correlation matrix, and records the mean of its off-diagonal
entries — a single scalar tracking diversification decay. The two plot helpers
live in `plotting.py` (matplotlib only; the heatmap uses `imshow` + in-cell
annotations + a colorbar — no seaborn).

## Max-Sharpe via candidate selection

Maximising the Sharpe ratio is a non-convex fractional programme. Rather than a
fragile non-convex solve, `maximize_sharpe_ratio` reuses the constrained CVaR
efficient frontier as a generator of constraint-feasible candidate portfolios
(each produced by an LP), evaluates `(E[r] - rf) / vol` per candidate, and
returns the best. This is always feasible, honours every box / long-only
constraint, needs only the open-source LP solvers already in use, and reuses
`generate_cvar_efficient_frontier` + `calculate_portfolio_scenario_metrics`.

## Expected-return robustness layer

`estimate_expected_returns` gains `shrinkage_to_zero` (`shrinkage_weight × mean`).
On top of any estimator, `views.apply_manual_expected_return_views` blends user
point views into the vector (`blend_weight = 1` replaces, `0.5` is an even
blend). `views.py` is intentionally minimal — `AssetReturnView.confidence` is
reserved (unused) so a future Black-Litterman / Entropy-Pooling layer can plug in
behind the same call without touching the optimisers.

## Risk-free rate integration

`utils.annual_to_horizon_rate(annual, horizon_days, day_count)` =
`(1+annual)**(horizon_days/day_count) - 1`. The conversion happens at the
app/config boundary; the resulting per-horizon rate flows into the Sharpe ratio
(`calculate_portfolio_scenario_metrics(..., risk_free_rate=)`), the `Sharpe`
column of `compare_current_vs_optimized`, the Max-Sharpe optimiser, and the cash
asset's return. The existing "Include cash asset" checkbox and the manual cash
input are preserved (the manual input is used when the risk-free mode is "Zero").

## Streamlit session_state fix

The asset editor previously did
`st.session_state["assets_df"] = st.data_editor(st.session_state["assets_df"],
key="asset_editor")`. Re-assigning the return value into the same key that seeds
the widget makes Streamlit re-apply edit deltas on top of an already-edited
baseline and silently drop the first edit. The fix keeps the initial-default
baseline stable and reads edits from the widget's return value without writing
back — so the widget key persists edits across reruns. Validated with a Streamlit
`AppTest` (no network: the data fetch only fires on the Run click).
