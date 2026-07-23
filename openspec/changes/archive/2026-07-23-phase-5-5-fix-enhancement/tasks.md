# Phase 5.5 — Fix & Enhancement Update — task list

## A. Backtesting fixes (`backtesting.py`, `plotting.py`, `app.py`)

- [x] `backtest_mode` (`overlapping` / `non_overlapping`) + `step_size` in
      `rolling_var_forecast`; output columns added.
- [x] Thread `backtest_mode` through `backtest_var_model` and
      `compare_var_models_backtest`; `Mode` column in the report table.
- [x] `calculate_rolling_breach_rate` + `plot_rolling_breach_rate`.
- [x] `get_worst_realized_losses` (Date / Actual Return / VaR Forecast / Breach / Loss).
- [x] `summarize_backtest_by_period` (per-year breach summary).
- [x] Backtesting-tab widget + explanatory caption; breach-rate chart, worst-losses
      table (CSV), and yearly summary wired in.

## B. Distribution enhancements (`returns.py`, `plotting.py`, `app.py`)

- [x] `calculate_horizon_returns` (simple/log, overlapping/non-overlapping).
- [x] Horizon-matched Distribution tab + `xlabel` on the chart.
- [x] "Show all VaR lines" overlay (`extra_var_lines`).
- [x] `plot_asset_return_distributions` + Portfolio/Asset-level/Both selector.
- [x] `plot_qq_vs_normal` (QQ plot expander).
- [x] `plot_tail_zoom_distribution` (left-tail zoom expander).
- [x] Methodology markdown panel.

## C. Asset-level charts (`risk_metrics.py`, `plotting.py`, `app.py`)

- [x] `calculate_asset_drawdowns`.
- [x] `plot_asset_cumulative_returns`, `plot_asset_drawdowns`.
- [x] Cumulative and Drawdown tabs show portfolio + asset-level.

## D. Correlation & Diversification (`correlation.py`, `plotting.py`, `app.py`)

- [x] `calculate_correlation_matrix` (pearson / spearman).
- [x] `calculate_rolling_average_correlation`.
- [x] `plot_correlation_heatmap` (matplotlib, in-cell values, colorbar).
- [x] `plot_rolling_average_correlation`.
- [x] New "Correlation & Diversification" tab (matrix, heatmap, rolling chart, downloads).

## E. Optimization enhancements (`optimization.py`, `views.py`, `app.py`)

- [x] `maximize_sharpe_ratio` (candidate selection over the CVaR frontier).
- [x] `estimate_expected_returns` gains `shrinkage_to_zero` + `shrinkage_weight`.
- [x] `views.py`: `AssetReturnView`, `apply_manual_expected_return_views`.
- [x] Objective selector + estimator selector + manual-views section wired in.

## F. Risk-free / Sharpe (`utils.py`, `optimization.py`, `app.py`, config)

- [x] `annual_to_horizon_rate`.
- [x] `risk_free_rate` + `sharpe_ratio` in `calculate_portfolio_scenario_metrics`;
      `Sharpe` column in `compare_current_vs_optimized`.
- [x] Risk-free rate mode (Zero / Manual / Auto from config) in the optimisation tab.

## G. Streamlit state fix (`app.py`)

- [x] Asset editor no longer writes the widget return value back into its own
      `session_state` key; edits persist across reruns.

## H. Config (`configs/config.yaml`)

- [x] `backtesting.default_mode` / `modes` / `rolling_breach_window`.
- [x] `distribution`, `correlation`, `risk_free_rate` sections.
- [x] `optimization.enable_max_sharpe` / `expected_return_estimators` /
      `shrinkage_weight` / `enable_manual_views`.

## I. Tests

- [x] Backtesting: non-overlap < overlap, h=1 modes equal, rolling breach rate,
      worst losses, by-period.
- [x] Distribution: horizon returns simple, horizon ≠ daily, QQ figure, tail-zoom figure.
- [x] Asset charts: asset cumulative shape, asset drawdowns ≤ 0, plot figures.
- [x] Correlation: square + named, diagonal == 1, heatmap figure, rolling series.
- [x] Optimization: Max-Sharpe weights sum to 1, has `sharpe_ratio`, shrinkage ==
      weight × mean, manual-views blend.
- [x] Risk-free: annual→horizon, zero ⇒ 0, manual value.
- [x] Streamlit state: asset-table init does not overwrite `session_state` (AppTest).
- [x] Full suite green (194 tests).

## J. README

- [x] Phase 5.5 section + roadmap + test count.

## K. OpenSpec

- [x] `proposal.md`, `tasks.md`, `design.md`.
- [x] Spec updates: `backtesting.md`, `cvar-optimization.md`, `core-risk-engine.md`,
      `streamlit-dashboard.md`.

## Out of scope

- Stress testing, risk contribution / component VaR/CVaR, GARCH, copulas,
  Black-Litterman, Entropy Pooling, PDF reports, commercial solvers (all Phase 6).
