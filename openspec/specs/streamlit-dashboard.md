# Streamlit Dashboard

> **Status: Implemented**

## Purpose

Deliver an interactive Streamlit application that wraps the analytics engine,
letting analysts configure portfolios, change confidence levels and horizons,
and inspect risk, validation, scenario, assumption, and optimization outputs
without modifying code.

## Implemented Scope

- Streamlit single-page, tabbed app (`app.py`) with sidebar configuration
  controls.
- Asset selector with live weight editor and validation.
- Confidence-level and horizon sliders.
- Tabs for: portfolio overview, distribution + VaR/CVaR chart, drawdown,
  per-method comparison table.
- Caching via `st.cache_data` for expensive fetches.
- Download buttons for the risk summary CSV and chart PNGs.

## Dependencies

Consumes the public analytics modules under `var_cvar_crypto_risk`; the
calculation modules do not import Streamlit.

## Current Capabilities

- **New "Correlation & Diversification" tab** — correlation matrix, matplotlib
  heatmap, and rolling average pairwise correlation (30/60/90/180-day window).
- **Distribution tab** — horizon-matched chart, Portfolio/Asset-level/Both
  selector, "show all VaR lines" overlay, QQ-plot, left-tail zoom, methodology
  panel.
- **Cumulative & Drawdown tabs** — now show portfolio + asset-level charts.
- **Backtesting tab** — overlapping / non-overlapping mode selector, rolling
  breach-rate chart, worst-realised-losses table (CSV), and per-year summary.
- **Optimization tab** — Maximize-Sharpe objective, shrinkage-to-zero estimator
  (with weight slider), manual expected-return views section, and a risk-free
  rate mode (Zero / Manual / Auto from config) feeding the Sharpe column and cash.
- **Asset editor** — edits persist across reruns (session_state write-back fix).
