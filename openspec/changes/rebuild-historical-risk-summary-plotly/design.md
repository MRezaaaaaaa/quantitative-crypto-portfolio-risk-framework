# Design

## Data flow

```text
vendor prices + source-quality metadata
        ↓
Portfolio Path V1 (selected policy, costs, finalized UTC observations)
        ↓
HistoricalRiskSummary tables / existing risk result objects
        ↓
pure Plotly go.Figure builders
        ↓
Streamlit renderer + standalone HTML download
```

`historical_summary.py` consumes the already-built `PortfolioPathResult`; it
never reconstructs constant-weight returns.  Headline NAV, cumulative return,
drawdown and costs come from the finalized net path.  Daily descriptive moments
are unavailable when an included interval is not exactly one calendar day or
when a source-quality defect makes the daily label unsafe.

Source duplicate, order and missing-price observations are retained as DataFrame
metadata during normalization.  This lets accounting use its required clean
index without erasing defects from the audit display.

`plotting.py` and `portfolio_path_charts.py` remain Streamlit-independent.
Builders receive chart-ready inputs and return `go.Figure`; they do not estimate
risk, run tests, simulate scenarios or optimize weights.  `plotly_theme.py`
owns the palette and common layout.  Streamlit owns rendering and download
controls only.  PNG uses Plotly's client-side modebar; standalone HTML embeds
Plotly.js and therefore does not require Kaleido.

## Model-risk decisions

- Tail rows are historical distribution descriptions, not forward forecasts.
- Gapped finalized returns remain in the audited NAV path; they are not relabeled
  as daily observations or deleted to improve appearance.
- Monetary tail values scale the historical percentage statistic by ending net
  NAV and are not realized losses.
- Correlation scale remains fixed at `[-1, 1]`; the chosen method is explicit.
