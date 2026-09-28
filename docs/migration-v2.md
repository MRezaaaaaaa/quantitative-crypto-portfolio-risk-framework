# Migrating from v1.0 to v2.0

Version 2.0.0 adds persistent manual-portfolio monitoring and changes the public
chart contract from Matplotlib figures to Plotly figures. It does not change the
project's documented return convention or, by itself, validate any financial
model or performance claim.

## Chart API changes

Public chart builders now return `plotly.graph_objects.Figure`. Integrations
that previously expected Matplotlib figures must use a Plotly renderer or export
API instead. In Streamlit, render a figure with:

```python
st.plotly_chart(figure, width="stretch")
```

Use `figure.write_html(...)` for a self-contained interactive HTML export. The
browser modebar provides client-side image export where supported. Matplotlib is
no longer a runtime dependency.

The standalone `plot_tail_zoom_distribution` builder was removed. The primary
historical-distribution figure now contains the aligned historical tail overlay
and the VaR/CVaR reference lines. Consumers should use that figure and its Plotly
zoom, pan, autoscale and reset controls.

## Portfolio Monitor

Version 2.0.0 adds a separate Portfolio Monitor workspace. Before using it,
initialize or upgrade the private monitoring database:

```bash
uv sync --locked --extra app --extra dev
uv run --locked --no-sync alembic upgrade head
uv run --locked --no-sync streamlit run app.py
```

Set `QCPRF_MONITORING_DATABASE_URL` when the default local SQLite location is
not appropriate. Treat the database as private: it can contain holdings,
quantities, prices and realized portfolio results, and it must not be committed
to the public repository.

New monitoring experiments accept explicit manual asset weights. Experiment
creation does not run the optimizer, fit expected returns or reuse an optimizer
result from Streamlit session state. Existing version-1 risk analysis and
optimization remain available in the separate Risk Lab workspace.

The monitor records fixed quantities after launch and supports historical OOS,
live-forward and hybrid evaluation modes. A historical replay does not prove
that retrospectively entered weights were known at the declared decision date;
users must control hindsight and look-ahead bias outside the software. Live
monitoring is not trading, order generation or execution.

## Financial-methodology boundary

- Simple returns remain the decision-path convention for scenario wealth,
  portfolio paths, VaR/CVaR optimization and monitoring.
- The Plotly migration changes presentation and integration contracts, not the
  underlying financial formulas.
- Monitoring forecasts and realized observations remain point-in-time records;
  they are not evidence of predictive skill or future performance.
- Publication still requires legally usable pinned data, preserved provenance
  and an explicit out-of-sample design.

Read the [User Guide](user-guide.md),
[Methodology and Horizons](methodology.md),
[Portfolio Monitoring](portfolio-monitoring.md), and
[Model Risk](model-risk.md) before interpreting or publishing results.
