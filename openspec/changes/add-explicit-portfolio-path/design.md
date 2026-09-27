# Portfolio Path V1 design

`portfolio_path.py` is a pure analytics boundary with no Streamlit, SQLAlchemy
or monitoring repository dependency. It accepts daily prices, long-only
unlevered initial/target weights and a frozen configuration. The result retains
portfolio- and asset-level histories plus versioned provenance.

At launch, quantities are established from initial capital and no setup cost is
charged. Every later complete close first values carried quantities. A scheduled
rebalance then solves post-cost NAV and trades jointly so post-trade weights
equal target. Those quantities apply only to the following interval. Costs are
the combined basis-point rate times gross absolute traded notional.

Weekly uses completed UTC `W-SUN` periods; monthly and quarterly use completed
calendar periods. Incomplete scheduled closes are never forward-filled. The
event remains pending and executes at the next complete observation with both
dates recorded. Risk horizon is not an engine input.

Risk Lab holds the selected path in versioned session state. Historical
descriptive/backtest analytics use its sequential net returns. Monte Carlo and
the optimizer's Current comparator use weights available at the end cutoff.
Current money risk uses latest net NAV. Portfolio Monitor remains on its existing
manual fixed-holdings contract and persistence schema.
