# Architecture

## Design objective

The platform separates reusable quantitative calculations from data access,
output generation, and Streamlit presentation. The long-term objective is a
testable Python package that can support multiple interfaces without duplicating
financial logic.

## Data flow

```text
CoinGecko / yfinance / CSV
            ↓
data_loader.py and preprocessing.py
            ↓
returns.py and portfolio.py
            ↓
┌───────────────┬─────────────────┬──────────────────┐
│ VaR / CVaR    │ Backtesting     │ Monte Carlo      │
│ risk metrics  │ validation      │ scenarios        │
└───────────────┴─────────────────┴──────────────────┘
            ↓                         ↓
       diagnostics              optimization
            └──────────────┬──────────┘
                           ↓
                 plotting / exports / UI
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Load and validate YAML configuration. |
| `coingecko_client.py`, `yfinance_client.py` | Vendor-specific price retrieval. |
| `data_loader.py`, `preprocessing.py` | Normalize and validate price data. |
| `returns.py`, `portfolio.py` | Return conventions, horizon aggregation, weights, and portfolio series. |
| `return_conventions.py` | Resolve the Automatic/Advanced return policy and calculation boundaries. |
| `var_models.py`, `cvar_models.py`, `risk_metrics.py` | Core risk measures. |
| `backtesting.py` | Rolling forecasts, breaches, coverage tests, and reports. |
| `monte_carlo.py` | Parameter estimates, scenarios, paths, and scenario risk. |
| `assumptions.py`, `views.py` | Robust optimizer inputs and manual return views. |
| `optimization.py` | Scenario construction, CVaR programs, frontier analysis, and diagnostics. |
| `correlation.py` | Dependence diagnostics. |
| `plotting.py`, `export.py` | Presentation-neutral figures and generated files. |
| `app.py` | Streamlit state, orchestration, and rendering. |

## Current technical debt

`app.py`, `optimization.py`, `plotting.py`, and `backtesting.py` are oversized.
The Streamlit file also owns substantial orchestration and state invalidation.
This increases the regression surface and makes it harder to distinguish UI
behavior from financial behavior.

Refactoring must occur only after numerical regression fixtures exist. The
target direction is:

```text
streamlit_app/
  app.py
  pages/
  components/

src/var_cvar_crypto_risk/
  data/
  risk/
  validation/
  simulation/
  optimization/
  reporting/
  services/
```

The target structure is directional, not a commitment to move every file.
Public APIs and numerical outputs should remain stable unless a methodology
change is separately specified and reviewed.

## State and caching

Streamlit session state stores analysis, backtesting, simulation, assumption,
and optimization results. Cached calculations improve responsiveness but create
a stale-state risk if an input is omitted from a cache key or invalidation path.
Future extraction should use explicit immutable request/configuration objects
and attach data provenance to each result.

## Architectural rules

1. Domain modules must not import Streamlit.
2. Vendor clients must not contain portfolio or risk calculations.
3. Plotting must consume results rather than recompute them.
4. All horizon conversions must be explicit and tested.
5. Scenario, wealth, and optimization boundaries must reject Log inputs rather
   than silently applying Simple-return arithmetic.
6. Methodology changes and architectural refactors must not share a commit.
7. Generated data, caches, and reports must remain outside version control.
