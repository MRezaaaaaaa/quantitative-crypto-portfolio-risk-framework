# Phase 1 — Design

## System architecture

The engine is a layered library with a single end-to-end demo on top.

```
┌──────────────────────────────────────────────────────────┐
│                  run_phase1_demo.py                      │
│  Orchestrates: config → data → returns → portfolio →     │
│                risk summary → charts → exports → console │
└──────────────────────────────────────────────────────────┘
                          │ uses
┌──────────────────────────────────────────────────────────┐
│ Output layer: risk_metrics.py, plotting.py, export.py    │
└──────────────────────────────────────────────────────────┘
                          │ uses
┌──────────────────────────────────────────────────────────┐
│ Calculation layer:                                       │
│   returns.py        portfolio.py                         │
│   var_models.py     cvar_models.py     utils.py          │
└──────────────────────────────────────────────────────────┘
                          │ uses
┌──────────────────────────────────────────────────────────┐
│ Data layer:                                              │
│   config.py        data_loader.py                        │
│   coingecko_client.py  yfinance_client.py                │
│   preprocessing.py                                       │
└──────────────────────────────────────────────────────────┘
```

Layers depend only on layers below them. The public `__init__.py` exposes a
small, deliberate API surface so consumers (Streamlit, backtesters, future
phases) never reach into private modules.

## Module boundaries

- **config.py** — Pure I/O on YAML files plus key-presence validation. No
  domain logic.
- **coingecko_client.py / yfinance_client.py** — One file per external
  vendor. Each owns its retry, header, and column-normalization logic.
- **data_loader.py** — Source selection and fallback orchestration. The only
  module that knows about more than one vendor.
- **preprocessing.py** — Sort / dedupe / align / fill. No vendor concerns.
- **returns.py** — Pure math on prices.
- **portfolio.py** — Weight validation, normalization, aggregation.
- **var_models.py / cvar_models.py** — Abstract base class plus concrete
  models. Module-level convenience functions delegate to instances. A small
  `_DISPATCH` dict gives a string-keyed entry point.
- **risk_metrics.py** — Composes VaR/CVaR models with descriptive statistics
  and produces the canonical summary table.
- **plotting.py / export.py** — File-side effects only.
- **utils.py** — Tiny helpers shared across modules.

## Data flow

```
configs/*.yaml
      │
      ▼
load_project_config ──► config dict
      │
      ▼
load_price_data ──► raw prices ──► clean_price_data ──► validated prices
      │
      ▼
calculate_returns ──► asset returns
      │
      ▼
get_weights_from_config ──► normalize_weights ──► validate_weights
      │
      ▼
calculate_portfolio_returns ──► portfolio return series
      │
      ▼
generate_risk_summary ──► risk DataFrame
      │
      ▼
plotting + export ──► PNGs + CSVs
```

## Key design decisions

- **ABC for VaR/CVaR.** New methods (Monte Carlo, GARCH, Filtered Historical)
  drop into the dispatch dict without touching consumers.
- **Sign convention.** All VaR and CVaR values are positive loss numbers.
  Drawdown is signed. This is documented in module docstrings *and* enforced
  in tests.
- **Configuration-driven.** No asset names, paths, or thresholds are
  hard-coded inside calculation functions. Everything that varies between
  runs lives in `config.yaml` / `assets.yaml`.
- **Deterministic tests.** Fixtures use seeded RNGs. No test touches the
  network.
- **Cache-friendly CoinGecko client.** Raw JSON payloads are cached under
  `data/cache/` keyed by `(coin_id, start, end)` to keep iterative work fast
  and to avoid hammering the free endpoint during tests.

## Future upgrade path

| Phase | Plug-in surface | Engine change required? |
| --- | --- | --- |
| Phase 2 — Streamlit | Imports public API from `__init__.py` | None |
| Phase 3 — Backtesting | New module consumes `VaRModel`/`CVaRModel` instances | None |
| Phase 4 — Monte Carlo | New ABC subclasses register in `_DISPATCH` | Add a key |
| Phase 5 — CVaR optimization | New `optimization.py` produces weights consumed by `validate_weights` | None |
| Phase 6 — GARCH / Copula / DCC | New ABC subclasses; rolling-window estimation lives in its own module | None |

Every future phase consumes the existing core; none rewrite it.
