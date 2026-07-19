![Phase](https://img.shields.io/badge/Phase-7%20Robust%20Assumptions-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Active-success)
![Version](https://img.shields.io/badge/Version-0.5.0-informational)

# Flexible VaR / CVaR Crypto Portfolio Risk Engine

> This project implements a flexible crypto portfolio VaR/CVaR risk engine in Python.
> The current 0.5.0 development version includes historical, Gaussian, and
> Cornish-Fisher VaR; historical and Gaussian CVaR; portfolio-level risk metrics;
> a Streamlit dashboard; VaR backtesting; Monte Carlo simulation; scenario-based
> CVaR optimization; and robust optimizer assumptions. Stress testing and risk
> contribution analysis remain planned extensions.

---

## Why VaR and CVaR matter

Value-at-Risk (VaR) gives a single, interpretable number for the loss threshold a
portfolio is unlikely to exceed at a given confidence level over a given horizon.
It is the canonical tool used by banks, brokers, and asset managers under the
Basel and FRTB capital frameworks. Conditional VaR (CVaR / Expected Shortfall)
addresses VaR's well-known weakness — it ignores how bad the tail can get — by
averaging the losses *beyond* the VaR threshold. CVaR is coherent in the sense of
Artzner et al. and is the regulatory standard under FRTB, which is why a serious
crypto risk engine needs both side-by-side.

---

## Methods implemented

| Family | Method | Module |
| --- | --- | --- |
| VaR | Historical Simulation | `var_models.HistoricalVaR` |
| VaR | Parametric Gaussian | `var_models.GaussianVaR` |
| VaR | Cornish-Fisher (skew/kurtosis adjusted) | `var_models.CornishFisherVaR` |
| CVaR | Historical Expected Shortfall | `cvar_models.HistoricalCVaR` |
| CVaR | Analytical Gaussian Expected Shortfall | `cvar_models.GaussianCVaR` |

All models extend an abstract base class so future families (Monte Carlo, GARCH,
Filtered Historical) can plug in without changing the consuming code.

---

## Data sources

| Source | Role | Notes |
| --- | --- | --- |
| **CoinGecko** | Primary | Free public endpoint, optional `COINGECKO_API_KEY` header. Cached to `data/cache/`. |
| **yfinance** | Fallback | Used if CoinGecko fails. Educational/research only — not licensed for production. |
| **CSV** | Manual | `load_price_data_from_csv()` for offline / reproducible runs. |

---

## Project structure

```
risk-management/
├── configs/
│   ├── config.yaml
│   └── assets.yaml
├── data/
│   ├── raw/    processed/    cache/
├── notebooks/
│   └── 01_core_var_cvar_validation.ipynb
├── outputs/
│   ├── charts/    tables/    reports/
├── openspec/
│   ├── specs/         (one per phase)
│   └── changes/phase-1-core-risk-engine/
├── src/var_cvar_crypto_risk/
│   ├── config.py            coingecko_client.py
│   ├── yfinance_client.py   data_loader.py
│   ├── preprocessing.py     returns.py
│   ├── portfolio.py         var_models.py
│   ├── cvar_models.py       risk_metrics.py
│   ├── backtesting.py       monte_carlo.py
│   ├── optimization.py      plotting.py
│   ├── export.py            utils.py
├── tests/
├── run_demo.py                              # Phases 1, 3, 4
└── run_phase5_optimization_demo.py          # Phase 5 (CVaR optimisation)
```

---

## Installation

```bash
pip install -e .
pip install -r requirements.txt
```

Python 3.10 or later is required.

---

## Running the demo

```bash
python run_demo.py
```

A single entry point that runs the **entire pipeline** end to end:
core risk engine (prices → returns → VaR / CVaR / drawdown + charts) **and**
Phase 3 backtesting (rolling forecasts, Kupiec / Christoffersen tests,
traffic-light verdict, model comparison). The script reads
`configs/config.yaml` + `configs/assets.yaml`, writes every CSV / PNG / JSON
to `outputs/`, and prints a clean console summary.

## Running the Streamlit web app

```bash
streamlit run app.py
```

This opens an interactive frontend at `http://localhost:8501` where you can:
- pick the data source (CoinGecko / yfinance) and date range,
- edit asset symbols, vendor IDs, and weights live in a table,
- adjust confidence level, time horizon, and which VaR/CVaR methods to compare,
- see headline VaR/CVaR cards plus the full risk summary table,
- inspect the distribution / cumulative / drawdown charts in tabs,
- download every CSV and PNG with one click.

## Running the tests

```bash
pytest tests/ -v
```

---

## Example console output

```
══════════════════════════════════════════════════
 Flexible VaR/CVaR Crypto Risk Engine — Core Risk Analysis
══════════════════════════════════════════════════

 Assets         : BTC (50%), ETH (30%), SOL (20%)
 Date Range     : 2021-01-01 to 2026-05-07
 Observations   : 1,583 trading days
 Initial Capital: $100,000
 Confidence     : 95.0%

──────────────────────────────────────────────────
 RISK METRICS
──────────────────────────────────────────────────
 Historical VaR     :   3.85%   ($3,850)
 Gaussian VaR       :   4.10%   ($4,100)
 Cornish-Fisher VaR :   4.65%   ($4,650)
 Historical CVaR    :   6.20%   ($6,200)
 Gaussian CVaR      :   5.15%   ($5,150)
 Max Drawdown       : -72.40%

──────────────────────────────────────────────────
 OUTPUTS SAVED
──────────────────────────────────────────────────
 outputs/tables/price_data.csv
 outputs/tables/asset_returns.csv
 outputs/tables/portfolio_returns.csv
 outputs/tables/risk_summary.csv
 outputs/charts/return_distribution_var_cvar.png
 outputs/charts/cumulative_returns.png
 outputs/charts/drawdown.png
══════════════════════════════════════════════════
```

---

## Methodology

**Historical VaR.** Take the empirical left-tail percentile of observed returns
and negate it to get a positive loss number. No distributional assumption — but
the answer is only as informative as the historical sample.

**Gaussian VaR.** Assume returns are normal with sample mean μ and sample std σ.
Compute `VaR = -(μ + σ · Φ⁻¹(1 − confidence))`. Fast and analytic, but
systematically understates risk for fat-tailed assets like crypto.

**Cornish-Fisher VaR.** Adjust the Gaussian z-score using sample skewness and
excess kurtosis, then apply the Gaussian formula. This captures the fat-tailed,
asymmetric behaviour of crypto returns much better than plain Gaussian VaR.

**Historical CVaR.** Average all observed returns at or below the historical VaR
threshold and negate. Non-parametric Expected Shortfall.

**Gaussian CVaR.** Use the closed-form expression
`CVaR = -(μ − σ · φ(z) / α)` with `α = 1 − confidence` and `z = Φ⁻¹(α)`.

---

## Limitations

- **Square-root-of-time scaling** assumes i.i.d. returns. Crypto exhibits
  volatility clustering and fat tails, so multi-day VaR computed this way is an
  *approximation* — typically a conservative one for short horizons and a
  permissive one when stress regimes persist.
- **Crypto regimes** shift dramatically (bull / bear / chop). Historical VaR
  trained on a single regime will misrepresent risk in the next one.
- **Free CoinGecko endpoint** has rate limits. The client retries with backoff
  and can fall back to yfinance, but production deployments should use a paid
  data feed.

---

## Phase 3: VaR Backtesting & Model Validation

### Why Backtesting Matters

A VaR number is only useful if it actually holds up. Basel III requires
internal models to be backtested daily against a 250-day window of clean
P&L; models with too many breaches face a higher supervisory capital
multiplier. Crypto markets shift regime far more often than equities, so
backtesting is doubly important here — a model calibrated on a calm month
can quietly under-report risk for the rest of the year. Phase 3 adds the
rolling forecast engine, the standard statistical coverage tests, and a
Basel-aligned traffic-light verdict to make that judgement defensible.

### Methodology

#### Rolling VaR Forecast

For each date `t` after the first `window` observations, the engine
computes a one-step-ahead VaR using only `returns[t-window:t]` —
strictly excluding `t` itself — and records the realised return
together with a breach indicator. The unit test
`test_rolling_var_forecast_no_lookahead` enforces this invariant
exactly, so any future refactor that reintroduces look-ahead bias fails
deterministically.

#### Breach / Exception Definition

`breach = actual_return < -var_forecast`

#### Statistical Tests

| Test | Null Hypothesis | Distribution | df |
|------|----------------|--------------|-----|
| Kupiec POF | Breach freq = model freq | χ² | 1 |
| Christoffersen Independence | Breaches are i.i.d. | χ² | 1 |
| Christoffersen CC | Both POF and independence | χ² | 2 |

#### Traffic Light System

**Basel III Mode** (250-day window, BCBS 2019):

| Zone | Breach Count | Action |
|------|-------------|--------|
| 🟢 Green | 0 – 4 | No action required |
| 🟡 Yellow | 5 – 9 | Review model |
| 🔴 Red | 10+ | Model likely inadequate |

**Rate-Based Mode** (non-standard windows):

| Zone | Breach Ratio |
|------|-------------|
| 🟢 Green | 0.75x – 1.25x expected |
| 🟡 Yellow | 0.50x – 1.75x expected |
| 🔴 Red | outside range |

Auto mode picks Basel III when `240 ≤ n ≤ 260`, rate-based otherwise.

### How to Run

```bash
streamlit run app.py
# Click "▶️ Run risk analysis" first, then open the
# "🔬 Backtesting & Model Validation" tab.
```

### Output Files

```
outputs/tables/var_forecasts_<method>.csv
outputs/tables/backtesting_results.csv
outputs/tables/model_comparison.csv
outputs/charts/var_backtesting_exceptions_<method>.png
outputs/charts/breach_timeline_<method>.png
outputs/charts/model_comparison_backtest.png
```

### Tests

```bash
pytest tests/test_backtesting.py -v
```

### Resume Bullet

> Implemented VaR backtesting and model validation using rolling
> one-step-ahead forecasts, breach analysis, Kupiec Proportion of
> Failures test, Christoffersen Independence and Conditional Coverage
> tests (Basel III traffic light framework), and multi-model comparison
> across Historical, Gaussian, and Cornish-Fisher VaR.

---

## Phase 3 Stabilization Update

Phase 3 outputs are now reproducible from a single command. The package
imports were made lightweight so the core analytics modules
(`var_models`, `cvar_models`, `backtesting`, `returns`, `portfolio`) do
not pull in `yfinance` or other optional data-source dependencies.

```bash
pip install -r requirements.txt
python run_demo.py          # full pipeline: Phase 1 core risk + Phase 3 backtesting
streamlit run app.py        # interactive dashboard (same pipeline, browser UI)
pytest                      # 194-test regression suite (Phases 1-5.5)
```

`python run_demo.py` saves:

- `outputs/tables/price_data.csv`, `asset_returns.csv`,
  `portfolio_returns.csv`, `portfolio_value.csv`, `risk_summary.csv`
- `outputs/tables/var_forecasts_<method>.csv` for every configured method
- `outputs/tables/model_comparison.csv`
- `outputs/tables/backtesting_results_<method>.json`
- `outputs/charts/return_distribution_var_cvar.png`, `cumulative_returns.png`,
  `drawdown.png`
- `outputs/charts/var_backtesting_exceptions_<method>.png`
- `outputs/charts/breach_timeline_<method>.png`
- `outputs/charts/model_comparison_backtest.png`

Rolling VaR forecasts, the model-comparison table, breach-timeline
charts, and the traffic-light verdict are written to disk as
deterministic files — not just downloadable from Streamlit — so they can
be checked into a results folder, used in a portfolio writeup, or
attached to a LinkedIn / GitHub README.

---

## Phase 4: Monte Carlo Scenario Engine

### Why Monte Carlo matters

Phase 1 and Phase 3 estimate risk *from what already happened*. Crypto
markets routinely produce shocks far worse than the historical tail.
The Monte Carlo engine simulates **thousands of forward-looking
scenarios** from a joint return model, so the user can answer "what
plausible losses could the next h days bring?" — not just "what is the
worst h-day loss we ever observed?".

### Normal Monte Carlo

Multivariate Normal scenarios are generated from the estimated mean
vector and covariance matrix. For `horizon_days = h`, both are scaled
linearly (i.i.d. approximation). VaR / CVaR are the empirical
left-tail percentile and tail mean of the simulated portfolio returns.

### Student-t Monte Carlo

Crypto return distributions are fat-tailed. A multivariate Student-t
with `df ≈ 4–6` reproduces this much better than Gaussian. The
covariance is rescaled by `(df − 2) / df` so the simulated samples
match the historical covariance, while still exhibiting heavier
tails — yielding more conservative VaR and CVaR estimates.

### Scenario-based VaR / CVaR

* `scenario_var` — left-tail percentile of simulated portfolio returns
  (positive loss number).
* `scenario_cvar` — average of simulated losses worse than the VaR
  threshold (positive loss number, ≥ VaR).
* `monte_carlo_risk_summary` — single-distribution Metric/Value/Unit
  table, with optional money VaR / CVaR.

### Portfolio path simulation

`simulate_portfolio_paths` draws Normal or Student-t daily returns and
compounds them into portfolio value paths over a forward horizon. The
first row of the output equals `initial_value` for every path; the rest
is a `(horizon × n_paths)` grid of plausible trajectories. The chart
overlays the mean path on a translucent fan of individual paths.

### Horizon-aware backtesting fix

Phase 3 previously backtested daily one-step-ahead VaR even when the
user selected `time_horizon_days > 1` for the headline VaR cards.
That inconsistency is fixed in Phase 4:

* `calculate_realized_horizon_returns(returns, horizon_days, method)` —
  forward h-day return at each forecast date.
* `rolling_var_forecast(..., horizon_days=h, return_method=...)` — for
  `h > 1`, the lookback window is aggregated into rolling h-day returns
  and VaR is computed from those. The realised value is the realised
  h-day forward return. No look-ahead bias.
* `horizon_days` is propagated through `backtest_var_model`,
  `compare_var_models_backtest`, and the report table (now includes a
  `Horizon (days)` column).
* The Streamlit Backtesting tab exposes the horizon input and labels
  every chart/table with the chosen horizon.

### How to run Phase 4

The unified entry point already runs Phase 4 as Step 3:

```bash
python run_demo.py
```

The Streamlit `🎲 Monte Carlo Scenario Engine` tab provides interactive
versions of every computation:

```bash
streamlit run app.py
```

### Output files generated

`outputs/tables/`:

- `mc_scenarios_normal.csv`, `mc_scenarios_student_t.csv`
- `mc_portfolio_returns_normal.csv`, `mc_portfolio_returns_student_t.csv`
- `mc_risk_summary.csv`
- `model_risk_comparison.csv`

`outputs/charts/`:

- `mc_loss_distribution_normal.png`,
  `mc_loss_distribution_student_t.png`
- `mc_portfolio_paths.png`
- `normal_vs_student_t_mc.png`
- `var_cvar_method_comparison.png`

### Limitations

* Multi-day Monte Carlo uses an i.i.d. approximation — volatility
  clustering is not modelled (deferred to Phase 6, GARCH).
* The Normal and Student-t copula structure is the same; dependence
  modelling will arrive with the copula layer (Phase 6).
* VaR/CVaR estimates carry simulation error of order
  `1 / √n_scenarios`; defaults are tuned for stable estimates but
  convergence diagnostics are not yet exposed.

### CVaR optimization integration

Phase 5 consumes the same `(n_scenarios × n_assets)` scenario matrix
produced by `simulate_normal_returns` / `simulate_student_t_returns` as input
to a Rockafellar–Uryasev linear program for minimum-CVaR portfolios.

### Resume bullet

> Developed a Monte Carlo scenario engine for multi-asset crypto
> portfolios, supporting Normal and Student-t simulations,
> scenario-based VaR/CVaR estimation, portfolio path simulation, and
> method comparison against historical and parametric risk models.

> Improved the VaR backtesting engine to support horizon-aware
> validation, allowing multi-day VaR forecasts to be compared against
> realised multi-day forward returns.

---

## Phase 5: CVaR Portfolio Optimization

### Why CVaR optimization matters

Phases 1–4 *measure* risk; Phase 5 *acts on it*. Given the scenarios you
trust — historical, Normal MC, or Student-t MC — the project now answers
the practical portfolio-construction question: **which weights should I
hold?**

* **Mean-variance has well-known weaknesses for crypto**: variance is
  symmetric (it punishes upside), and the Gaussian assumption is a poor
  fit for crypto's fat tails.
* **CVaR (Expected Shortfall)** measures the *expected loss in the worst
  β-tail*, focuses only on downside, and is mathematically coherent.
* Rockafellar & Uryasev (2000) reformulate scenario CVaR as a **linear
  programme**, which means a single LP — globally optimal, no curvature,
  no warm-starts — produces the entire CVaR-optimal weight vector.

### Scenario-based optimization in one paragraph

Let `R ∈ R^{N × d}` be the scenario return matrix (rows = scenarios,
columns = assets) and `w ∈ R^d` the weights. Portfolio loss per
scenario is `ℓ_i(w) = −R_i w`. For confidence level `β`, CVaR is

```
minimize_{w, t, u}   t + (1 / ((1 − β) · N)) · Σ_i u_i
subject to           u_i ≥ ℓ_i(w) − t        ∀ i
                     u_i ≥ 0                  ∀ i
                     Σ_j w_j = 1
                     w_j ∈ [w_min, w_max]   (or w_j ≥ 0 for long-only)
```

`t` is the implicit VaR; the objective value is the CVaR.

### Optimization objectives implemented

| Objective | Function |
|---|---|
| Minimum CVaR | `minimize_cvar` |
| Maximum expected return under CVaR ≤ L | `maximize_return_with_cvar_constraint` |
| Minimum CVaR for `E[r] @ w ≥ r*` | `minimize_cvar_for_target_return` |
| CVaR efficient frontier (sweep of `r*`) | `generate_cvar_efficient_frontier` |

Each optimiser returns a uniform result dict with `status`, `weights`,
`expected_return`, `VaR`, `CVaR`, `volatility`, `solver`, `message`
(plus objective-specific extras). Infeasible problems never raise — they
return `status != "optimal"` and a human-readable message, so the
Streamlit UI and the demo stay alive.

### Constraints supported

* **Long-only** (`w ≥ 0`) or short-selling allowed.
* **Min / max weight per asset** (box constraints).
* **Optional cash asset** with a constant per-horizon return — injected
  as an extra column in the scenario matrix.
* **Confidence level** in `(0, 1)`.
* **Target return** and **CVaR cap** for the constrained objectives.

### Scenario sources supported

The single `build_optimization_scenarios` adapter handles all three:

* `"historical"` — historical asset returns (rolling h-day simple
  aggregation for `horizon_days > 1`).
* `"normal_mc"` — multivariate Normal Monte Carlo (Phase 4).
* `"student_t_mc"` — multivariate Student-t Monte Carlo with covariance
  rescaling, for fat tails (Phase 4).

The LP is agnostic to where `R` came from — only the rows differ.

### How to run Phase 5

```bash
# CLI demo (saves CSVs and PNGs to outputs/)
python run_phase5_optimization_demo.py

# Streamlit (open the new "🎯 Portfolio Optimization" tab)
streamlit run app.py

# Regression suite
pytest
```

### How to use the Portfolio Optimization tab

1. Run the main **▶️ Run risk analysis** in the sidebar first so the app
   has prices and weights.
2. Open the **🎯 Portfolio Optimization** tab.
3. Pick a **scenario source** (Historical / Normal MC / Student-t MC),
   horizon, confidence level, and MC parameters.
4. Pick an **objective** (Min CVaR, Max return under CVaR cap, Target
   return, Efficient frontier, or Compare all).
5. Tune **constraints**: long-only, min/max weight, include cash,
   cash return, CVaR cap, target return, frontier points.
6. Click **▶️ Run optimization**. KPI cards, weights tables, the
   current-vs-optimised comparison, allocation chart, and (for the
   frontier) the efficient-frontier scatter all populate.
7. Download CSVs and PNGs from the buttons under each section.

### Output files generated

CSVs under `outputs/tables/`:

* `optimized_weights_min_cvar.csv`
* `optimized_weights_target_return.csv`
* `optimized_weights_cvar_cap.csv`
* `optimization_summary.csv`
* `portfolio_comparison.csv`
* `cvar_efficient_frontier.csv`

PNGs under `outputs/charts/`:

* `optimized_weights_min_cvar.png`
* `current_vs_optimized_risk.png`
* `cvar_efficient_frontier.png`
* `portfolio_allocation_comparison.png`

### Limitations

* Single-period optimisation only — no multi-period / drawdown-aware
  formulation.
* Expected returns are scenario averages (or medians, or zero); no
  Black-Litterman, Bayesian shrinkage, or forecasting layer.
* Open-source solvers only (CLARABEL / SCS / ECOS) — no Gurobi / Mosek.
* Risk contribution / component CVaR is deferred to Phase 6.

### Future roadmap to Phase 6

Phase 6 will add **risk contribution / component CVaR** on top of the
optimal scenario losses, plus **stress testing** by injecting
deterministic rows into the scenario matrix, and **GARCH / filtered
historical** scenario sources behind the same `build_optimization_scenarios`
interface. No breaking changes are expected.

### Resume bullets

> Implemented scenario-based CVaR portfolio optimization using Python
> and CVXPY, including minimum-CVaR portfolios, return maximization
> under CVaR constraints, target-return optimization, and CVaR
> efficient frontier analysis.

> Integrated historical and Monte Carlo scenario matrices into the
> optimization workflow, allowing portfolio weights to be optimized
> directly under tail-risk constraints.

---

## Phase 5.5: Fix & Enhancement Update

This update improves interpretability and robustness before adding advanced
Phase 6 risk layers. It avoids changing the core model architecture and focuses
on making the existing analytics more reliable and decision-useful.

**What's new**

- **Non-overlapping horizon backtesting** — `rolling_var_forecast`,
  `backtest_var_model`, and `compare_var_models_backtest` accept
  `backtest_mode="overlapping" | "non_overlapping"`. Non-overlapping uses
  independent horizon blocks (step = `horizon_days`), which is the statistically
  appropriate basis for the Christoffersen independence tests. Plus a rolling
  breach-rate chart, a worst-realised-losses table, and a by-year breach summary.
- **Horizon-matched distribution charts** — the Distribution tab now shows
  realised *h*-day returns (not √t-scaled daily VaR) when the risk horizon is
  > 1, with an optional "show all VaR lines" overlay, a QQ-plot vs Normal, a
  left-tail zoom, and a methodology panel.
- **Asset-level cumulative-return and drawdown charts** — alongside the
  portfolio-level views.
- **Correlation & Diversification tab** — Pearson/Spearman correlation matrix,
  a matplotlib heatmap, and a rolling average pairwise-correlation chart
  (diversification decay).
- **Maximum Sharpe Portfolio** — a constraint-feasible candidate-selection
  optimiser (`maximize_sharpe_ratio`) built on the CVaR frontier; no commercial
  solvers, always feasible.
- **Shrinkage expected-return estimator** — `shrinkage_to_zero`
  (`shrinkage_weight × mean`) to dampen the selection bias from assets with
  exceptional historical growth.
- **Manual expected-return views input layer** — `views.py`
  (`AssetReturnView`, `apply_manual_expected_return_views`): a clean seam for
  future Black-Litterman / Entropy-Pooling, without implementing either.
- **Risk-free rate support** — `annual_to_horizon_rate`, a Zero/Manual/Auto
  mode in the optimisation tab, a Sharpe column in the comparison table, and a
  cash return derived from the annual rate.
- **Asset editor state fix** — edits to the portfolio table now persist across
  reruns (the conflicting `session_state` write-back was removed).

```bash
streamlit run app.py        # interactive dashboard (new tab + controls)
python run_demo.py          # Phase 1 + 3 + 4 pipeline (unchanged outputs)
python run_phase5_optimization_demo.py
pytest                      # 194-test regression suite (Phases 1–5.5)
```

---

## Phase 7: Robust Assumptions Engine & Optimizer Input Governance

Phase 7 makes the assumptions that feed the optimizer **visible, robust,
and governed** instead of implicit.

### Robust Assumptions Engine (`assumptions.py` + new 🧠 tab)

* **Expected returns per asset, side by side**: historical mean, median,
  trimmed mean, winsorized mean, shrinkage-to-zero, manual views, and the
  **final expected return the optimizer actually receives** — all labelled
  *per optimization horizon* (never silently daily or annualized).
* **Robust volatility**: sample, winsorized, and EWMA (RiskMetrics,
  configurable λ) — daily, √t-horizon, and annualized columns.
* **Robust covariance**: sample, EWMA, and linear shrinkage toward a
  diagonal or constant-correlation (Ledoit-Wolf-style) target, with an
  implied-correlation heatmap. The optimizer's Monte Carlo sources can
  simulate directly from the robust covariance.
* The recipe is stored as an `AssumptionConfig` and **re-applied to the
  optimizer's own scenario matrix**, so source and horizon always stay
  consistent between the two tabs. The config is the seam where
  Black-Litterman / Entropy Pooling / scenario reweighting plug in later.

### Optimizer input governance

* An **"Inputs the optimizer actually received"** panel: scenario source,
  matrix dimensions, horizon, confidence, the exact expected-return
  vector, constraints, cash/risk-free assumptions.
* A per-objective **constraint applicability matrix** (which constraint
  binds which objective), plus interpretation guides for scenario-source
  sensitivity and CVaR-cap regime shifts.
* Per-result interpretation: binding CVaR cap / target return, cash
  weight, excluded assets, assets pinned at the minimum weight (forced
  diversification), and a defensive / balanced / aggressive profile
  derived from where the portfolio's CVaR sits in the feasible risk range.
* **Feasibility diagnostics** (`diagnose_infeasibility`): infeasible runs
  now explain *why* — CVaR cap below the minimum achievable CVaR, target
  return above the maximum achievable expected return, min/max weight
  budget conflicts — with the actual numbers.

### Issue fixes in this phase

* **Zero expected returns**: return-based objectives (Max Return, Max
  Sharpe, positive target return) are clearly flagged as not meaningful,
  in the UI and in the result dict (`result["warning"]`).
* **Cash × Max Sharpe**: near-zero-volatility candidates (≈100 % cash)
  are excluded from the Sharpe grid (`min_volatility` floor) and
  cash-dominated Sharpe portfolios carry an explicit warning — cash is an
  *absolute* defensive asset, unlike BTC which is only defensive relative
  to other crypto.
* **Horizon consistency**: every tab now states its convention; the Risk
  summary tab includes a cross-tab horizon-convention reference table, and
  the √t-scaled headline cards are labelled as such.
* **Asset-level VaR/CVaR**: asset distribution charts annotate the actual
  VaR/CVaR values, and the Distribution tab adds a per-asset
  horizon-matched risk table.
* **Mixed asset loading**: assets without a CoinGecko ID (e.g. GLD, SPY)
  are routed to yfinance automatically and ticker→symbol mapping is
  explicit, so `BTC + Gold + S&P 500` portfolios load consistently
  (weekends are dropped to the common trading calendar). Missing symbols
  now warn instead of disappearing silently.
* **Correlation interpretation**: portfolio-weighted average correlation,
  stress-day vs normal-day correlation (worst-decile portfolio days), and
  a rolling-window lag explanation.
* **Shared cached data layer**: horizon returns and scenario matrices are
  built once per input combination (`st.cache_data`) and shared across the
  Distribution / Monte Carlo / Assumptions / Optimizer tabs — faster and
  guaranteed cross-tab consistency, with automatic invalidation when
  inputs change.

### How to run / test Phase 7

```bash
streamlit run app.py    # new 🧠 Robust Assumptions tab + optimizer governance
pytest                  # 244-test regression suite (Phases 1–7)
pytest tests/test_assumptions.py tests/test_optimizer_governance.py -v
```

---

## Roadmap

- ✅ **Phase 1: Core Risk Engine**
- ✅ **Phase 2: Streamlit Interactive Dashboard**
- ✅ **Phase 3: Backtesting & Model Validation**
- ✅ **Phase 4: Monte Carlo Scenario Engine + horizon-aware backtesting**
- ✅ **Phase 5: CVaR Portfolio Optimization (Rockafellar-Uryasev)**
- ✅ **Phase 5.5: Fix & Enhancement Update (robustness + interpretability)**
- 🔜 **Phase 6: Advanced Risk Layer (GARCH, Copula, Risk Contribution, Stress Testing)**
- ✅ **Phase 7: Robust Assumptions Engine & Optimizer Input Governance**

---

## Resume-friendly project summary

> Built an open-source quantitative crypto portfolio risk engine in Python implementing
> Historical, Gaussian, and Cornish-Fisher VaR with CVaR/Expected Shortfall, complete
> with statistical backtesting framework, modular architecture for Streamlit deployment,
> and Basel III-aligned model validation, Monte Carlo scenario analysis, and
> scenario-based CVaR portfolio optimization.
