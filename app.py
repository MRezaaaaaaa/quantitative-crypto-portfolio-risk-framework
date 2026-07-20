"""Streamlit frontend for the VaR/CVaR Crypto Portfolio Risk Engine.

Run from the project root:

    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from var_cvar_crypto_risk.coingecko_client import (  # noqa: E402
    CoinGeckoError,
    fetch_multiple_coingecko_prices,
)
from var_cvar_crypto_risk.backtesting import (  # noqa: E402
    backtest_var_model,
    calculate_rolling_breach_rate,
    compare_var_models_backtest,
    create_backtesting_report_table,
    get_worst_realized_losses,
    summarize_backtest_by_period,
)
from var_cvar_crypto_risk.assumptions import (  # noqa: E402
    AssumptionConfig,
    build_assumption_table,
    build_volatility_table,
)
from var_cvar_crypto_risk.correlation import (  # noqa: E402
    calculate_correlation_matrix,
    calculate_rolling_average_correlation,
    calculate_stress_vs_normal_correlation,
    calculate_weighted_average_correlation,
)
from var_cvar_crypto_risk.cvar_models import calculate_cvar  # noqa: E402
from var_cvar_crypto_risk.data_loader import validate_price_data  # noqa: E402
from var_cvar_crypto_risk.monte_carlo import (  # noqa: E402
    calculate_portfolio_scenario_returns,
    compare_all_risk_methods,
    scenario_cvar,
    scenario_var,
    simulate_portfolio_paths,
)
from var_cvar_crypto_risk.optimization import (  # noqa: E402
    add_cash_asset,
    build_optimization_scenarios,
    compare_current_vs_optimized,
    compute_feasible_risk_return_bounds,
    diagnose_infeasibility,
    estimate_expected_returns,
    format_weights_table,
    generate_cvar_efficient_frontier,
    interpret_optimization_result,
    maximize_return_with_cvar_constraint,
    maximize_sharpe_ratio,
    minimize_cvar,
    minimize_cvar_for_target_return,
)
from var_cvar_crypto_risk.plotting import (  # noqa: E402
    plot_allocation_comparison,
    plot_asset_cumulative_returns,
    plot_asset_drawdowns,
    plot_asset_return_distributions,
    plot_breach_timeline,
    plot_correlation_heatmap,
    plot_cumulative_returns,
    plot_cvar_efficient_frontier,
    plot_drawdown,
    plot_mc_loss_distribution,
    plot_mc_portfolio_paths,
    plot_model_comparison_backtest,
    plot_normal_vs_student_t_distribution,
    plot_optimized_weights,
    plot_portfolio_comparison,
    plot_qq_vs_normal,
    plot_return_distribution_with_var_cvar,
    plot_rolling_average_correlation,
    plot_rolling_breach_rate,
    plot_tail_zoom_distribution,
    plot_var_backtest,
    plot_var_cvar_method_comparison,
)
from var_cvar_crypto_risk.portfolio import (  # noqa: E402
    calculate_portfolio_returns,
    calculate_portfolio_value,
    normalize_weights,
    validate_weights,
)
from var_cvar_crypto_risk.preprocessing import clean_price_data  # noqa: E402
from var_cvar_crypto_risk.return_conventions import (  # noqa: E402
    resolve_return_policy,
)
from var_cvar_crypto_risk.returns import (  # noqa: E402
    calculate_horizon_returns,
    calculate_returns,
)
from var_cvar_crypto_risk.risk_metrics import (  # noqa: E402
    calculate_asset_drawdowns,
    calculate_max_drawdown,
    generate_risk_summary,
)
from var_cvar_crypto_risk.risk_conventions import (  # noqa: E402
    LOSS_SPACE_CONVENTION,
    loss_value_to_money,
)
from var_cvar_crypto_risk.utils import annual_to_horizon_rate  # noqa: E402
from var_cvar_crypto_risk.var_models import calculate_var  # noqa: E402
from var_cvar_crypto_risk.views import (  # noqa: E402
    AssetReturnView,
    apply_manual_expected_return_views,
)
from var_cvar_crypto_risk.yfinance_client import fetch_yfinance_prices  # noqa: E402


ASSETS_PATH = PROJECT_ROOT / "configs" / "assets.yaml"

VAR_METHODS = ["historical", "gaussian", "cornish_fisher"]
CVAR_METHODS = ["historical", "gaussian"]
METHOD_LABELS = {
    "historical": "Historical",
    "gaussian": "Gaussian",
    "cornish_fisher": "Cornish-Fisher",
}
RETURN_CONTRACT_VERSION = 1


# ─── Helpers ──────────────────────────────────────────────────────────────


def _load_default_assets() -> pd.DataFrame:
    with ASSETS_PATH.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    rows = []
    for symbol, meta in doc["assets"].items():
        rows.append(
            {
                "Symbol": symbol,
                "CoinGecko ID": meta.get("coingecko_id", ""),
                "yfinance Ticker": meta.get("yfinance_ticker", ""),
                "Weight": float(meta.get("weight", 0.0)),
            }
        )
    return pd.DataFrame(rows)


CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _load_risk_free_annual_from_config(default: float = 0.05) -> float:
    """Read ``risk_free_rate.annual_rate`` from config.yaml (default if absent)."""
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        return float(doc.get("risk_free_rate", {}).get("annual_rate", default))
    except (OSError, ValueError, TypeError):
        return default


def _fetch_yfinance_as_symbols(
    records: list[dict], start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch yfinance prices and rename columns from tickers to Symbols.

    yfinance normalizes ``BTC-USD`` → ``BTC`` but leaves non-crypto tickers
    (``GLD``, ``SPY``, ``^GSPC``) untouched, so a user Symbol like ``Gold``
    would silently never match. Renaming through an explicit ticker→Symbol
    map makes mixed crypto / non-crypto portfolios work consistently.
    """
    tickers = [str(row["yfinance Ticker"]).strip() for row in records]
    prices = fetch_yfinance_prices(
        tickers=tickers, start_date=start_date, end_date=end_date
    )
    ticker_to_symbol: dict[str, str] = {}
    for row in records:
        ticker = str(row["yfinance Ticker"]).strip()
        symbol = str(row["Symbol"]).strip()
        ticker_to_symbol[ticker] = symbol
        if ticker.upper().endswith("-USD"):  # yfinance client normalization
            ticker_to_symbol[ticker[:-4]] = symbol
    return prices.rename(columns=ticker_to_symbol)


@st.cache_data(show_spinner=False)
def _fetch_prices(
    source: str,
    fallback: str,
    assets_records: tuple,
    quote_currency: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    """Fetch and clean prices.

    Returns ``(prices, actual_source_used, warnings)``. Columns are the
    user's Symbols. Assets without a CoinGecko ID (e.g. Gold / S&P 500)
    are routed to yfinance even when the primary source is CoinGecko, so
    mixed crypto / traditional portfolios load consistently.
    """
    records = [dict(row) for row in assets_records]
    warnings: list[str] = []

    def _has_cg_id(row: dict) -> bool:
        return bool(str(row.get("CoinGecko ID", "") or "").strip())

    def _has_ticker(row: dict) -> bool:
        return bool(str(row.get("yfinance Ticker", "") or "").strip())

    used_source = source
    frames: list[pd.DataFrame] = []

    if source == "coingecko":
        cg_records = [r for r in records if _has_cg_id(r)]
        yf_only_records = [r for r in records if not _has_cg_id(r)]
        if yf_only_records:
            no_route = [r["Symbol"] for r in yf_only_records if not _has_ticker(r)]
            if no_route:
                raise ValueError(
                    f"Assets {no_route} have neither a CoinGecko ID nor a "
                    "yfinance Ticker — no data source can serve them."
                )
            warnings.append(
                "No CoinGecko ID for "
                + ", ".join(r["Symbol"] for r in yf_only_records)
                + " — fetched via yfinance instead. Note: non-crypto assets "
                "trade ~5 days/week, so mixed portfolios are aligned to "
                "common trading days (weekends dropped)."
            )
        if cg_records:
            assets_dict = {
                row["Symbol"]: {
                    "coingecko_id": row["CoinGecko ID"],
                    "yfinance_ticker": row["yfinance Ticker"],
                }
                for row in cg_records
            }
            try:
                frames.append(
                    fetch_multiple_coingecko_prices(
                        assets=assets_dict,
                        vs_currency=quote_currency,
                        start_date=start_date,
                        end_date=end_date,
                        cache_dir=str(PROJECT_ROOT / "data" / "cache"),
                        use_cache=True,
                    )
                )
            except CoinGeckoError as exc:
                if fallback == "yfinance":
                    frames.append(
                        _fetch_yfinance_as_symbols(
                            cg_records, start_date, end_date
                        )
                    )
                    used_source = "yfinance (fallback)"
                    warnings.append(
                        f"CoinGecko failed ({exc}). Using yfinance fallback."
                    )
                else:
                    raise
        if yf_only_records:
            frames.append(
                _fetch_yfinance_as_symbols(yf_only_records, start_date, end_date)
            )
            if cg_records and used_source == "coingecko":
                used_source = "coingecko + yfinance"
    elif source == "yfinance":
        prices_yf = _fetch_yfinance_as_symbols(records, start_date, end_date)
        frames.append(prices_yf)
    else:
        raise ValueError(f"Unsupported source: {source}")

    prices = pd.concat(frames, axis=1, join="outer") if frames else pd.DataFrame()

    symbols = [row["Symbol"] for row in records]
    available = [s for s in symbols if s in prices.columns]
    missing = [s for s in symbols if s not in prices.columns]
    if missing:
        warnings.append(
            f"No price data returned for: {', '.join(missing)} — these "
            "assets were dropped. Check the vendor ID / ticker."
        )
    prices = prices[available]
    cleaned = clean_price_data(prices)
    validate_price_data(cleaned)
    return cleaned, used_source, tuple(warnings)


# ─── Shared cached data layer (Phase 7) ───────────────────────────────────
# Base data (prices → returns → portfolio returns) is computed once per
# "Run risk analysis" click and stored in session_state. The two derived
# artifacts that used to be recomputed per tab / per rerun — horizon
# returns and scenario matrices — are cached here on their full input key,
# so every tab that asks for the same (inputs) gets the same object back
# instantly. st.cache_data hashes the DataFrame contents, so a new data
# run or a changed parameter invalidates dependent entries automatically
# (no stale-cache risk).


@st.cache_data(show_spinner=False)
def _horizon_returns_cached(
    returns: pd.Series, horizon_days: int, method: str
) -> pd.Series:
    return calculate_horizon_returns(
        returns, horizon_days=int(horizon_days), method=method
    )


@st.cache_data(show_spinner=False)
def _scenario_matrix_cached(
    asset_returns: pd.DataFrame,
    source: str,
    n_scenarios: int,
    horizon_days: int,
    student_t_df: float,
    random_seed: int,
    covariance_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Single scenario-matrix builder shared by the Monte Carlo, Robust
    Assumptions, and Optimizer tabs — same inputs ⇒ same matrix in every
    tab."""
    return build_optimization_scenarios(
        asset_returns=asset_returns,
        source=source,
        n_scenarios=int(n_scenarios),
        horizon_days=int(horizon_days),
        student_t_df=float(student_t_df),
        random_seed=int(random_seed),
        covariance_matrix=covariance_matrix,
        return_method="simple",
    )


def _df_to_csv_bytes(df: pd.DataFrame, include_index: bool = True) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=include_index)
    return buf.getvalue().encode("utf-8")


def _fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    return buf.getvalue()


def _format_money(value: float) -> str:
    return f"${value:,.0f}"


# ─── Page setup ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Crypto VaR/CVaR Risk Engine",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Crypto Portfolio VaR / CVaR Risk Engine")
st.caption(
    "Interactive frontend for the Phase 1 risk engine — "
    "Historical, Gaussian, Cornish-Fisher VaR + Historical / Gaussian CVaR."
)


# ─── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    source = st.selectbox(
        "Data source",
        options=["coingecko", "yfinance"],
        index=0,
        help="CoinGecko is primary. yfinance is the fallback.",
    )
    fallback_enabled = st.checkbox(
        "Fall back to yfinance if CoinGecko fails", value=True
    )
    quote_currency = st.selectbox("Quote currency", ["usd"], index=0)

    today = datetime.now(tz=timezone.utc).date()
    start_default = date(2021, 1, 1)
    start_date = st.date_input(
        "Start date", value=start_default, min_value=date(2015, 1, 1), max_value=today
    )
    end_date = st.date_input(
        "End date", value=today, min_value=start_date, max_value=today
    )

    st.divider()
    st.subheader("Returns & portfolio")
    return_handling_mode = st.selectbox(
        "Return handling",
        ["automatic", "advanced"],
        format_func=lambda value: value.title(),
        help=(
            "Automatic uses simple returns throughout. Advanced can use log "
            "returns for distribution diagnostics only."
        ),
    )
    if return_handling_mode == "advanced":
        diagnostic_return_method = st.selectbox(
            "Diagnostic return convention",
            ["simple", "log"],
            format_func=lambda value: value.title(),
        )
    else:
        diagnostic_return_method = "simple"
    return_policy = resolve_return_policy(
        return_handling_mode,
        diagnostic_method=diagnostic_return_method,
    )
    st.caption(
        "Portfolio construction, NAV, risk monitoring, Monte Carlo, and "
        "optimization always use simple returns. Advanced Log affects only "
        "distribution diagnostics."
    )
    initial_capital = st.number_input(
        "Initial capital (USD)",
        min_value=100.0,
        value=100_000.0,
        step=1_000.0,
        format="%.2f",
    )
    auto_normalize = st.checkbox("Auto-normalize weights to 1.0", value=True)
    allow_short = st.checkbox("Allow short selling (negative weights)", value=False)

    st.divider()
    st.subheader("Risk parameters")
    confidence_level = st.slider(
        "Confidence level", min_value=0.80, max_value=0.999, value=0.95, step=0.005
    )
    horizon_days = st.number_input(
        "Time horizon (days)",
        min_value=1,
        max_value=60,
        value=1,
        help="VaR is scaled by sqrt(horizon). Use 1 for daily VaR.",
    )

    selected_var_methods = st.multiselect(
        "VaR methods", options=VAR_METHODS, default=VAR_METHODS,
        format_func=lambda m: METHOD_LABELS[m],
    )
    selected_cvar_methods = st.multiselect(
        "CVaR methods", options=CVAR_METHODS, default=CVAR_METHODS,
        format_func=lambda m: METHOD_LABELS[m],
    )


# ─── Asset / weight editor ────────────────────────────────────────────────

st.subheader("📊 Portfolio")
st.caption(
    "Edit asset symbols, vendor IDs, and weights below. "
    "Weights should sum to 1.0 (auto-normalized if enabled in the sidebar)."
)

if "assets_df" not in st.session_state:
    st.session_state["assets_df"] = _load_default_assets()

# The data_editor's own widget state (key="asset_editor") persists edits
# across reruns against a *stable* baseline. Re-assigning the return value
# back into the baseline key would make Streamlit re-apply the edit deltas
# on top of an already-edited frame and silently drop the first edit, so we
# read the edited frame from the return value and do NOT write it back.
assets_df = st.data_editor(
    st.session_state["assets_df"],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Weight": st.column_config.NumberColumn(
            "Weight", min_value=-2.0, max_value=2.0, step=0.05, format="%.4f"
        ),
    },
    key="asset_editor",
)

weights_sum = float(assets_df["Weight"].sum())
col_a, col_b = st.columns([1, 3])
with col_a:
    st.metric("Sum of weights", f"{weights_sum:.4f}")
with col_b:
    if abs(weights_sum - 1.0) > 1e-6 and not auto_normalize:
        st.warning(
            "Weights do not sum to 1.0. Enable auto-normalize in the sidebar "
            "or fix manually before running."
        )

if "risk_results" not in st.session_state:
    st.session_state["risk_results"] = None
if "backtest_results" not in st.session_state:
    st.session_state["backtest_results"] = None
if "mc_results" not in st.session_state:
    st.session_state["mc_results"] = None
if "opt_results" not in st.session_state:
    st.session_state["opt_results"] = None
if "assumptions_results" not in st.session_state:
    st.session_state["assumptions_results"] = None

run = st.button("▶️ Run risk analysis", type="primary", use_container_width=True)


# ─── Main analysis ────────────────────────────────────────────────────────

if run:
    try:
        assets_records = tuple(
            assets_df.dropna(subset=["Symbol"])
            .assign(Symbol=lambda d: d["Symbol"].str.strip())
            .query("Symbol != ''")
            .to_dict(orient="records")
        )
        if not assets_records:
            st.error("Add at least one asset with a non-empty Symbol.")
            st.stop()

        with st.spinner("Fetching prices…"):
            prices, used_source, fetch_warnings = _fetch_prices(
                source=source,
                fallback="yfinance" if fallback_enabled else "",
                assets_records=assets_records,
                quote_currency=quote_currency,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )

        if prices.shape[1] < 1 or len(prices) < 5:
            st.error(
                f"Not enough price data (rows={len(prices)}, "
                f"cols={prices.shape[1]}). Widen the date range."
            )
            st.stop()

        weights = pd.Series(
            {row["Symbol"]: float(row["Weight"]) for row in assets_records},
            dtype=float,
        )
        weights = weights.reindex(prices.columns).dropna()
        if auto_normalize:
            weights = normalize_weights(weights)
        validate_weights(
            weights,
            assets=list(prices.columns),
            allow_short_selling=allow_short,
        )

        asset_returns = calculate_returns(
            prices,
            method=return_policy.portfolio_method,
        )
        portfolio_returns = calculate_portfolio_returns(
            asset_returns,
            weights,
            return_method=return_policy.portfolio_method,
        )
        portfolio_value = calculate_portfolio_value(
            portfolio_returns,
            initial_capital,
            return_method=return_policy.wealth_method,
        )
        diagnostic_asset_returns = (
            asset_returns
            if return_policy.diagnostic_method == "simple"
            else calculate_returns(prices, method="log")
        )
        diagnostic_portfolio_returns = calculate_portfolio_returns(
            diagnostic_asset_returns,
            weights,
            return_method=return_policy.diagnostic_method,
        )

        risk_summary = generate_risk_summary(
            portfolio_returns=portfolio_returns,
            confidence_level=confidence_level,
            initial_capital=initial_capital,
            var_methods=selected_var_methods,
            cvar_methods=selected_cvar_methods,
            return_method=return_policy.portfolio_method,
        )
    except CoinGeckoError as exc:
        st.error(f"CoinGecko error: {exc}")
        st.stop()
    except ValueError as exc:
        st.error(f"Validation error: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error: {exc}")
        raise

    for warning_msg in fetch_warnings:
        st.warning(warning_msg)

    st.session_state["risk_results"] = {
        "return_contract_version": RETURN_CONTRACT_VERSION,
        "prices": prices,
        "asset_returns": asset_returns,
        "portfolio_returns": portfolio_returns,
        "diagnostic_asset_returns": diagnostic_asset_returns,
        "diagnostic_portfolio_returns": diagnostic_portfolio_returns,
        "portfolio_value": portfolio_value,
        "risk_summary": risk_summary,
        "used_source": used_source,
        "selected_assets": list(prices.columns),
        "weights": weights,
        "confidence_level": confidence_level,
        "initial_value": initial_capital,
        "selected_var_methods": list(selected_var_methods),
        "selected_cvar_methods": list(selected_cvar_methods),
        "horizon_days": horizon_days,
        "return_handling_mode": return_policy.handling_mode,
        "diagnostic_return_method": return_policy.diagnostic_method,
        "returns_method": return_policy.diagnostic_method,
    }
    st.session_state["backtest_results"] = None
    st.session_state["mc_results"] = None
    st.session_state["opt_results"] = None
    st.session_state["assumptions_results"] = None

results = st.session_state.get("risk_results")
if (
    results is not None
    and results.get("return_contract_version") != RETURN_CONTRACT_VERSION
):
    st.session_state["risk_results"] = None
    st.session_state["backtest_results"] = None
    st.session_state["mc_results"] = None
    st.session_state["opt_results"] = None
    st.session_state["assumptions_results"] = None
    results = None
    st.info(
        "The return-convention contract changed. Run the analysis again to "
        "rebuild all results with Simple-return core inputs."
    )
if results is None:
    st.info("Configure inputs in the sidebar and click **Run risk analysis**.")
    st.stop()

prices = results["prices"]
asset_returns = results["asset_returns"]
portfolio_returns = results["portfolio_returns"]
diagnostic_asset_returns = results.get("diagnostic_asset_returns", asset_returns)
diagnostic_portfolio_returns = results.get(
    "diagnostic_portfolio_returns", portfolio_returns
)
portfolio_value = results["portfolio_value"]
risk_summary = results["risk_summary"]
used_source = results["used_source"]
weights = results["weights"]
confidence_level = results["confidence_level"]
initial_capital = results["initial_value"]
selected_var_methods = results["selected_var_methods"]
selected_cvar_methods = results["selected_cvar_methods"]
horizon_days = results["horizon_days"]
return_handling_mode = results.get("return_handling_mode", "automatic")
diagnostic_return_method = results.get(
    "diagnostic_return_method",
    results.get("returns_method", "simple"),
)


# ─── Run summary ──────────────────────────────────────────────────────────

st.success(
    f"Loaded {len(prices):,} price rows × {prices.shape[1]} assets "
    f"from **{used_source}** ({prices.index.min().date()} → {prices.index.max().date()})."
)
st.caption(
    "Return conventions — core portfolio/NAV/scenarios/optimization: "
    f"**Simple** · diagnostics: **{diagnostic_return_method.title()}** · "
    f"mode: **{return_handling_mode.title()}**"
)

obs = len(portfolio_returns)
cum_return = float((1.0 + portfolio_returns).prod() - 1.0)
ann_vol = float(portfolio_returns.std(ddof=1)) * (365 ** 0.5)
max_dd = calculate_max_drawdown(portfolio_returns)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Observations", f"{obs:,}")
m2.metric("Cumulative return", f"{cum_return * 100:.2f}%")
m3.metric("Ann. volatility", f"{ann_vol * 100:.2f}%")
m4.metric("Max drawdown", f"{max_dd * 100:.2f}%")


# ─── Headline VaR / CVaR cards ────────────────────────────────────────────

st.subheader(f"🎯 VaR & CVaR at {confidence_level * 100:.1f}% confidence")
st.caption(
    f"Sign convention — {LOSS_SPACE_CONVENTION} Headline risk uses simple "
    "portfolio returns."
)

import numpy as np  # noqa: E402

scale = float(np.sqrt(horizon_days))
horizon_label = f"{horizon_days}-day" if horizon_days != 1 else "1-day"

card_cols = st.columns(max(1, len(selected_var_methods) + len(selected_cvar_methods)))
idx = 0
for method in selected_var_methods:
    var_pct = calculate_var(portfolio_returns, method, confidence_level) * scale
    money_var = loss_value_to_money(var_pct, initial_capital)
    card_cols[idx].metric(
        f"{METHOD_LABELS[method]} VaR ({horizon_label})",
        f"{var_pct * 100:.2f}%",
        delta=_format_money(money_var),
        delta_color="off",
    )
    idx += 1
for method in selected_cvar_methods:
    cvar_pct = calculate_cvar(portfolio_returns, method, confidence_level) * scale
    money_cvar = loss_value_to_money(cvar_pct, initial_capital)
    card_cols[idx].metric(
        f"{METHOD_LABELS[method]} CVaR ({horizon_label})",
        f"{cvar_pct * 100:.2f}%",
        delta=_format_money(money_cvar),
        delta_color="off",
    )
    idx += 1

if horizon_days > 1:
    st.caption(
        f"⚠️ Headline cards are **√t-scaled daily** VaR/CVaR "
        f"(daily × √{horizon_days}, i.i.d. approximation). The Distribution "
        f"tab shows **realised {horizon_days}-day** returns instead — the "
        "two conventions can legitimately differ."
    )


# ─── Tabs ─────────────────────────────────────────────────────────────────

(
    tab_summary,
    tab_dist,
    tab_growth,
    tab_dd,
    tab_corr,
    tab_assumptions,
    tab_data,
    tab_backtest,
    tab_mc,
    tab_opt,
) = st.tabs(
    [
        "📋 Risk summary",
        "📈 Distribution",
        "💹 Cumulative",
        "📉 Drawdown",
        "🧩 Correlation & Diversification",
        "🧠 Robust Assumptions",
        "🗂 Data",
        "🔬 Backtesting & Model Validation",
        "🎲 Monte Carlo Scenario Engine",
        "🎯 Portfolio Optimization",
    ]
)

with tab_summary:
    st.caption(
        "All VaR/CVaR figures in this table are **1-day (daily)** at the "
        "selected confidence level, computed from daily portfolio returns. "
        "Return/volatility rows state their own horizon (daily or "
        "annualized) in the metric name."
    )
    st.dataframe(risk_summary, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download risk_summary.csv",
        data=_df_to_csv_bytes(risk_summary, include_index=False),
        file_name="risk_summary.csv",
        mime="text/csv",
    )

    with st.expander("📖 Horizon conventions across tabs", expanded=False):
        st.markdown(
            f"""
Different tabs intentionally use different horizon conventions. Same
label, different basis — this table is the reference:

| Where | Convention | Basis |
|---|---|---|
| Risk summary table (this tab) | **Daily** VaR/CVaR | Daily returns |
| Headline cards (top of page) | Daily × √{int(horizon_days)} (**√t-scaled**) | i.i.d. approximation |
| Distribution tab | **Realised {int(horizon_days)}-day** returns | Overlapping horizon returns |
| Backtesting tab | Horizon selected in that tab | Rolling / non-overlapping realised returns |
| Monte Carlo tab | **Simulated h-day** scenarios | Mean/cov scaled ×h (i.i.d.) |
| Optimizer tab | **h-day scenario** VaR/CVaR | Scenario source selected there |

For a 1-day horizon all conventions coincide. For multi-day horizons,
√t-scaling understates risk when losses cluster; realised horizon
returns and simulated scenarios capture that clustering differently —
so two numbers with the same confidence level can legitimately differ.
"""
        )

with tab_dist:
    if selected_var_methods and selected_cvar_methods:
        h = int(horizon_days)
        horizon_label = "Daily" if h == 1 else f"{h}-day"

        d_c1, d_c2 = st.columns([2, 1])
        with d_c1:
            dist_scope = st.radio(
                "Show",
                ["Portfolio", "Asset-level", "Both"],
                horizontal=True,
                key="dist_scope",
            )
        with d_c2:
            show_all_var = st.checkbox(
                "Show all VaR lines", value=False, key="dist_all_var"
            )

        if h > 1:
            st.caption(
                f"Distribution is horizon-matched: it shows realised "
                f"**{h}-day** returns (not √t-scaled daily VaR)."
            )
        st.caption(
            "Diagnostic convention: "
            f"**{diagnostic_return_method.title()} returns**. Log mode is "
            "diagnostic only and does not alter NAV, risk monitoring, "
            "Monte Carlo, or optimization."
        )

        # Horizon-matched portfolio returns (h == 1 ⇒ daily, unchanged).
        # Served from the shared cache — same series every tab, no recompute.
        dist_returns = _horizon_returns_cached(
            diagnostic_portfolio_returns,
            h,
            diagnostic_return_method,
        )

        primary_var = selected_var_methods[0]
        primary_cvar = selected_cvar_methods[0]
        var_value = calculate_var(dist_returns, primary_var, confidence_level)
        cvar_value = calculate_cvar(dist_returns, primary_cvar, confidence_level)

        extra_lines = None
        if show_all_var:
            extra_lines = {
                f"{METHOD_LABELS[m]} VaR": calculate_var(
                    dist_returns, m, confidence_level
                )
                for m in selected_var_methods
            }

        if dist_scope in ("Portfolio", "Both"):
            fig = plot_return_distribution_with_var_cvar(
                dist_returns,
                var_value=var_value,
                cvar_value=cvar_value,
                confidence_level=confidence_level,
                title=(
                    f"{horizon_label} Return Distribution — "
                    f"{METHOD_LABELS[primary_var]} VaR & "
                    f"{METHOD_LABELS[primary_cvar]} CVaR"
                ),
                xlabel=f"{horizon_label} Return",
                extra_var_lines=extra_lines,
            )
            st.pyplot(fig, use_container_width=True)
            st.download_button(
                "⬇️ Download distribution chart (PNG)",
                data=_fig_to_png_bytes(fig),
                file_name="return_distribution_var_cvar.png",
                mime="image/png",
                key="dl_dist_portfolio",
            )
            plt.close(fig)

        if dist_scope in ("Asset-level", "Both"):
            fig_assets = plot_asset_return_distributions(
                diagnostic_asset_returns,
                horizon_days=h,
                confidence_level=confidence_level,
                return_method=diagnostic_return_method,
            )
            st.pyplot(fig_assets, use_container_width=True)
            st.download_button(
                "⬇️ Download asset distributions (PNG)",
                data=_fig_to_png_bytes(fig_assets),
                file_name="asset_return_distributions.png",
                mime="image/png",
                key="dl_dist_assets",
            )
            plt.close(fig_assets)

            # Asset-level risk table (historical, horizon-matched).
            asset_risk_rows = []
            for asset in diagnostic_asset_returns.columns:
                a_series = _horizon_returns_cached(
                    diagnostic_asset_returns[asset].dropna(),
                    h,
                    diagnostic_return_method,
                )
                a_var = calculate_var(a_series, "historical", confidence_level)
                a_cvar = calculate_cvar(a_series, "historical", confidence_level)
                asset_risk_rows.append(
                    {
                        "Asset": asset,
                        f"{horizon_label} VaR (%)": a_var * 100.0,
                        f"{horizon_label} CVaR (%)": a_cvar * 100.0,
                        f"{horizon_label} Vol (%)": float(
                            a_series.std(ddof=1)
                        ) * 100.0,
                    }
                )
            st.markdown(
                f"**Asset-level historical risk "
                f"({horizon_label}, {confidence_level * 100:.1f}% confidence)**"
            )
            st.dataframe(
                pd.DataFrame(asset_risk_rows).round(2),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("🔬 QQ Plot vs Normal", expanded=False):
            st.caption(
                "Large deviations in the tails indicate non-normality and "
                "potential fat-tail behavior."
            )
            fig_qq = plot_qq_vs_normal(
                dist_returns,
                title=f"QQ Plot vs Normal — {horizon_label} Portfolio Returns",
            )
            st.pyplot(fig_qq, use_container_width=True)
            plt.close(fig_qq)

        with st.expander("🔻 Left-tail zoom", expanded=False):
            fig_tail = plot_tail_zoom_distribution(
                dist_returns,
                var_value=var_value,
                cvar_value=cvar_value,
                confidence_level=confidence_level,
            )
            st.pyplot(fig_tail, use_container_width=True)
            plt.close(fig_tail)

        with st.expander("📖 How each method is calculated", expanded=False):
            st.markdown(
                """
**Historical**
- VaR is the empirical left-tail percentile.
- CVaR is the average loss beyond the VaR threshold.

**Gaussian**
- VaR and CVaR use the mean and standard deviation under a normality
  assumption.
- This may underestimate tail risk when excess kurtosis is high.

**Cornish-Fisher**
- Adjusts the Gaussian quantile using skewness and excess kurtosis.
- Provides a modified VaR for non-normal returns.
- CVaR is not currently implemented for Cornish-Fisher.
"""
            )
    else:
        st.info("Select at least one VaR and one CVaR method in the sidebar.")

with tab_growth:
    st.markdown("**Portfolio cumulative return**")
    fig = plot_cumulative_returns(portfolio_returns)
    st.pyplot(fig, use_container_width=True)
    st.download_button(
        "⬇️ Download cumulative returns chart (PNG)",
        data=_fig_to_png_bytes(fig),
        file_name="cumulative_returns.png",
        mime="image/png",
        key="dl_cum_portfolio",
    )
    plt.close(fig)

    st.markdown("**Asset-level cumulative returns**")
    fig_assets = plot_asset_cumulative_returns(asset_returns)
    st.pyplot(fig_assets, use_container_width=True)
    st.download_button(
        "⬇️ Download asset cumulative returns (PNG)",
        data=_fig_to_png_bytes(fig_assets),
        file_name="asset_cumulative_returns.png",
        mime="image/png",
        key="dl_cum_assets",
    )
    plt.close(fig_assets)

with tab_dd:
    st.markdown("**Portfolio drawdown**")
    fig = plot_drawdown(portfolio_returns)
    st.pyplot(fig, use_container_width=True)
    st.download_button(
        "⬇️ Download drawdown chart (PNG)",
        data=_fig_to_png_bytes(fig),
        file_name="drawdown.png",
        mime="image/png",
        key="dl_dd_portfolio",
    )
    plt.close(fig)

    st.markdown("**Asset-level drawdowns**")
    asset_dd = calculate_asset_drawdowns(asset_returns)
    fig_assets_dd = plot_asset_drawdowns(asset_dd)
    st.pyplot(fig_assets_dd, use_container_width=True)
    st.download_button(
        "⬇️ Download asset drawdowns (PNG)",
        data=_fig_to_png_bytes(fig_assets_dd),
        file_name="asset_drawdowns.png",
        mime="image/png",
        key="dl_dd_assets",
    )
    plt.close(fig_assets_dd)

with tab_data:
    st.markdown("**Prices**")
    st.dataframe(prices.tail(10), use_container_width=True)
    st.download_button(
        "⬇️ Download prices.csv",
        data=_df_to_csv_bytes(prices),
        file_name="price_data.csv",
        mime="text/csv",
    )

    st.markdown("**Core asset returns (Simple)**")
    st.dataframe(asset_returns.tail(10), use_container_width=True)
    st.download_button(
        "⬇️ Download core_asset_returns_simple.csv",
        data=_df_to_csv_bytes(asset_returns),
        file_name="core_asset_returns_simple.csv",
        mime="text/csv",
    )

    st.markdown("**Core portfolio returns & value (Simple)**")
    pv_df = pd.DataFrame(
        {"portfolio_return": portfolio_returns, "portfolio_value": portfolio_value}
    )
    st.dataframe(pv_df.tail(10), use_container_width=True)
    st.download_button(
        "⬇️ Download core_portfolio_returns_simple.csv",
        data=_df_to_csv_bytes(pv_df),
        file_name="core_portfolio_returns_simple.csv",
        mime="text/csv",
    )

    if diagnostic_return_method == "log":
        st.markdown("**Advanced diagnostic returns (Log)**")
        diagnostic_df = diagnostic_asset_returns.copy()
        diagnostic_df["PORTFOLIO"] = diagnostic_portfolio_returns
        st.dataframe(diagnostic_df.tail(10), use_container_width=True)
        st.download_button(
            "⬇️ Download diagnostic_returns_log.csv",
            data=_df_to_csv_bytes(diagnostic_df),
            file_name="diagnostic_returns_log.csv",
            mime="text/csv",
        )


# ─── Tab: Correlation & Diversification ───────────────────────────────────

with tab_corr:
    st.header("🧩 Correlation & Diversification")
    st.caption(
        "Pearson / Spearman correlations and how the average pairwise "
        "correlation evolves over time (diversification decay). This tab "
        "uses the core simple-return series."
    )

    if asset_returns.shape[1] < 2:
        st.info("Add at least two assets in the portfolio to compute correlations.")
    else:
        cc1, cc2 = st.columns(2)
        with cc1:
            corr_method = st.selectbox(
                "Correlation method",
                ["pearson", "spearman"],
                format_func=lambda x: x.title(),
                key="corr_method",
            )
        with cc2:
            corr_window = st.selectbox(
                "Rolling window (days)", [30, 60, 90, 180], index=2, key="corr_window"
            )

        corr_matrix = calculate_correlation_matrix(asset_returns, method=corr_method)
        st.markdown("#### Correlation matrix")
        st.dataframe(corr_matrix, use_container_width=True)

        # ── Diversification headline metrics (Phase 7) ────────────────────
        n_corr = corr_matrix.shape[0]
        off_diag_mean = float(
            (corr_matrix.to_numpy().sum() - n_corr) / (n_corr * (n_corr - 1))
        )
        met1, met2, met3 = st.columns(3)
        met1.metric("Avg pairwise correlation", f"{off_diag_mean:.3f}")
        try:
            weighted_corr = calculate_weighted_average_correlation(
                corr_matrix, weights
            )
            met2.metric(
                "Portfolio-weighted avg correlation",
                f"{weighted_corr:.3f}",
                help=(
                    "Pairwise correlations weighted by the product of the "
                    "portfolio weights — the correlation your portfolio "
                    "actually experiences."
                ),
            )
        except ValueError:
            met2.metric("Portfolio-weighted avg correlation", "N/A")
        try:
            stress_corr = calculate_stress_vs_normal_correlation(
                asset_returns, portfolio_returns, stress_quantile=0.10
            )
            met3.metric(
                "Stress-day avg correlation",
                f"{stress_corr['stress_avg_corr']:.3f}",
                delta=(
                    f"{stress_corr['stress_avg_corr'] - stress_corr['normal_avg_corr']:+.3f}"
                    " vs normal days"
                ),
                delta_color="inverse",
                help=(
                    f"Average pairwise correlation on the worst 10% of "
                    f"portfolio days ({stress_corr['n_stress_days']} days, "
                    f"portfolio return ≤ "
                    f"{stress_corr['stress_threshold'] * 100:.2f}%) vs the "
                    f"remaining {stress_corr['n_normal_days']} days."
                ),
            )
            if stress_corr["stress_avg_corr"] > stress_corr["normal_avg_corr"]:
                st.caption(
                    "⚠️ Correlations are **higher on stress days** — "
                    "diversification weakens exactly when it is needed most. "
                    "Scenario-based CVaR (Optimizer tab) accounts for this "
                    "better than volatility-based measures."
                )
        except ValueError:
            met3.metric("Stress-day avg correlation", "N/A")

        fig_hm = plot_correlation_heatmap(
            corr_matrix, title=f"Asset Return Correlation ({corr_method.title()})"
        )
        st.pyplot(fig_hm, use_container_width=True)
        cdl1, cdl2 = st.columns(2)
        with cdl1:
            st.download_button(
                "⬇️ Download heatmap (PNG)",
                data=_fig_to_png_bytes(fig_hm),
                file_name="correlation_heatmap.png",
                mime="image/png",
                key="dl_corr_hm",
            )
        with cdl2:
            st.download_button(
                "⬇️ Download correlation_matrix.csv",
                data=_df_to_csv_bytes(corr_matrix),
                file_name="correlation_matrix.csv",
                mime="text/csv",
                key="dl_corr_csv",
            )
        plt.close(fig_hm)

        st.markdown("#### Rolling average pairwise correlation")
        st.caption(
            f"⏱ Rolling correlation **lags by construction**: each point "
            f"averages the past {corr_window} days, so a regime change "
            f"only shows up gradually as new days roll into the window. "
            f"Shorter windows react faster but are noisier."
        )
        if len(asset_returns.dropna()) >= int(corr_window):
            rolling_corr = calculate_rolling_average_correlation(
                asset_returns, window=int(corr_window)
            )
            fig_rc = plot_rolling_average_correlation(
                rolling_corr,
                title=f"Rolling Average Pairwise Correlation ({corr_window}d window)",
            )
            st.pyplot(fig_rc, use_container_width=True)
            st.download_button(
                "⬇️ Download rolling correlation (PNG)",
                data=_fig_to_png_bytes(fig_rc),
                file_name="rolling_average_correlation.png",
                mime="image/png",
                key="dl_corr_roll",
            )
            plt.close(fig_rc)
        else:
            st.info(
                f"Not enough observations for a {corr_window}-day rolling window."
            )


# ─── Tab: Robust Assumptions Engine (Phase 7) ─────────────────────────────

with tab_assumptions:
    st.header("🧠 Robust Assumptions Engine")
    st.caption(
        "Build, inspect, and govern the assumptions that feed the optimizer: "
        "expected returns (raw vs robust vs manual views), volatility, and "
        "covariance. The Optimizer tab can consume this recipe directly — "
        "select **Robust Assumptions Engine** as its expected-return "
        "estimator."
    )

    # ── Scenario basis ────────────────────────────────────────────────────
    st.markdown("#### Scenario basis")
    ra_c1, ra_c2, ra_c3, ra_c4 = st.columns(4)
    with ra_c1:
        ra_source = st.selectbox(
            "Scenario source",
            ["historical", "normal_mc", "student_t_mc"],
            format_func=lambda x: {
                "historical": "Historical",
                "normal_mc": "Normal Monte Carlo",
                "student_t_mc": "Student-t Monte Carlo",
            }[x],
            key="ra_source",
        )
    with ra_c2:
        ra_horizon = st.number_input(
            "Horizon (days)",
            min_value=1,
            max_value=60,
            value=int(horizon_days),
            step=1,
            key="ra_horizon",
        )
    with ra_c3:
        ra_n_scenarios = st.number_input(
            "MC scenarios",
            min_value=500,
            max_value=50_000,
            value=5000,
            step=500,
            key="ra_n_scenarios",
            disabled=(ra_source == "historical"),
        )
    with ra_c4:
        ra_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2**31 - 1,
            value=42,
            step=1,
            key="ra_seed",
            disabled=(ra_source == "historical"),
        )
    ra_student_df = st.number_input(
        "Student-t df",
        min_value=3,
        max_value=30,
        value=5,
        step=1,
        key="ra_student_df",
        disabled=(ra_source != "student_t_mc"),
    )
    st.caption(
        f"All expected returns below are **per {int(ra_horizon)}-day "
        f"horizon** (the return over one optimization period), not daily "
        "and not annualized."
    )

    # ── Expected-return recipe ────────────────────────────────────────────
    st.markdown("#### Expected-return estimators")
    ra_e1, ra_e2, ra_e3, ra_e4 = st.columns(4)
    with ra_e1:
        ra_final_method = st.selectbox(
            "Final estimator (used by optimizer)",
            [
                "mean",
                "median",
                "trimmed_mean",
                "winsorized_mean",
                "shrinkage_to_zero",
                "zero",
            ],
            format_func=lambda x: {
                "mean": "Historical mean",
                "median": "Historical median",
                "trimmed_mean": "Trimmed mean",
                "winsorized_mean": "Winsorized mean",
                "shrinkage_to_zero": "Shrinkage to zero",
                "zero": "Zero (pure tail-risk)",
            }[x],
            key="ra_final_method",
        )
    with ra_e2:
        ra_trim = st.slider(
            "Trim proportion (each tail)",
            min_value=0.0,
            max_value=0.25,
            value=0.10,
            step=0.01,
            key="ra_trim",
        )
    with ra_e3:
        ra_winsor = st.slider(
            "Winsor proportion (each tail)",
            min_value=0.0,
            max_value=0.25,
            value=0.05,
            step=0.01,
            key="ra_winsor",
        )
    with ra_e4:
        ra_shrink_w = st.slider(
            "Shrinkage weight on mean",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="ra_shrink_w",
            help="E[r] = weight × historical mean; the rest shrinks to zero.",
        )

    if ra_final_method == "zero":
        st.warning(
            "Zero expected returns make **return-based objectives** "
            "(Max Return, Max Sharpe, Target Return) meaningless — only "
            "pure tail-risk objectives (Min CVaR) remain interpretable."
        )

    with st.expander("🧭 Manual expected-return views (optional)", expanded=False):
        st.caption(
            "Point views per asset, expressed **per horizon**. "
            "Final E[r] = blend × view + (1 − blend) × base estimate. "
            "This seam is where Black-Litterman / Entropy-Pooling plug in "
            "later."
        )
        ra_use_views = st.checkbox("Enable views", value=False, key="ra_use_views")
        ra_view_blend = st.slider(
            "View blend weight",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            key="ra_view_blend",
            disabled=not ra_use_views,
        )
        ra_views: dict[str, float] = {}
        ra_view_cols = st.columns(max(1, len(asset_returns.columns)))
        for i, asset in enumerate(asset_returns.columns):
            with ra_view_cols[i % len(ra_view_cols)]:
                ra_views[asset] = st.number_input(
                    f"{asset} E[r]/horizon",
                    value=0.0,
                    step=0.001,
                    format="%.4f",
                    key=f"ra_view_{asset}",
                    disabled=not ra_use_views,
                )

    # ── Risk-assumption recipe ────────────────────────────────────────────
    st.markdown("#### Volatility & covariance estimators")
    ra_r1, ra_r2, ra_r3, ra_r4 = st.columns(4)
    with ra_r1:
        ra_cov_method = st.selectbox(
            "Covariance estimator",
            ["sample", "shrinkage", "ewma"],
            format_func=lambda x: {
                "sample": "Sample",
                "shrinkage": "Shrinkage (Ledoit-Wolf-style)",
                "ewma": "EWMA (RiskMetrics)",
            }[x],
            key="ra_cov_method",
        )
    with ra_r2:
        ra_shrink_delta = st.slider(
            "Covariance shrinkage δ",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            key="ra_shrink_delta",
            disabled=(ra_cov_method != "shrinkage"),
        )
    with ra_r3:
        ra_shrink_target = st.selectbox(
            "Shrinkage target",
            ["constant_correlation", "diagonal"],
            format_func=lambda x: {
                "constant_correlation": "Constant correlation",
                "diagonal": "Diagonal (zero correlation)",
            }[x],
            key="ra_shrink_target",
            disabled=(ra_cov_method != "shrinkage"),
        )
    with ra_r4:
        ra_lambda = st.slider(
            "EWMA decay λ",
            min_value=0.80,
            max_value=0.99,
            value=0.94,
            step=0.01,
            key="ra_lambda",
            disabled=(ra_cov_method != "ewma"),
        )

    run_assumptions = st.button(
        "🧮 Build assumptions",
        type="primary",
        use_container_width=True,
        key="run_assumptions",
    )

    if run_assumptions:
        try:
            ra_config = AssumptionConfig(
                expected_return_method=ra_final_method,
                trim_proportion=float(ra_trim),
                winsor_proportion=float(ra_winsor),
                shrinkage_weight=float(ra_shrink_w),
                manual_views=(dict(ra_views) if ra_use_views else {}),
                view_blend_weight=float(ra_view_blend),
                covariance_method=ra_cov_method,
                shrinkage_delta=float(ra_shrink_delta),
                shrinkage_target=ra_shrink_target,
                decay_lambda=float(ra_lambda),
            )
            ra_scenarios = _scenario_matrix_cached(
                asset_returns,
                ra_source,
                int(ra_n_scenarios),
                int(ra_horizon),
                float(ra_student_df),
                int(ra_seed),
            )
            ra_table = build_assumption_table(ra_scenarios, ra_config)
            ra_vol_table = build_volatility_table(
                asset_returns,
                horizon_days=int(ra_horizon),
                winsor_proportion=float(ra_winsor),
                decay_lambda=float(ra_lambda),
            )
            ra_cov = ra_config.covariance(asset_returns)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Assumption build failed: {exc}")
            st.session_state["assumptions_results"] = None
        else:
            st.session_state["assumptions_results"] = {
                "config": ra_config,
                "table": ra_table,
                "vol_table": ra_vol_table,
                "covariance": ra_cov,
                "source": ra_source,
                "horizon_days": int(ra_horizon),
                "n_scenarios": int(ra_scenarios.shape[0]),
                "assets": list(ra_scenarios.columns),
            }

    ra_state = st.session_state.get("assumptions_results")
    if ra_state is None:
        st.info(
            "Configure the recipe above and click **Build assumptions** to "
            "see every estimate side by side."
        )
    else:
        ra_cfg: AssumptionConfig = ra_state["config"]
        h_used = ra_state["horizon_days"]
        st.success(
            f"Assumptions built from **{ra_state['source']}** scenarios "
            f"({ra_state['n_scenarios']:,} × {len(ra_state['assets'])} "
            f"assets, {h_used}-day horizon). Final estimator: "
            f"**{ra_cfg.expected_return_method}**"
            + (
                f" + manual views (blend {ra_cfg.view_blend_weight:.1f})"
                if ra_cfg.manual_views
                else ""
            )
            + "."
        )

        st.markdown(f"### Expected returns per asset (per {h_used}-day horizon)")
        display_table = ra_state["table"].copy() * 100.0
        display_table.columns = [
            "Mean (%)",
            "Median (%)",
            "Trimmed Mean (%)",
            "Winsorized Mean (%)",
            "Shrinkage (%)",
            "Manual View (%)",
            "Final E[r] (%)",
        ]
        st.dataframe(
            display_table.round(4).reset_index(names="Asset"),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "**Final E[r]** is exactly what the optimizer receives when its "
            "estimator is set to *Robust Assumptions Engine*. Large gaps "
            "between mean and median/trimmed columns flag assets whose "
            "average is driven by a few extreme days."
        )
        st.download_button(
            "⬇️ Download expected_return_assumptions.csv",
            data=_df_to_csv_bytes(ra_state["table"]),
            file_name="expected_return_assumptions.csv",
            mime="text/csv",
            key="dl_ra_mu",
        )

        st.markdown("### Volatility per asset")
        vol_display = ra_state["vol_table"].copy() * 100.0
        vol_display.columns = [
            "Daily Vol (%)",
            "Winsorized Daily Vol (%)",
            "EWMA Daily Vol (%)",
            f"{h_used}-day Vol (√t) (%)",
            f"{h_used}-day EWMA Vol (√t) (%)",
            "Annualized Vol (%)",
        ]
        st.dataframe(
            vol_display.round(2).reset_index(names="Asset"),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Volatility estimated from **daily** returns; horizon columns "
            "use √t scaling (i.i.d. approximation). EWMA (RiskMetrics, "
            f"λ={ra_cfg.decay_lambda:.2f}) reacts faster to recent regime "
            "changes; winsorized dampens single-day outliers."
        )
        st.download_button(
            "⬇️ Download volatility_assumptions.csv",
            data=_df_to_csv_bytes(ra_state["vol_table"]),
            file_name="volatility_assumptions.csv",
            mime="text/csv",
            key="dl_ra_vol",
        )

        st.markdown(f"### Covariance ({ra_cfg.covariance_method}, daily)")
        cov_used: pd.DataFrame = ra_state["covariance"]
        cov_sd = np.sqrt(pd.Series(np.diag(cov_used), index=cov_used.index))
        implied_corr = cov_used / np.outer(cov_sd, cov_sd)
        cov_col1, cov_col2 = st.columns(2)
        with cov_col1:
            st.markdown("**Covariance matrix (daily)**")
            st.dataframe((cov_used * 1e4).round(3), use_container_width=True)
            st.caption("Values ×10⁻⁴ for readability.")
        with cov_col2:
            fig_ic = plot_correlation_heatmap(
                implied_corr,
                title=f"Implied Correlation — {ra_cfg.covariance_method}",
            )
            st.pyplot(fig_ic, use_container_width=True)
            plt.close(fig_ic)
        st.download_button(
            "⬇️ Download covariance_assumptions.csv",
            data=_df_to_csv_bytes(cov_used),
            file_name="covariance_assumptions.csv",
            mime="text/csv",
            key="dl_ra_cov",
        )
        if ra_cfg.covariance_method != "sample":
            st.caption(
                "The Optimizer tab can generate its Monte Carlo scenarios "
                "from this robust covariance — enable **Use robust "
                "covariance** there."
            )

        with st.expander("📖 How the final expected return is built", expanded=False):
            st.markdown(
                """
1. **Base estimator** — applied per asset to the scenario matrix:
   mean, median, trimmed mean (cut *p* from each tail), winsorized mean
   (clip at the [*p*, 1−*p*] quantiles), shrinkage (weight × mean, rest
   toward zero), or zero.
2. **Manual views** — if enabled:
   `final = blend × view + (1 − blend) × base` per asset with a view.
3. The result is the **Final E[r]** column and is what the optimizer
   maximizes / constrains against, per optimization horizon.

Extensions that plug into this same seam later: Black-Litterman,
Meucci Entropy Pooling, scenario-probability reweighting, and
regime-conditional estimators.
"""
            )


# ─── Tab 6: Backtesting & Model Validation ────────────────────────────────

with tab_backtest:
    st.header("🔬 VaR Backtesting & Model Validation")
    st.caption(
        "Horizon-aware rolling VaR forecasts vs horizon-matched realised returns. "
        "Kupiec POF, Christoffersen Independence, and Conditional Coverage tests."
    )
    st.info(
        "Backtesting compares horizon-matched realised returns against "
        "horizon-matched VaR forecasts. Select **Horizon (days) = 1** for "
        "the classic one-step-ahead test."
    )

    bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
    with bt_col1:
        bt_method = st.selectbox(
            "VaR method",
            ["historical", "gaussian", "cornish_fisher", "compare_all"],
            format_func=lambda x: {
                "historical": "Historical",
                "gaussian": "Gaussian",
                "cornish_fisher": "Cornish-Fisher",
                "compare_all": "⚖️ Compare All",
            }[x],
            key="bt_method",
        )
    with bt_col2:
        bt_confidence = st.selectbox(
            "Confidence level",
            [0.90, 0.95, 0.975, 0.99],
            index=1,
            format_func=lambda x: f"{x * 100:.1f}%",
            key="bt_confidence",
        )
    with bt_col3:
        bt_window = st.selectbox(
            "Rolling window (days)",
            [60, 126, 252, 500],
            index=2,
            key="bt_window",
        )
    with bt_col4:
        bt_horizon = st.number_input(
            "Horizon (days)",
            min_value=1,
            max_value=60,
            value=int(horizon_days),
            step=1,
            key="bt_horizon",
            help=(
                "1 = one-step-ahead daily backtest. >1 compares realised "
                "h-day returns against h-day VaR forecasts."
            ),
        )

    bt_mode = st.radio(
        "Backtesting mode",
        ["overlapping", "non_overlapping"],
        format_func=lambda x: {
            "overlapping": "Overlapping rolling",
            "non_overlapping": "Non-overlapping horizon",
        }[x],
        horizontal=True,
        key="bt_mode",
    )
    st.caption(
        "Overlapping mode uses daily rolling horizon returns. Non-overlapping "
        "mode uses independent horizon blocks and is more appropriate for "
        "independence-based tests such as Christoffersen."
    )

    with st.expander("⚙️ Test period (optional)", expanded=False):
        bt_period = st.radio(
            "Backtest on",
            ["Full period", "Last 1 year", "Last 2 years", "Custom range"],
            horizontal=True,
            key="bt_period",
        )
        bt_start: date | None = None
        bt_end: date | None = None
        if bt_period == "Custom range":
            min_dt = portfolio_returns.index.min().date()
            max_dt = portfolio_returns.index.max().date()
            bt_start = st.date_input(
                "From", value=min_dt, min_value=min_dt, max_value=max_dt,
                key="bt_start",
            )
            bt_end = st.date_input(
                "To", value=max_dt, min_value=min_dt, max_value=max_dt,
                key="bt_end",
            )

    run_backtest = st.button(
        "▶️ Run Backtest",
        type="primary",
        use_container_width=True,
        key="run_backtest",
    )

    # ── On click: validate, compute, store in session_state ──────────────
    if run_backtest:
        if portfolio_returns is None or len(portfolio_returns) == 0:
            st.error(
                "No portfolio returns available. "
                "Click **Run risk analysis** in the sidebar first."
            )
            st.session_state["backtest_results"] = None
        else:
            sliced_returns = portfolio_returns
            if bt_period == "Last 1 year":
                cutoff = portfolio_returns.index.max() - pd.Timedelta(days=365)
                sliced_returns = portfolio_returns.loc[
                    portfolio_returns.index >= cutoff
                ]
            elif bt_period == "Last 2 years":
                cutoff = portfolio_returns.index.max() - pd.Timedelta(days=730)
                sliced_returns = portfolio_returns.loc[
                    portfolio_returns.index >= cutoff
                ]
            elif bt_period == "Custom range" and bt_start and bt_end:
                mask = (portfolio_returns.index.date >= bt_start) & (
                    portfolio_returns.index.date <= bt_end
                )
                sliced_returns = portfolio_returns.loc[mask]

            if bt_window < int(bt_horizon):
                st.error(
                    f"Rolling window {bt_window} is smaller than the horizon "
                    f"{bt_horizon}. Pick a larger window or a smaller horizon."
                )
                st.session_state["backtest_results"] = None
            elif len(sliced_returns) <= bt_window + int(bt_horizon):
                st.error(
                    f"Selected test period has {len(sliced_returns):,} observations, "
                    f"but window + horizon = {bt_window + int(bt_horizon)}. "
                    "Choose a longer test period, smaller window, smaller horizon, "
                    "or a wider date range in the sidebar."
                )
                st.session_state["backtest_results"] = None
            elif bt_method == "compare_all":
                try:
                    forecasts_by_method, comparison_df = compare_var_models_backtest(
                        sliced_returns,
                        methods=["historical", "gaussian", "cornish_fisher"],
                        confidence_level=bt_confidence,
                        window=bt_window,
                        horizon_days=int(bt_horizon),
                        backtest_mode=bt_mode,
                    )
                except (ValueError, RuntimeError) as exc:
                    st.error(f"Backtest failed: {exc}")
                    st.session_state["backtest_results"] = None
                else:
                    st.session_state["backtest_results"] = {
                        "mode": "compare_all",
                        "forecasts_by_method": forecasts_by_method,
                        "comparison_df": comparison_df,
                        "confidence_level": bt_confidence,
                        "window": bt_window,
                        "horizon_days": int(bt_horizon),
                        "backtest_mode": bt_mode,
                    }
            else:
                try:
                    forecast_df, result = backtest_var_model(
                        sliced_returns,
                        method=bt_method,
                        confidence_level=bt_confidence,
                        window=bt_window,
                        horizon_days=int(bt_horizon),
                        backtest_mode=bt_mode,
                    )
                except (ValueError, RuntimeError) as exc:
                    st.error(f"Backtest failed: {exc}")
                    st.session_state["backtest_results"] = None
                else:
                    st.session_state["backtest_results"] = {
                        "mode": "single",
                        "method": bt_method,
                        "forecast_df": forecast_df,
                        "result": result,
                        "confidence_level": bt_confidence,
                        "window": bt_window,
                        "horizon_days": int(bt_horizon),
                        "backtest_mode": bt_mode,
                    }

    # ── Render whatever is in session_state (persists across reruns) ─────
    bt_state = st.session_state.get("backtest_results")

    if bt_state is None:
        st.info("Configure backtest parameters and click **Run Backtest**.")
    elif bt_state["mode"] == "compare_all":
        forecasts_by_method = bt_state["forecasts_by_method"]
        comparison_df = bt_state["comparison_df"]
        confidence_used = bt_state["confidence_level"]
        horizon_used = bt_state.get("horizon_days", 1)

        st.markdown(
            f"**{horizon_used}-day VaR Backtest — confidence "
            f"{confidence_used * 100:.0f}%, window {bt_state['window']} days**"
        )

        report_table = create_backtesting_report_table(comparison_df)

        def _color_traffic_light(value: str) -> str:
            if value == "Green":
                return "background-color: #d4edda; color: #155724;"
            if value == "Yellow":
                return "background-color: #fff3cd; color: #856404;"
            if value == "Red":
                return "background-color: #f8d7da; color: #721c24;"
            return ""

        styled = report_table.style.applymap(
            _color_traffic_light, subset=["Traffic Light"]
        )
        st.dataframe(styled, use_container_width=True)

        fig_cmp = plot_model_comparison_backtest(comparison_df)
        st.pyplot(fig_cmp, use_container_width=True)
        st.download_button(
            "⬇️ Download comparison chart (PNG)",
            data=_fig_to_png_bytes(fig_cmp),
            file_name="model_comparison_backtest.png",
            mime="image/png",
        )
        plt.close(fig_cmp)

        st.download_button(
            "⬇️ Download model_comparison.csv",
            data=_df_to_csv_bytes(comparison_df, include_index=False),
            file_name="model_comparison.csv",
            mime="text/csv",
        )

        for method_name in ["historical", "gaussian", "cornish_fisher"]:
            label = METHOD_LABELS[method_name]
            with st.expander(f"📊 {label} detail", expanded=False):
                if method_name in forecasts_by_method:
                    fc_df = forecasts_by_method[method_name]
                    fig_a = plot_var_backtest(fc_df, method_name, confidence_used)
                    st.pyplot(fig_a, use_container_width=True)
                    plt.close(fig_a)

                    fig_b = plot_breach_timeline(fc_df, method_name)
                    st.pyplot(fig_b, use_container_width=True)
                    plt.close(fig_b)

                    rbw_cmp = min(100, max(2, len(fc_df) // 2))
                    fig_rate_cmp = plot_rolling_breach_rate(
                        calculate_rolling_breach_rate(fc_df, window=rbw_cmp),
                        expected_breach_rate=1.0 - confidence_used,
                        method=method_name,
                    )
                    st.pyplot(fig_rate_cmp, use_container_width=True)
                    plt.close(fig_rate_cmp)

                    st.download_button(
                        f"⬇️ Download var_forecasts_{method_name}.csv",
                        data=_df_to_csv_bytes(fc_df),
                        file_name=f"var_forecasts_{method_name}.csv",
                        mime="text/csv",
                        key=f"dl_fc_{method_name}",
                    )
                else:
                    st.warning(
                        "This method failed during the backtest — see "
                        "the error column in the comparison table above."
                    )
    else:  # single-method mode
        forecast_df = bt_state["forecast_df"]
        result = bt_state["result"]
        method_used = bt_state["method"]
        confidence_used = bt_state["confidence_level"]
        horizon_used = bt_state.get("horizon_days", 1)

        st.markdown(
            f"**{horizon_used}-day VaR Backtest — "
            f"{METHOD_LABELS[method_used]}, "
            f"confidence {confidence_used * 100:.0f}%, "
            f"window {bt_state['window']} days**"
        )

        row1 = st.columns(4)
        row1[0].metric("Observations", f"{result['observations']:,}")
        row1[1].metric("Actual Breaches", f"{result['actual_breaches']:,}")
        row1[2].metric("Expected Breaches", f"{result['expected_breaches']:.1f}")
        row1[3].metric(
            "Actual Breach Rate",
            f"{result['actual_breach_rate'] * 100:.2f}%",
        )

        row2 = st.columns(4)

        def _fmt_p(value: float) -> str:
            return (
                f"{value:.4f}"
                if value is not None and pd.notna(value)
                else "N/A"
            )

        row2[0].metric("Kupiec p-value", _fmt_p(result["kupiec_p_value"]))
        row2[1].metric(
            "Christoffersen p-value", _fmt_p(result["christoffersen_p_value"])
        )
        row2[2].metric("CC p-value", _fmt_p(result["cc_p_value"]))

        with row2[3]:
            status = result["traffic_light"]
            if status == "Green":
                st.success("🟢 Green — Breach Count Within Threshold")
            elif status == "Yellow":
                st.warning("🟡 Yellow — Review Breach Count")
            elif status == "Red":
                st.error("🔴 Red — Breach Count Outside Threshold")
            else:
                st.info(f"Status: {status}")

        st.caption(result["interpretation"])

        if result["christoffersen_pass"] is None:
            st.warning(result["christoffersen_interpretation"])
        if result["cc_pass"] is None:
            st.warning(result["cc_interpretation"])

        chart_a, chart_b = st.tabs(["📈 Backtest Chart", "📅 Breach Timeline"])
        with chart_a:
            fig_a = plot_var_backtest(forecast_df, method_used, confidence_used)
            st.pyplot(fig_a, use_container_width=True)
            st.download_button(
                "⬇️ Download backtest chart (PNG)",
                data=_fig_to_png_bytes(fig_a),
                file_name=f"var_backtesting_exceptions_{method_used}.png",
                mime="image/png",
            )
            plt.close(fig_a)
        with chart_b:
            fig_b = plot_breach_timeline(forecast_df, method_used)
            st.pyplot(fig_b, use_container_width=True)
            st.download_button(
                "⬇️ Download breach timeline (PNG)",
                data=_fig_to_png_bytes(fig_b),
                file_name=f"breach_timeline_{method_used}.png",
                mime="image/png",
            )
            plt.close(fig_b)

        with st.expander("📋 Forecast Data", expanded=False):
            st.dataframe(forecast_df.tail(50), use_container_width=True)

        # ── Rolling breach rate over time ────────────────────────────────
        st.markdown("**📉 Rolling breach rate**")
        rbw = min(100, max(2, len(forecast_df) // 2))
        rolling_rate = calculate_rolling_breach_rate(forecast_df, window=rbw)
        fig_rate = plot_rolling_breach_rate(
            rolling_rate,
            expected_breach_rate=result["expected_breach_rate"],
            method=method_used,
        )
        st.pyplot(fig_rate, use_container_width=True)
        plt.close(fig_rate)

        # ── Worst realised horizon losses ────────────────────────────────
        st.markdown("**🔻 Worst realised horizon losses**")
        worst_losses = get_worst_realized_losses(forecast_df, n=10)
        st.dataframe(worst_losses, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download worst_realised_losses.csv",
            data=_df_to_csv_bytes(worst_losses, include_index=False),
            file_name="worst_realised_losses.csv",
            mime="text/csv",
            key="dl_worst_losses",
        )

        # ── Breach summary by year ───────────────────────────────────────
        st.markdown("**📅 Breach summary by year**")
        try:
            by_year = summarize_backtest_by_period(
                forecast_df, confidence_level=confidence_used, freq="Y"
            )
            st.dataframe(by_year, use_container_width=True, hide_index=True)
        except ValueError as exc:
            st.caption(f"Per-period summary unavailable: {exc}")

        dl_a, dl_b = st.columns(2)
        with dl_a:
            st.download_button(
                "⬇️ Download Forecast CSV",
                data=_df_to_csv_bytes(forecast_df),
                file_name=f"var_forecasts_{method_used}.csv",
                mime="text/csv",
            )
        with dl_b:
            import json  # noqa: PLC0415

            result_json = json.dumps(
                {
                    k: (None if isinstance(v, float) and pd.isna(v) else v)
                    for k, v in result.items()
                },
                indent=2,
                default=str,
            )
            st.download_button(
                "⬇️ Download Backtest Results JSON",
                data=result_json.encode("utf-8"),
                file_name=f"backtesting_results_{method_used}.json",
                mime="application/json",
            )


# ─── Tab 7: Monte Carlo Scenario Engine ───────────────────────────────────

with tab_mc:
    st.header("🎲 Monte Carlo Scenario Engine")
    st.caption(
        "Monte Carlo VaR/CVaR estimates risk from simulated portfolio return "
        "scenarios rather than only historical observations."
    )

    mc_col1, mc_col2, mc_col3 = st.columns(3)
    with mc_col1:
        mc_distribution = st.selectbox(
            "Distribution",
            ["normal", "student_t", "compare"],
            format_func=lambda x: {
                "normal": "Normal",
                "student_t": "Student-t",
                "compare": "⚖️ Compare Normal vs Student-t",
            }[x],
            key="mc_distribution",
        )
    with mc_col2:
        mc_n_scenarios = st.number_input(
            "Number of scenarios",
            min_value=1000,
            max_value=100_000,
            value=5000,
            step=1000,
            key="mc_n_scenarios",
        )
    with mc_col3:
        mc_horizon = st.number_input(
            "Horizon (days)",
            min_value=1,
            max_value=60,
            value=int(horizon_days),
            step=1,
            key="mc_horizon",
        )

    mc_col4, mc_col5, mc_col6 = st.columns(3)
    with mc_col4:
        mc_confidence = st.selectbox(
            "Confidence level",
            [0.90, 0.95, 0.975, 0.99],
            index=[0.90, 0.95, 0.975, 0.99].index(
                confidence_level
                if confidence_level in (0.90, 0.95, 0.975, 0.99)
                else 0.95
            ),
            format_func=lambda x: f"{x * 100:.1f}%",
            key="mc_confidence",
        )
    with mc_col5:
        mc_df = st.number_input(
            "Student-t df",
            min_value=3,
            max_value=30,
            value=5,
            step=1,
            key="mc_df",
        )
    with mc_col6:
        mc_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2**31 - 1,
            value=42,
            step=1,
            key="mc_seed",
        )

    mc_col7, mc_col8 = st.columns(2)
    with mc_col7:
        mc_n_paths = st.number_input(
            "Path simulations (n_paths)",
            min_value=10,
            max_value=10_000,
            value=500,
            step=50,
            key="mc_n_paths",
        )
    with mc_col8:
        mc_path_horizon = st.number_input(
            "Path horizon (days)",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            key="mc_path_horizon",
        )

    run_mc = st.button(
        "▶️ Run Monte Carlo",
        type="primary",
        use_container_width=True,
        key="run_mc",
    )

    if run_mc:
        try:
            # Shared cached scenario builder — the Optimizer tab with the
            # same (source, n, horizon, df, seed) reuses these exact
            # matrices instead of regenerating them.
            normal_scen = _scenario_matrix_cached(
                asset_returns,
                "normal_mc",
                int(mc_n_scenarios),
                int(mc_horizon),
                float(mc_df),
                int(mc_seed),
            )
            student_scen = _scenario_matrix_cached(
                asset_returns,
                "student_t_mc",
                int(mc_n_scenarios),
                int(mc_horizon),
                float(mc_df),
                int(mc_seed),
            )
            normal_pf = calculate_portfolio_scenario_returns(normal_scen, weights)
            student_pf = calculate_portfolio_scenario_returns(student_scen, weights)

            pf_mean = float(portfolio_returns.mean())
            pf_vol = float(portfolio_returns.std(ddof=1))
            paths_distribution = (
                "normal" if mc_distribution == "normal" else "student_t"
            )
            paths = simulate_portfolio_paths(
                portfolio_daily_mean=pf_mean,
                portfolio_daily_volatility=pf_vol,
                initial_value=float(initial_capital),
                n_paths=int(mc_n_paths),
                horizon_days=int(mc_path_horizon),
                distribution=paths_distribution,
                df=float(mc_df),
                random_seed=int(mc_seed),
                return_method="simple",
            )

            comparison_all = compare_all_risk_methods(
                portfolio_returns=portfolio_returns,
                asset_returns=asset_returns,
                weights=weights,
                confidence_level=float(mc_confidence),
                horizon_days=int(mc_horizon),
                n_scenarios=int(mc_n_scenarios),
                student_t_df=float(mc_df),
                random_seed=int(mc_seed),
                return_method="simple",
            )
        except (ValueError, RuntimeError) as exc:
            st.error(f"Monte Carlo failed: {exc}")
            st.session_state["mc_results"] = None
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected Monte Carlo error: {exc}")
            st.session_state["mc_results"] = None
        else:
            st.session_state["mc_results"] = {
                "distribution": mc_distribution,
                "horizon_days": int(mc_horizon),
                "confidence_level": float(mc_confidence),
                "n_scenarios": int(mc_n_scenarios),
                "student_t_df": float(mc_df),
                "random_seed": int(mc_seed),
                "normal_pf": normal_pf,
                "student_pf": student_pf,
                "paths": paths,
                "comparison_all": comparison_all,
            }

    mc_state = st.session_state.get("mc_results")
    if mc_state is None:
        st.info("Configure Monte Carlo parameters and click **Run Monte Carlo**.")
    else:
        dist = mc_state["distribution"]
        normal_pf = mc_state["normal_pf"]
        student_pf = mc_state["student_pf"]
        paths = mc_state["paths"]
        comparison_all = mc_state["comparison_all"]
        mc_horizon_used = mc_state["horizon_days"]
        mc_conf_used = mc_state["confidence_level"]

        if dist == "student_t":
            pf_selected = student_pf
            label = "Student-t"
        else:
            pf_selected = normal_pf
            label = "Normal"

        primary_var = scenario_var(pf_selected, mc_conf_used)
        primary_cvar = scenario_cvar(pf_selected, mc_conf_used)
        money_var = loss_value_to_money(primary_var, initial_capital)
        money_cvar = loss_value_to_money(primary_cvar, initial_capital)

        m_row = st.columns(5)
        m_row[0].metric(
            f"{label} MC VaR ({mc_horizon_used}d)",
            f"{primary_var * 100:.2f}%",
            delta=_format_money(money_var),
            delta_color="off",
        )
        m_row[1].metric(
            f"{label} MC CVaR ({mc_horizon_used}d)",
            f"{primary_cvar * 100:.2f}%",
            delta=_format_money(money_cvar),
            delta_color="off",
        )
        m_row[2].metric(
            "Mean simulated return",
            f"{float(pf_selected.mean()) * 100:.2f}%",
        )
        m_row[3].metric(
            "Worst simulated return",
            f"{float(pf_selected.min()) * 100:.2f}%",
        )
        m_row[4].metric("Scenarios", f"{mc_state['n_scenarios']:,}")

        st.markdown("### Scenario distribution")
        fig_dist = plot_mc_loss_distribution(
            pf_selected,
            var_value=primary_var,
            cvar_value=primary_cvar,
            title=(
                f"{label} MC — {mc_horizon_used}-day Portfolio Return Distribution "
                f"({mc_state['n_scenarios']:,} scenarios)"
            ),
        )
        st.pyplot(fig_dist, use_container_width=True)
        dist_file_tag = "compare" if dist == "compare" else dist
        st.download_button(
            "⬇️ Download MC distribution chart (PNG)",
            data=_fig_to_png_bytes(fig_dist),
            file_name=f"mc_loss_distribution_{dist_file_tag}.png",
            mime="image/png",
            key="dl_mc_dist",
        )
        plt.close(fig_dist)

        if dist == "compare":
            st.markdown("### Normal vs Student-t comparison")
            fig_cmp_dist = plot_normal_vs_student_t_distribution(normal_pf, student_pf)
            st.pyplot(fig_cmp_dist, use_container_width=True)
            plt.close(fig_cmp_dist)

            cmp_rows = []
            for cmp_label, cmp_series in (
                ("Normal", normal_pf),
                ("Student-t", student_pf),
            ):
                cmp_rows.append(
                    {
                        "Distribution": cmp_label,
                        "VaR (%)": scenario_var(cmp_series, mc_conf_used) * 100.0,
                        "CVaR (%)": scenario_cvar(cmp_series, mc_conf_used) * 100.0,
                        "Mean (%)": float(cmp_series.mean()) * 100.0,
                        "Vol (%)": float(cmp_series.std(ddof=1)) * 100.0,
                        "Worst (%)": float(cmp_series.min()) * 100.0,
                        "Best (%)": float(cmp_series.max()) * 100.0,
                    }
                )
            st.dataframe(
                pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True
            )

        st.markdown("### Portfolio value paths")
        fig_paths = plot_mc_portfolio_paths(paths)
        st.pyplot(fig_paths, use_container_width=True)
        st.download_button(
            "⬇️ Download portfolio paths chart (PNG)",
            data=_fig_to_png_bytes(fig_paths),
            file_name="mc_portfolio_paths.png",
            mime="image/png",
            key="dl_mc_paths",
        )
        plt.close(fig_paths)

        st.markdown("### Method comparison (Historical / Gaussian / CF / MC)")
        st.caption(
            f"All rows are **{mc_horizon_used}-day** VaR/CVaR on the same "
            "basis: Historical / Gaussian / Cornish-Fisher use realised "
            "rolling horizon returns (no √t scaling); the MC rows use "
            "simulated h-day scenarios. These are directly comparable to "
            "each other, but not to the √t-scaled headline cards at the top "
            "of the page."
        )
        styled_cmp = comparison_all.copy()
        styled_cmp["VaR"] = styled_cmp["VaR"].apply(
            lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "N/A"
        )
        styled_cmp["CVaR"] = styled_cmp["CVaR"].apply(
            lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "N/A"
        )
        st.dataframe(styled_cmp, use_container_width=True, hide_index=True)

        fig_cmp = plot_var_cvar_method_comparison(comparison_all)
        st.pyplot(fig_cmp, use_container_width=True)
        st.download_button(
            "⬇️ Download method comparison chart (PNG)",
            data=_fig_to_png_bytes(fig_cmp),
            file_name="var_cvar_method_comparison.png",
            mime="image/png",
            key="dl_mc_cmp",
        )
        plt.close(fig_cmp)

        st.download_button(
            "⬇️ Download model_risk_comparison.csv",
            data=_df_to_csv_bytes(comparison_all, include_index=False),
            file_name="model_risk_comparison.csv",
            mime="text/csv",
            key="dl_mc_cmp_csv",
        )


# ─── Tab 8: Portfolio Optimization (Phase 5) ──────────────────────────────

with tab_opt:
    st.header("🎯 Portfolio Optimization")
    st.caption(
        "Scenario-based CVaR optimization using the Rockafellar-Uryasev "
        "linear programming formulation. Uses CVXPY under the hood."
    )

    # ── Scenario source controls ────────────────────────────────────────
    st.markdown("#### Scenario source")
    opt_c1, opt_c2, opt_c3, opt_c4 = st.columns(4)
    with opt_c1:
        opt_source = st.selectbox(
            "Source",
            ["historical", "normal_mc", "student_t_mc"],
            format_func=lambda x: {
                "historical": "Historical",
                "normal_mc": "Normal Monte Carlo",
                "student_t_mc": "Student-t Monte Carlo",
            }[x],
            key="opt_source",
        )
    with opt_c2:
        opt_horizon = st.number_input(
            "Horizon (days)",
            min_value=1,
            max_value=60,
            value=int(horizon_days),
            step=1,
            key="opt_horizon",
        )
    with opt_c3:
        opt_n_scenarios = st.number_input(
            "MC scenarios",
            min_value=500,
            max_value=50_000,
            value=5000,
            step=500,
            key="opt_n_scenarios",
            disabled=(opt_source == "historical"),
            help="Used only for Normal / Student-t Monte Carlo sources.",
        )
    with opt_c4:
        opt_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2**31 - 1,
            value=42,
            step=1,
            key="opt_seed",
        )

    opt_c5, opt_c6, opt_c7 = st.columns(3)
    with opt_c5:
        opt_student_df = st.number_input(
            "Student-t df",
            min_value=3,
            max_value=30,
            value=5,
            step=1,
            key="opt_student_df",
            disabled=(opt_source != "student_t_mc"),
        )
    with opt_c6:
        opt_confidence = st.selectbox(
            "Confidence level",
            [0.90, 0.95, 0.975, 0.99],
            index=[0.90, 0.95, 0.975, 0.99].index(
                confidence_level
                if confidence_level in (0.90, 0.95, 0.975, 0.99)
                else 0.95
            ),
            format_func=lambda x: f"{x * 100:.1f}%",
            key="opt_confidence",
        )
    with opt_c7:
        opt_expected_method = st.selectbox(
            "Expected return estimator",
            ["mean", "median", "zero", "shrinkage_to_zero", "assumptions_engine"],
            format_func=lambda x: {
                "mean": "Mean",
                "median": "Median",
                "zero": "Zero (pure tail-risk)",
                "shrinkage_to_zero": "Shrinkage to zero",
                "assumptions_engine": "🧠 Robust Assumptions Engine",
            }[x],
            key="opt_expected_method",
        )

    opt_shrinkage_weight = 0.5
    if opt_expected_method == "shrinkage_to_zero":
        opt_shrinkage_weight = st.slider(
            "Historical mean weight (shrinkage)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="opt_shrinkage_weight",
            help="Expected return = weight × historical mean (rest shrinks to zero).",
        )

    if opt_expected_method == "assumptions_engine":
        _ra_state = st.session_state.get("assumptions_results")
        if _ra_state is None:
            st.warning(
                "No assumptions built yet — open the **🧠 Robust "
                "Assumptions** tab and click *Build assumptions* first. "
                "The stored recipe (estimator, trim/winsor/shrinkage "
                "parameters, and manual views) will then be re-applied to "
                "this tab's scenario matrix so horizons stay consistent."
            )
        else:
            _ra_cfg = _ra_state["config"]
            st.info(
                f"Using the Robust Assumptions recipe: "
                f"**{_ra_cfg.expected_return_method}**"
                + (
                    f" + manual views (blend {_ra_cfg.view_blend_weight:.1f})"
                    if _ra_cfg.manual_views
                    else ""
                )
                + " — re-applied to this tab's scenario source and horizon."
            )
    elif opt_expected_method == "zero":
        st.warning(
            "**Zero expected returns**: return-based objectives (Max Return "
            "under CVaR cap, Max Sharpe, positive Target Return) are **not "
            "meaningful** under this estimator — the solver only resolves "
            "constraint feasibility. Use Min CVaR, or pick a non-zero "
            "estimator for return-seeking objectives."
        )

    opt_use_robust_cov = st.checkbox(
        "Use robust covariance from the Assumptions engine for MC scenario "
        "generation",
        value=False,
        key="opt_use_robust_cov",
        disabled=(
            opt_source == "historical"
            or st.session_state.get("assumptions_results") is None
        ),
        help=(
            "Only affects Normal / Student-t Monte Carlo sources: scenarios "
            "are simulated from the covariance recipe (shrinkage / EWMA) "
            "built in the Robust Assumptions tab instead of the sample "
            "covariance."
        ),
    )

    with st.expander(
        "📖 Scenario source — how it shapes the result", expanded=False
    ):
        st.markdown(
            """
The scenario matrix **is** the optimizer's model of the world, so the
same objective can produce different weights per source:

* **Historical** — preserves realized co-movements, volatility
  clustering, and the exact empirical tail. Limited to what actually
  happened.
* **Normal MC** — smooths the empirical tail (thin-tailed by
  construction); tends to *understate* tail risk for crypto and can
  therefore allow more aggressive allocations.
* **Student-t MC** — heavier simulated tails; typically *raises*
  scenario CVaR and pushes the optimizer toward defensive assets.

In practice: assets with fragile tails (e.g. smaller alts) often get
dropped when moving from Historical to Student-t scenarios, while
relatively defensive majors (e.g. BTC) gain weight. If an allocation
flips across sources, that flip is itself information — the position is
tail-model-sensitive. Compare at least Historical vs Student-t before
acting on a result.
"""
        )

    # ── Objective controls ──────────────────────────────────────────────
    st.markdown("#### Objective")
    opt_objective = st.selectbox(
        "Optimization objective",
        [
            "minimize_cvar",
            "max_return_cvar_cap",
            "min_cvar_target_return",
            "maximize_sharpe",
            "efficient_frontier",
            "compare_all",
        ],
        format_func=lambda x: {
            "minimize_cvar": "Minimize CVaR",
            "max_return_cvar_cap": "Maximize return under CVaR cap",
            "min_cvar_target_return": "Minimize CVaR for target return",
            "maximize_sharpe": "Maximize Sharpe ratio",
            "efficient_frontier": "Generate CVaR efficient frontier",
            "compare_all": "Compare all objectives",
        }[x],
        key="opt_objective",
    )

    # ── Risk-free rate ──────────────────────────────────────────────────
    st.markdown("#### Risk-free rate")
    rf_c1, rf_c2, rf_c3 = st.columns(3)
    with rf_c1:
        rf_mode = st.selectbox(
            "Risk-free rate mode",
            ["Zero", "Manual", "Auto from config"],
            key="opt_rf_mode",
        )
    with rf_c2:
        rf_annual = st.number_input(
            "Annual risk-free rate",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.005,
            format="%.4f",
            key="opt_rf_annual",
            disabled=(rf_mode != "Manual"),
        )
    with rf_c3:
        rf_day_count = st.number_input(
            "Day count",
            min_value=1,
            max_value=366,
            value=365,
            step=1,
            key="opt_rf_day_count",
        )

    if rf_mode == "Zero":
        rf_annual_effective = 0.0
    elif rf_mode == "Auto from config":
        rf_annual_effective = _load_risk_free_annual_from_config()
    else:
        rf_annual_effective = float(rf_annual)
    rf_per_horizon = annual_to_horizon_rate(
        rf_annual_effective,
        horizon_days=int(opt_horizon),
        day_count=int(rf_day_count),
    )
    st.caption(
        f"Cash return per horizon (from {rf_annual_effective * 100:.2f}% annual "
        f"over {opt_horizon}d): **{rf_per_horizon * 100:.4f}%** — used for the "
        "Sharpe ratio, the Max-Sharpe portfolio, and the cash asset."
    )

    # ── Constraint controls ─────────────────────────────────────────────
    st.markdown("#### Constraints")
    opt_d1, opt_d2, opt_d3, opt_d4 = st.columns(4)
    with opt_d1:
        opt_long_only = st.checkbox(
            "Long-only",
            value=True,
            key="opt_long_only",
            help="If unchecked, short-selling is allowed and `min_weight` "
            "can be negative.",
        )
    with opt_d2:
        opt_min_weight = st.number_input(
            "Min weight per asset",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            format="%.2f",
            key="opt_min_weight",
        )
    with opt_d3:
        opt_max_weight = st.number_input(
            "Max weight per asset",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.05,
            format="%.2f",
            key="opt_max_weight",
        )
    with opt_d4:
        opt_include_cash = st.checkbox(
            "Include cash asset",
            value=False,
            key="opt_include_cash",
        )

    opt_e1, opt_e2, opt_e3, opt_e4 = st.columns(4)
    with opt_e1:
        opt_cash_return = st.number_input(
            "Cash return per horizon",
            min_value=-0.1,
            max_value=0.1,
            value=0.0,
            step=0.0001,
            format="%.4f",
            key="opt_cash_return",
            disabled=not opt_include_cash,
        )
    with opt_e2:
        opt_cvar_limit = st.number_input(
            "CVaR cap (loss, e.g. 0.10 = 10%)",
            min_value=0.001,
            max_value=1.0,
            value=0.10,
            step=0.005,
            format="%.3f",
            key="opt_cvar_limit",
            disabled=(opt_objective not in (
                "max_return_cvar_cap",
                "compare_all",
            )),
        )
    with opt_e3:
        opt_target_return = st.number_input(
            "Target return (per horizon)",
            min_value=-0.5,
            max_value=0.5,
            value=0.001,
            step=0.0005,
            format="%.4f",
            key="opt_target_return",
            disabled=(opt_objective not in (
                "min_cvar_target_return",
                "compare_all",
            )),
        )
    with opt_e4:
        opt_n_frontier = st.number_input(
            "Frontier points",
            min_value=2,
            max_value=100,
            value=20,
            step=1,
            key="opt_n_frontier",
            disabled=(opt_objective not in (
                "efficient_frontier",
                "compare_all",
            )),
        )

    if opt_long_only and opt_min_weight < 0:
        st.warning(
            "Long-only is enabled — negative min_weight will be clipped to 0."
        )
    if opt_min_weight > 0:
        st.caption(
            f"ℹ️ Min weight {opt_min_weight:.2f} **forces diversification**: "
            "every asset must be held at least at this weight, including "
            "assets the optimizer would otherwise avoid. That lowers "
            "concentration risk but can reduce expected return and Sharpe, "
            "or raise CVaR — the result panel flags assets pinned at the "
            "minimum."
        )

    with st.expander(
        "📖 Which constraints apply to which objective?", expanded=False
    ):
        st.markdown(
            """
| Constraint | Min CVaR | Max Return (CVaR cap) | Min CVaR (target return) | Max Sharpe | Frontier |
|---|---|---|---|---|---|
| Long-only | ✅ | ✅ | ✅ | ✅ | ✅ |
| Min / max weight | ✅ | ✅ | ✅ | ✅ | ✅ |
| **CVaR cap** | — | ✅ *(defines it)* | — | — | — |
| **Target return** | — | — | ✅ *(defines it)* | — | swept over a range |
| Cash asset | ✅ | ✅ | ✅ | ✅ | ✅ |
| Risk-free rate | — | — | — | ✅ *(in Sharpe)* | — |

* **Long-only / min / max weight** are *universal* — they bound every
  objective (and every frontier point).
* **CVaR cap** applies **only** to *Max Return under CVaR cap*.
* **Target return** applies **only** to *Min CVaR for target return*
  (the frontier sweeps many targets internally).
* **Cash**, when enabled, is added as an extra constant-return column to
  every objective. For *Min CVaR* it acts as a safe harbour (expect a
  large cash weight). For *Max Sharpe*, near-100 % cash candidates are
  excluded because a ~zero-volatility portfolio makes the Sharpe ratio
  meaningless — cash is an **absolute** defensive asset, unlike BTC,
  which is only defensive *relative to other crypto*.
"""
        )

    with st.expander(
        "📖 CVaR cap = your risk budget (regime shifts)", expanded=False
    ):
        st.markdown(
            """
The CVaR cap is a **hard risk budget**, and the optimal allocation can
shift *regime-like* as it moves — small cap changes near a transition
point can produce large weight changes:

* **Cap below the minimum achievable CVaR** → *infeasible* (the
  diagnostics below will say so and report the minimum).
* **Tight but feasible cap** → defensive, diversified weights; the cap
  is *binding* (portfolio CVaR = cap).
* **Transition region** → the optimizer rotates from defensive to
  return-seeking assets; allocations are most sensitive here.
* **Loose cap** → the cap stops binding; you effectively get the
  unconstrained max-return portfolio (often concentrated in the
  highest-E[r], highest-risk assets).

The result panel reports whether the cap was **binding**. A binding cap
means the risk budget — not expected return — decided the allocation;
sweep the cap ±2 % to see how stable the weights are.
"""
        )

    # ── Manual expected-return views (optional input layer) ──────────────
    with st.expander("🧭 Manual Expected Return Views (optional)", expanded=False):
        st.caption(
            "Override the estimated expected returns per asset. A clean input "
            "seam for future Black-Litterman / Entropy-Pooling — today it blends "
            "your views with the base estimate."
        )
        opt_use_views = st.checkbox(
            "Enable manual views", value=False, key="opt_use_views"
        )
        opt_views_blend = st.slider(
            "Blend weight (1.0 = fully replace base with your view)",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            key="opt_views_blend",
            disabled=not opt_use_views,
        )
        opt_view_inputs: dict[str, float] = {}
        view_assets = list(asset_returns.columns)
        v_cols = st.columns(max(1, len(view_assets)))
        for i, asset in enumerate(view_assets):
            with v_cols[i % len(v_cols)]:
                opt_view_inputs[asset] = st.number_input(
                    f"{asset} E[r]/horizon",
                    value=0.0,
                    step=0.001,
                    format="%.4f",
                    key=f"opt_view_{asset}",
                    disabled=not opt_use_views,
                )

    run_opt = st.button(
        "▶️ Run optimization",
        type="primary",
        use_container_width=True,
        key="run_opt",
    )

    if run_opt and (
        opt_expected_method == "assumptions_engine"
        and st.session_state.get("assumptions_results") is None
    ):
        st.error(
            "Expected-return estimator is set to the Robust Assumptions "
            "Engine, but no assumptions have been built. Open the 🧠 Robust "
            "Assumptions tab and click **Build assumptions** first."
        )
    elif run_opt:
        try:
            # Optional robust covariance override (MC sources only).
            opt_cov_override = None
            robust_cov_used = False
            if opt_use_robust_cov and opt_source != "historical":
                _ra_state = st.session_state.get("assumptions_results")
                if _ra_state is not None:
                    ra_cov = _ra_state["config"].covariance(asset_returns)
                    if list(ra_cov.columns) == list(asset_returns.columns):
                        opt_cov_override = ra_cov
                        robust_cov_used = True

            scenarios = _scenario_matrix_cached(
                asset_returns,
                opt_source,
                int(opt_n_scenarios),
                int(opt_horizon),
                float(opt_student_df),
                int(opt_seed),
                covariance_matrix=opt_cov_override,
            )
            # Cash earns the risk-free rate when an rf mode is active,
            # otherwise the manual cash-return input is used.
            effective_cash_return = (
                rf_per_horizon if rf_mode != "Zero" else float(opt_cash_return)
            )
            current_weights_full = weights.copy()
            if opt_include_cash:
                scenarios = add_cash_asset(
                    scenarios, cash_return=float(effective_cash_return)
                )
                if "CASH" not in current_weights_full.index:
                    current_weights_full = pd.concat(
                        [current_weights_full, pd.Series({"CASH": 0.0})]
                    )

            estimator_label = opt_expected_method
            if opt_expected_method == "assumptions_engine":
                _ra_state = st.session_state["assumptions_results"]
                ra_cfg = _ra_state["config"]
                # Re-apply the stored recipe to THIS tab's scenario matrix
                # so source and horizon are always consistent.
                expected_returns_vec = ra_cfg.final_expected_returns(
                    scenarios.drop(columns="CASH", errors="ignore")
                )
                if opt_include_cash:
                    expected_returns_vec = pd.concat(
                        [
                            expected_returns_vec,
                            pd.Series({"CASH": float(effective_cash_return)}),
                        ]
                    )
                estimator_label = (
                    f"assumptions_engine ({ra_cfg.expected_return_method}"
                    + (" + views" if ra_cfg.manual_views else "")
                    + ")"
                )
            else:
                expected_returns_vec = estimate_expected_returns(
                    scenarios,
                    method=opt_expected_method,
                    shrinkage_weight=float(opt_shrinkage_weight),
                )
            if opt_use_views:
                views = [
                    AssetReturnView(asset=a, expected_return=float(v))
                    for a, v in opt_view_inputs.items()
                ]
                expected_returns_vec = apply_manual_expected_return_views(
                    expected_returns_vec, views, blend_weight=float(opt_views_blend)
                )

            optimized_results: dict = {}

            def _run_min_cvar() -> dict:
                return minimize_cvar(
                    scenarios,
                    confidence_level=float(opt_confidence),
                    long_only=bool(opt_long_only),
                    min_weight=float(opt_min_weight),
                    max_weight=float(opt_max_weight),
                    include_cash=False,
                )

            def _run_max_ret() -> dict:
                return maximize_return_with_cvar_constraint(
                    scenarios,
                    expected_returns=expected_returns_vec,
                    cvar_limit=float(opt_cvar_limit),
                    confidence_level=float(opt_confidence),
                    long_only=bool(opt_long_only),
                    min_weight=float(opt_min_weight),
                    max_weight=float(opt_max_weight),
                    include_cash=False,
                )

            def _run_target() -> dict:
                return minimize_cvar_for_target_return(
                    scenarios,
                    expected_returns=expected_returns_vec,
                    target_return=float(opt_target_return),
                    confidence_level=float(opt_confidence),
                    long_only=bool(opt_long_only),
                    min_weight=float(opt_min_weight),
                    max_weight=float(opt_max_weight),
                    include_cash=False,
                )

            def _run_max_sharpe() -> dict:
                return maximize_sharpe_ratio(
                    scenarios,
                    expected_returns=expected_returns_vec,
                    risk_free_rate=float(rf_per_horizon),
                    confidence_level=float(opt_confidence),
                    long_only=bool(opt_long_only),
                    min_weight=float(opt_min_weight),
                    max_weight=float(opt_max_weight),
                    include_cash=False,
                    n_grid=int(opt_n_frontier),
                )

            if opt_objective in ("minimize_cvar", "compare_all"):
                optimized_results["Min CVaR"] = _run_min_cvar()
            if opt_objective in ("max_return_cvar_cap", "compare_all"):
                optimized_results["Max Return (CVaR Cap)"] = _run_max_ret()
            if opt_objective in ("min_cvar_target_return", "compare_all"):
                optimized_results["Min CVaR (Target Return)"] = _run_target()
            if opt_objective in ("maximize_sharpe", "compare_all"):
                optimized_results["Max Sharpe"] = _run_max_sharpe()

            frontier_df = pd.DataFrame()
            if opt_objective in ("efficient_frontier", "compare_all"):
                frontier_df = generate_cvar_efficient_frontier(
                    scenarios,
                    expected_returns=expected_returns_vec,
                    confidence_level=float(opt_confidence),
                    n_points=int(opt_n_frontier),
                    long_only=bool(opt_long_only),
                    min_weight=float(opt_min_weight),
                    max_weight=float(opt_max_weight),
                    include_cash=False,
                )

            comparison_df = compare_current_vs_optimized(
                scenarios,
                current_weights_full,
                optimized_results,
                confidence_level=float(opt_confidence),
                initial_capital=float(initial_capital),
                risk_free_rate=float(rf_per_horizon),
            )

            # ── Governance: feasible bounds, per-result interpretation,
            #    and diagnostics for anything that failed to solve ──────
            risk_bounds = compute_feasible_risk_return_bounds(
                scenarios,
                expected_returns=expected_returns_vec,
                confidence_level=float(opt_confidence),
                long_only=bool(opt_long_only),
                min_weight=float(opt_min_weight),
                max_weight=float(opt_max_weight),
            )
            interpretations: dict[str, dict] = {}
            diagnostics: dict[str, list[str]] = {}
            for label, result in optimized_results.items():
                interpretations[label] = interpret_optimization_result(
                    result,
                    cvar_limit=(
                        float(opt_cvar_limit)
                        if label == "Max Return (CVaR Cap)"
                        else None
                    ),
                    target_return=(
                        float(opt_target_return)
                        if label == "Min CVaR (Target Return)"
                        else None
                    ),
                    min_weight=float(opt_min_weight),
                    min_cvar_bound=risk_bounds["min_cvar"],
                    max_return_cvar=risk_bounds["max_return_cvar"],
                )
                if str(result.get("status")) not in (
                    "optimal",
                    "optimal_inaccurate",
                ):
                    diagnostics[label] = diagnose_infeasibility(
                        scenarios,
                        expected_returns=expected_returns_vec,
                        confidence_level=float(opt_confidence),
                        long_only=bool(opt_long_only),
                        min_weight=float(opt_min_weight),
                        max_weight=float(opt_max_weight),
                        cvar_limit=(
                            float(opt_cvar_limit)
                            if label == "Max Return (CVaR Cap)"
                            else None
                        ),
                        target_return=(
                            float(opt_target_return)
                            if label == "Min CVaR (Target Return)"
                            else None
                        ),
                        cash_enabled=bool(opt_include_cash),
                    )

        except (ValueError, RuntimeError) as exc:
            st.error(f"Optimization failed: {exc}")
            st.session_state["opt_results"] = None
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected optimization error: {exc}")
            st.session_state["opt_results"] = None
        else:
            st.session_state["opt_results"] = {
                "source": opt_source,
                "objective": opt_objective,
                "horizon_days": int(opt_horizon),
                "confidence_level": float(opt_confidence),
                "include_cash": bool(opt_include_cash),
                "current_weights": current_weights_full,
                "optimized_results": optimized_results,
                "comparison": comparison_df,
                "frontier": frontier_df,
                "n_scenarios": int(scenarios.shape[0]),
                "assets": list(scenarios.columns),
                # Governance (Phase 7)
                "expected_returns": expected_returns_vec,
                "estimator_label": estimator_label,
                "robust_cov_used": robust_cov_used,
                "risk_bounds": risk_bounds,
                "interpretations": interpretations,
                "diagnostics": diagnostics,
                "constraints": {
                    "long_only": bool(opt_long_only),
                    "min_weight": float(opt_min_weight),
                    "max_weight": float(opt_max_weight),
                    "cvar_limit": float(opt_cvar_limit),
                    "target_return": float(opt_target_return),
                    "cash_return": float(effective_cash_return),
                    "risk_free_per_horizon": float(rf_per_horizon),
                },
            }

    opt_state = st.session_state.get("opt_results")
    if opt_state is None:
        st.info(
            "Pick a scenario source, objective, constraints, then click "
            "**Run optimization**."
        )
    else:
        st.success(
            f"Scenario matrix: {opt_state['n_scenarios']:,} × "
            f"{len(opt_state['assets'])} assets   ·   "
            f"Confidence {opt_state['confidence_level'] * 100:.1f}%   ·   "
            f"Horizon {opt_state['horizon_days']}d"
        )

        opt_results_map = opt_state["optimized_results"]
        comparison_df = opt_state["comparison"]
        frontier_df = opt_state["frontier"]
        opt_interpretations = opt_state.get("interpretations", {})
        opt_diagnostics = opt_state.get("diagnostics", {})
        opt_bounds = opt_state.get("risk_bounds", {})

        # ── Optimizer input governance (Phase 7) ──────────────────────
        with st.expander("🧾 Inputs the optimizer actually received", expanded=True):
            g1, g2, g3, g4 = st.columns(4)
            g1.metric(
                "Scenario source",
                {
                    "historical": "Historical",
                    "normal_mc": "Normal MC",
                    "student_t_mc": "Student-t MC",
                }.get(opt_state["source"], opt_state["source"]),
            )
            g2.metric(
                "Matrix",
                f"{opt_state['n_scenarios']:,} × {len(opt_state['assets'])}",
            )
            g3.metric("Horizon", f"{opt_state['horizon_days']} day(s)")
            g4.metric(
                "Confidence", f"{opt_state['confidence_level'] * 100:.1f}%"
            )

            mu_used = opt_state.get("expected_returns")
            if isinstance(mu_used, pd.Series):
                st.markdown(
                    f"**Expected returns passed to the optimizer** — "
                    f"estimator: `{opt_state.get('estimator_label', '?')}`, "
                    f"**per {opt_state['horizon_days']}-day horizon**:"
                )
                mu_table = pd.DataFrame(
                    {
                        "Asset": mu_used.index.astype(str),
                        f"E[r] per {opt_state['horizon_days']}d (%)": (
                            mu_used.values * 100.0
                        ),
                    }
                )
                st.dataframe(
                    mu_table.round(4), use_container_width=True, hide_index=True
                )
                if bool(np.allclose(mu_used.to_numpy(dtype=float), 0.0)):
                    st.warning(
                        "All expected returns are **zero** — return-based "
                        "objectives in these results reflect constraint "
                        "feasibility only, not return-seeking."
                    )

            cons = opt_state.get("constraints", {})
            if cons:
                st.markdown(
                    f"**Constraints** — long-only: "
                    f"`{cons.get('long_only')}` · min weight: "
                    f"`{cons.get('min_weight'):.2f}` · max weight: "
                    f"`{cons.get('max_weight'):.2f}` · cash: "
                    f"`{'enabled @ ' + format(cons.get('cash_return', 0.0) * 100, '.4f') + '%/horizon' if opt_state['include_cash'] else 'disabled'}` · "
                    f"risk-free/horizon: "
                    f"`{cons.get('risk_free_per_horizon', 0.0) * 100:.4f}%`"
                )
            if opt_state.get("robust_cov_used"):
                st.caption(
                    "✅ MC scenarios were generated from the **robust "
                    "covariance** built in the Assumptions tab."
                )
            if opt_bounds:
                b_min = opt_bounds.get("min_cvar", float("nan"))
                b_max = opt_bounds.get("max_return", float("nan"))
                if pd.notna(b_min) or pd.notna(b_max):
                    st.caption(
                        "Feasible envelope under these constraints — "
                        f"minimum achievable CVaR: "
                        f"**{b_min * 100:.2f}%**"
                        + (
                            f" · maximum achievable E[r]: "
                            f"**{b_max * 100:.3f}%**"
                            if pd.notna(b_max)
                            else ""
                        )
                        + " (per horizon)."
                    )

        # KPI cards for the *primary* optimized portfolio: pick the first
        # result key in the order we'd present them.
        for primary_label in (
            "Min CVaR",
            "Min CVaR (Target Return)",
            "Max Return (CVaR Cap)",
            "Max Sharpe",
        ):
            if primary_label in opt_results_map:
                primary = opt_results_map[primary_label]
                break
        else:
            primary = None
            primary_label = None

        if primary is not None:
            kpis = st.columns(5)
            kpis[0].metric(
                "Status",
                str(primary.get("status", "n/a")),
            )
            er = primary.get("expected_return", float("nan"))
            kpis[1].metric(
                "Expected return",
                f"{er * 100:.2f}%" if pd.notna(er) else "N/A",
            )
            vol = primary.get("volatility", float("nan"))
            kpis[2].metric(
                "Volatility",
                f"{vol * 100:.2f}%" if pd.notna(vol) else "N/A",
            )
            v_var = primary.get("VaR", float("nan"))
            kpis[3].metric(
                "VaR",
                f"{v_var * 100:.2f}%" if pd.notna(v_var) else "N/A",
            )
            v_cvar = primary.get("CVaR", float("nan"))
            kpis[4].metric(
                "CVaR",
                f"{v_cvar * 100:.2f}%" if pd.notna(v_cvar) else "N/A",
            )
            st.caption(
                f"KPI cards reflect: **{primary_label}** — "
                f"{primary.get('message', '')}"
            )

        # ── Weights tables + chart ────────────────────────────────────
        if opt_results_map:
            st.markdown("### Optimized weights")
            cols = st.columns(min(3, len(opt_results_map)))
            for i, (label, result) in enumerate(opt_results_map.items()):
                col = cols[i % len(cols)]
                with col:
                    status_str = str(result.get("status", "?"))
                    status_icon = (
                        "✅"
                        if status_str in ("optimal", "optimal_inaccurate")
                        else "❌"
                    )
                    st.markdown(f"**{label}** · {status_icon} `{status_str}`")
                    weights_series = result.get("weights")
                    if isinstance(weights_series, pd.Series) and not weights_series.isna().all():
                        st.dataframe(
                            format_weights_table(weights_series),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.write(result.get("message", "No weights returned."))
                        for reason in opt_diagnostics.get(label, []):
                            st.error(f"🔎 {reason}")
                    if result.get("warning"):
                        st.warning(result["warning"])
                    interp = opt_interpretations.get(label)
                    if interp and interp.get("notes"):
                        with st.expander(
                            f"🔍 Interpretation — {label}", expanded=False
                        ):
                            profile = interp.get("risk_profile", "unknown")
                            profile_icon = {
                                "defensive": "🛡",
                                "balanced": "⚖️",
                                "aggressive": "🔥",
                            }.get(profile, "❔")
                            st.markdown(
                                f"{profile_icon} **{profile.title()}**"
                                if profile != "unknown"
                                else "❔ Risk profile unknown"
                            )
                            for note in interp["notes"]:
                                st.markdown(f"- {note}")

            # Chart for the primary optimizer (Min CVaR if present).
            if primary is not None and isinstance(
                primary.get("weights"), pd.Series
            ) and not primary["weights"].isna().all():
                fig_w = plot_optimized_weights(
                    primary["weights"],
                    title=f"{primary_label} — Optimized Weights",
                )
                st.pyplot(fig_w, use_container_width=True)
                st.download_button(
                    "⬇️ Download optimized weights chart (PNG)",
                    data=_fig_to_png_bytes(fig_w),
                    file_name=(
                        f"optimized_weights_"
                        f"{primary_label.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
                    ),
                    mime="image/png",
                    key="dl_opt_w",
                )
                plt.close(fig_w)

                st.download_button(
                    "⬇️ Download optimized weights CSV",
                    data=_df_to_csv_bytes(
                        format_weights_table(primary["weights"]),
                        include_index=False,
                    ),
                    file_name=(
                        f"optimized_weights_"
                        f"{primary_label.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"
                    ),
                    mime="text/csv",
                    key="dl_opt_w_csv",
                )

        # ── Comparison table + chart ──────────────────────────────────
        st.markdown("### Current vs optimized — risk comparison")
        comp_display = comparison_df.copy()
        for col in (
            "Expected Return",
            "Volatility",
            "VaR",
            "CVaR",
        ):
            if col in comp_display.columns:
                comp_display[col] = comp_display[col].apply(
                    lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "N/A"
                )
        for col in ("Money VaR", "Money CVaR"):
            if col in comp_display.columns:
                comp_display[col] = comp_display[col].apply(
                    lambda v: f"${v:,.0f}" if pd.notna(v) else "N/A"
                )
        if "Sharpe" in comp_display.columns:
            comp_display["Sharpe"] = comp_display["Sharpe"].apply(
                lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"
            )
        st.dataframe(
            comp_display, use_container_width=True, hide_index=True
        )
        st.download_button(
            "⬇️ Download portfolio_comparison.csv",
            data=_df_to_csv_bytes(comparison_df, include_index=False),
            file_name="portfolio_comparison.csv",
            mime="text/csv",
            key="dl_opt_cmp_csv",
        )

        fig_cmp = plot_portfolio_comparison(comparison_df)
        st.pyplot(fig_cmp, use_container_width=True)
        st.download_button(
            "⬇️ Download risk comparison chart (PNG)",
            data=_fig_to_png_bytes(fig_cmp),
            file_name="current_vs_optimized_risk.png",
            mime="image/png",
            key="dl_opt_cmp_png",
        )
        plt.close(fig_cmp)

        # ── Allocation comparison ─────────────────────────────────────
        if len(opt_results_map) >= 1:
            st.markdown("### Allocation comparison")
            weights_dict = {"Current": opt_state["current_weights"]}
            for label, res in opt_results_map.items():
                w = res.get("weights")
                if isinstance(w, pd.Series) and not w.isna().all():
                    weights_dict[label] = w
            fig_alloc = plot_allocation_comparison(weights_dict)
            st.pyplot(fig_alloc, use_container_width=True)
            st.download_button(
                "⬇️ Download allocation comparison chart (PNG)",
                data=_fig_to_png_bytes(fig_alloc),
                file_name="portfolio_allocation_comparison.png",
                mime="image/png",
                key="dl_opt_alloc",
            )
            plt.close(fig_alloc)

        # ── Efficient frontier ────────────────────────────────────────
        if not frontier_df.empty:
            st.markdown("### CVaR efficient frontier")
            fig_f = plot_cvar_efficient_frontier(frontier_df)
            st.pyplot(fig_f, use_container_width=True)
            st.download_button(
                "⬇️ Download frontier chart (PNG)",
                data=_fig_to_png_bytes(fig_f),
                file_name="cvar_efficient_frontier.png",
                mime="image/png",
                key="dl_opt_frontier_png",
            )
            plt.close(fig_f)

            st.dataframe(
                frontier_df, use_container_width=True, hide_index=True
            )
            st.download_button(
                "⬇️ Download cvar_efficient_frontier.csv",
                data=_df_to_csv_bytes(frontier_df, include_index=False),
                file_name="cvar_efficient_frontier.csv",
                mime="text/csv",
                key="dl_opt_frontier_csv",
            )
