"""Streamlit frontend for the Quantitative Crypto Portfolio Risk Framework.

Run from the project root:

    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
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
from var_cvar_crypto_risk.assumption_charts import (  # noqa: E402
    DEFAULT_EXPECTED_RETURN_VIEW_MODE,
    EXPECTED_RETURN_COMPARISONS,
    EXPECTED_RETURN_VIEW_MODES,
    build_expected_return_dumbbell,
    build_expected_return_estimator_comparison,
)
from var_cvar_crypto_risk.correlation import (  # noqa: E402
    calculate_correlation_matrix,
    calculate_rolling_average_correlation,
    calculate_stress_vs_normal_correlation,
    calculate_weighted_average_correlation,
)
from var_cvar_crypto_risk.covariance import (  # noqa: E402
    prepare_covariance_matrix,
)
from var_cvar_crypto_risk.cvar_models import calculate_cvar  # noqa: E402
from var_cvar_crypto_risk.data_loader import validate_price_data  # noqa: E402
from var_cvar_crypto_risk.historical_summary import (  # noqa: E402
    build_historical_risk_summary,
    format_historical_table as _format_historical_table_contract,
    partition_historical_prices,
)
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
    plot_cvar_efficient_frontier,
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
    plot_var_backtest,
    plot_var_cvar_method_comparison,
)
from var_cvar_crypto_risk.portfolio import (  # noqa: E402
    normalize_weights,
    validate_weights,
)
from var_cvar_crypto_risk.portfolio_path import (  # noqa: E402
    PORTFOLIO_PATH_VERSION,
    PortfolioEvolutionPolicy,
    PortfolioPathConfig,
    build_portfolio_path,
)
from var_cvar_crypto_risk.portfolio_path_charts import (  # noqa: E402
    plot_comparative_drawdown,
    plot_gross_vs_net_nav,
    plot_hold_vs_selected_nav,
    plot_turnover_and_costs,
    plot_weights_drift_and_rebalances,
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
)
from var_cvar_crypto_risk.risk_conventions import (  # noqa: E402
    loss_value_to_money,
)
from var_cvar_crypto_risk.streamlit_ui import (  # noqa: E402
    render_monitoring_workspace,
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
RETURN_CONTRACT_VERSION = 3


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


def _portfolio_path_metadata(provenance: dict) -> dict[str, object]:
    """Flatten path provenance for CSV/table exports."""
    policy = str(provenance["policy"])
    frequency = {
        "buy_and_hold": "none",
        "daily_rebalance": "daily",
        "weekly_rebalance": "weekly",
        "monthly_rebalance": "monthly",
        "quarterly_rebalance": "quarterly",
    }.get(policy, policy)
    return {
        "portfolio_policy": policy,
        "rebalance_frequency": frequency,
        "initial_or_target_weights": json.dumps(
            provenance["target_weights"], sort_keys=True
        ),
        "execution_convention": provenance["execution_convention"],
        "commission_bps": provenance["commission_bps"],
        "slippage_bps": provenance["slippage_bps"],
        "missing_price_policy": provenance["missing_price_policy"],
        "portfolio_path_methodology_version": provenance["methodology_version"],
        "risk_base_value": provenance["risk_base_value"],
        "risk_base_date": provenance["risk_base_date"],
        "risk_base_type": provenance["risk_base_type"],
    }


def _with_portfolio_path_metadata(
    frame: pd.DataFrame, provenance: dict
) -> pd.DataFrame:
    """Return an export copy carrying auditable portfolio-path provenance."""
    result = frame.copy()
    for column, value in _portfolio_path_metadata(provenance).items():
        result[column] = value
    return result


def _portfolio_path_caption(provenance: dict) -> str:
    """Human-readable provenance shared by advanced Risk Lab outputs."""
    metadata = _portfolio_path_metadata(provenance)
    return (
        f"Portfolio basis: **{metadata['portfolio_policy']}** · rebalance: "
        f"**{metadata['rebalance_frequency']}** · commission "
        f"**{float(metadata['commission_bps']):.1f} bps** · slippage "
        f"**{float(metadata['slippage_bps']):.1f} bps** · current risk base "
        f"**{_format_money(float(metadata['risk_base_value']))}** at "
        f"**{metadata['risk_base_date']}** · methodology "
        f"**{metadata['portfolio_path_methodology_version']}**."
    )


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
                        _fetch_yfinance_as_symbols(cg_records, start_date, end_date)
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
    cleaned = clean_price_data(prices, preserve_missing=True)
    validate_price_data(cleaned)
    return cleaned, used_source, tuple(warnings)


# ─── Shared cached data layer ─────────────────────────────────────────────
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


def _render_plotly_chart(
    figure: go.Figure,
    *,
    key: str,
    file_stem: str,
    html_download: bool = True,
) -> None:
    """Render a reusable Plotly figure with client-side PNG export."""
    st.plotly_chart(
        figure,
        width="stretch",
        key=key,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": file_stem,
                "scale": 2,
            },
        },
    )
    if html_download:
        st.download_button(
            "⬇️ Download interactive HTML",
            data=figure.to_html(include_plotlyjs=True, full_html=True).encode("utf-8"),
            file_name=f"{file_stem}.html",
            mime="text/html",
            key=f"download_html_{key}",
        )


def _format_money(value: float) -> str:
    return f"${value:,.0f}"


def _format_historical_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Delegate UI formatting to the Historical Risk Summary contract."""
    return _format_historical_table_contract(frame)


# ─── Page setup ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Quantitative Crypto Portfolio Risk Framework",
    page_icon="📉",
    layout="wide",
)

workspace = st.sidebar.radio(
    "Workspace",
    ["Risk Lab", "Portfolio Monitor"],
    horizontal=True,
    key="workspace",
)
if workspace == "Portfolio Monitor":
    render_monitoring_workspace(PROJECT_ROOT)
    st.stop()

st.title("📉 Quantitative Crypto Portfolio Risk Framework")
st.caption(
    "Interactive risk measurement, backtesting, scenario analysis, robust "
    "assumptions, and CVaR portfolio optimization."
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

    portfolio_policy_label = st.selectbox(
        "Portfolio evolution policy",
        [
            "Buy & Hold — fixed quantities",
            "Periodic rebalance to target weights",
            "Daily rebalanced — legacy constant-weight model",
        ],
        help=(
            "This setting is independent of the VaR/risk horizon. The launch "
            "allocation is treated as already established and has no setup cost."
        ),
    )
    periodic_frequency = "monthly"
    if portfolio_policy_label == "Periodic rebalance to target weights":
        periodic_frequency = st.selectbox(
            "Rebalance frequency",
            ["weekly", "monthly", "quarterly"],
            format_func=lambda value: value.title(),
        )
    is_rebalanced_policy = portfolio_policy_label != "Buy & Hold — fixed quantities"
    if is_rebalanced_policy:
        cost_cols = st.columns(2)
        commission_bps = cost_cols[0].number_input(
            "Commission (bps)", min_value=0.0, value=0.0, step=1.0
        )
        slippage_bps = cost_cols[1].number_input(
            "Slippage (bps)", min_value=0.0, value=0.0, step=1.0
        )
        st.caption(
            "Rebalances execute at a complete UTC close. Weekly/monthly/quarterly "
            "means the last observed date of a completed UTC period. If required "
            "prices are missing, execution is deferred and audited. Close execution "
            "is an idealized research mark, not evidence of an executable fill."
        )
    else:
        commission_bps = 0.0
        slippage_bps = 0.0

    if portfolio_policy_label == "Buy & Hold — fixed quantities":
        portfolio_path_policy = PortfolioEvolutionPolicy.BUY_AND_HOLD
    elif portfolio_policy_label == "Daily rebalanced — legacy constant-weight model":
        portfolio_path_policy = PortfolioEvolutionPolicy.DAILY_REBALANCE
    else:
        portfolio_path_policy = PortfolioEvolutionPolicy(
            f"{periodic_frequency}_rebalance"
        )

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
        "VaR methods",
        options=VAR_METHODS,
        default=VAR_METHODS,
        format_func=lambda m: METHOD_LABELS[m],
    )
    selected_cvar_methods = st.multiselect(
        "CVaR methods",
        options=CVAR_METHODS,
        default=CVAR_METHODS,
        format_func=lambda m: METHOD_LABELS[m],
    )


# ─── Asset / weight editor ────────────────────────────────────────────────

st.subheader("📊 Portfolio")
st.caption(
    "Edit asset symbols, vendor IDs, and weights below. "
    + (
        "These are initial weights at launch; quantities then remain fixed. "
        if portfolio_path_policy is PortfolioEvolutionPolicy.BUY_AND_HOLD
        else "These are target weights restored after each rebalance. "
    )
    + "Weights should sum to 1.0 (auto-normalized if enabled in the sidebar)."
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
    width="stretch",
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

run = st.button("▶️ Run risk analysis", type="primary", width="stretch")


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
            raw_prices, used_source, fetch_warnings = _fetch_prices(
                source=source,
                fallback="yfinance" if fallback_enabled else "",
                assets_records=assets_records,
                quote_currency=quote_currency,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )

        if raw_prices.shape[1] < 1 or len(raw_prices) < 5:
            st.error(
                f"Not enough price data (rows={len(raw_prices)}, "
                f"cols={raw_prices.shape[1]}). Widen the date range."
            )
            st.stop()

        analysis_now_utc = datetime.now(timezone.utc)
        price_partition = partition_historical_prices(
            raw_prices,
            now_utc=analysis_now_utc,
            minimum_finalized_observations=5,
        )
        prices = price_partition.finalized
        provisional_prices = price_partition.provisional

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

        complete_prices = prices.dropna(how="any")
        if len(complete_prices) < 5:
            raise ValueError(
                "Fewer than five complete cross-asset price observations remain."
            )
        asset_returns = calculate_returns(
            complete_prices,
            method=return_policy.portfolio_method,
        )
        path_config = PortfolioPathConfig(
            initial_capital=float(initial_capital),
            policy=portfolio_path_policy,
            commission_bps=float(commission_bps),
            slippage_bps=float(slippage_bps),
        )
        portfolio_path = build_portfolio_path(prices, weights, path_config)
        hold_path = build_portfolio_path(
            prices,
            weights,
            PortfolioPathConfig(
                initial_capital=float(initial_capital),
                policy=PortfolioEvolutionPolicy.BUY_AND_HOLD,
            ),
        )
        portfolio_returns = portfolio_path.net_returns
        portfolio_value = portfolio_path.portfolio.loc[
            portfolio_path.portfolio["finalized"], "net_nav"
        ].rename("portfolio_value")
        current_policy_weights = portfolio_path.latest_weights
        risk_base_value = portfolio_path.latest_net_nav
        risk_base_date = portfolio_path.latest_date.date()
        diagnostic_asset_returns = (
            asset_returns
            if return_policy.diagnostic_method == "simple"
            else calculate_returns(complete_prices, method="log")
        )
        diagnostic_portfolio_returns = (
            portfolio_returns
            if return_policy.diagnostic_method == "simple"
            else pd.Series(
                np.log1p(portfolio_returns.to_numpy(dtype=float)),
                index=portfolio_returns.index,
                name="portfolio_return",
            )
        )

        historical_summary = build_historical_risk_summary(
            path=portfolio_path,
            prices=raw_prices,
            price_source=used_source,
            return_convention="Simple close-to-close net portfolio returns",
            confidence_level=confidence_level,
            var_methods=selected_var_methods,
            cvar_methods=selected_cvar_methods,
            now_utc=analysis_now_utc,
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
        "raw_prices": raw_prices,
        "provisional_prices": provisional_prices,
        "asset_returns": asset_returns,
        "portfolio_returns": portfolio_returns,
        "portfolio_path": portfolio_path,
        "hold_path": hold_path,
        "portfolio_path_version": PORTFOLIO_PATH_VERSION,
        "portfolio_path_provenance": portfolio_path.provenance(),
        "current_policy_weights": current_policy_weights,
        "risk_base_value": risk_base_value,
        "risk_base_date": risk_base_date,
        "risk_base_type": "current_net_nav",
        "diagnostic_asset_returns": diagnostic_asset_returns,
        "diagnostic_portfolio_returns": diagnostic_portfolio_returns,
        "portfolio_value": portfolio_value,
        "historical_summary": historical_summary,
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
if results is not None and (
    results.get("return_contract_version") != RETURN_CONTRACT_VERSION
    or results.get("portfolio_path_version") != PORTFOLIO_PATH_VERSION
):
    st.session_state["risk_results"] = None
    st.session_state["backtest_results"] = None
    st.session_state["mc_results"] = None
    st.session_state["opt_results"] = None
    st.session_state["assumptions_results"] = None
    results = None
    st.info(
        "The return/portfolio-path methodology changed. Run the analysis again "
        "to rebuild results under an explicit evolution policy."
    )
if results is None:
    st.info("Configure inputs in the sidebar and click **Run risk analysis**.")
    st.stop()

for derived_state_key in ("backtest_results", "mc_results", "opt_results"):
    derived_state = st.session_state.get(derived_state_key)
    if (
        derived_state is not None
        and derived_state.get("portfolio_path_version") != PORTFOLIO_PATH_VERSION
    ):
        st.session_state[derived_state_key] = None

prices = results["prices"]
raw_prices = results.get("raw_prices", prices)
provisional_prices = results.get("provisional_prices", prices.iloc[0:0])
asset_returns = results["asset_returns"]
portfolio_returns = results["portfolio_returns"]
portfolio_path = results["portfolio_path"]
hold_path = results["hold_path"]
current_policy_weights = results["current_policy_weights"]
diagnostic_asset_returns = results.get("diagnostic_asset_returns", asset_returns)
diagnostic_portfolio_returns = results.get(
    "diagnostic_portfolio_returns", portfolio_returns
)
portfolio_value = results["portfolio_value"]
historical_summary = results["historical_summary"]
used_source = results["used_source"]
confidence_level = results["confidence_level"]
selected_var_methods = results["selected_var_methods"]
selected_cvar_methods = results["selected_cvar_methods"]
horizon_days = results["horizon_days"]
risk_base_value = float(results["risk_base_value"])
risk_base_date = results["risk_base_date"]
risk_base_type = results["risk_base_type"]
return_handling_mode = results.get("return_handling_mode", "automatic")
diagnostic_return_method = results.get(
    "diagnostic_return_method",
    results.get("returns_method", "simple"),
)


# ─── Run summary ──────────────────────────────────────────────────────────

st.success(
    f"Loaded {len(raw_prices):,} source price rows × {prices.shape[1]} assets "
    f"from **{used_source}**; finalized historical analysis uses {len(prices):,} "
    f"rows ({prices.index.min().date()} → {prices.index.max().date()})."
)
if not provisional_prices.empty:
    st.info(
        f"Excluded {len(provisional_prices):,} provisional current/future UTC "
        "row(s) from finalized NAV, returns, drawdown and VaR/CVaR."
    )
st.caption(
    "Return conventions — core portfolio/NAV/scenarios/optimization: "
    f"**Simple** · diagnostics: **{diagnostic_return_method.title()}** · "
    f"mode: **{return_handling_mode.title()}**"
)
st.caption(
    f"Portfolio path — **{portfolio_path.config.policy.value}** · "
    f"commission **{portfolio_path.config.commission_bps:.1f} bps** · "
    f"slippage **{portfolio_path.config.slippage_bps:.1f} bps** · "
    f"methodology **{portfolio_path.config.methodology_version}**. Risk horizon "
    "does not control the rebalance calendar."
)
for path_warning in portfolio_path.warnings:
    st.warning(path_warning)


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
    st.subheader("Historical Analysis Context")
    st.caption(
        "Historical descriptive analysis of the selected portfolio path. "
        "These results are not forecasts of future performance."
    )
    context_display = historical_summary.context.copy()
    context_display["Value"] = context_display["Value"].map(str)
    st.dataframe(
        context_display,
        width="stretch",
        hide_index=True,
    )

    if historical_summary.quality.is_clean:
        st.success("Historical data-quality checks passed for a complete daily sample.")
    else:
        for quality_warning in historical_summary.quality.warnings():
            st.warning(quality_warning)
    with st.expander(
        "Data-quality audit", expanded=not historical_summary.quality.is_clean
    ):
        quality_display = historical_summary.quality.rows()
        quality_display["Result"] = quality_display["Result"].map(str)
        st.dataframe(
            quality_display,
            width="stretch",
            hide_index=True,
        )
        if historical_summary.quality.non_one_day_intervals:
            interval_rows = pd.DataFrame(
                historical_summary.quality.non_one_day_intervals,
                columns=["Previous finalized date", "Current finalized date", "Days"],
            )
            st.dataframe(interval_rows, width="stretch", hide_index=True)

    primary = historical_summary.primary.set_index("Metric")["Value"]
    first_cards = st.columns(3)
    first_cards[0].metric("Ending Net NAV", _format_money(primary["Ending Net NAV"]))
    first_cards[1].metric(
        "Net Cumulative Return", f"{float(primary['Net Cumulative Return']):.2%}"
    )
    first_cards[2].metric(
        "Maximum Drawdown", f"{float(primary['Maximum Drawdown']):.2%}"
    )
    second_cards = st.columns(3)
    second_cards[0].metric(
        "Total Transaction Costs",
        f"${float(primary['Total Transaction Costs']):,.2f}",
    )
    second_cards[1].metric("Historical Period", str(primary["Historical Period"]))
    second_cards[2].metric(
        "Finalized Observations", f"{int(primary['Finalized Observations']):,}"
    )

    st.subheader("Historical Descriptive Statistics")
    st.caption(
        "Sample descriptions of finalized net returns under the selected policy; "
        "they are not expected-return or volatility forecasts."
    )
    st.dataframe(
        _format_historical_table(historical_summary.descriptive),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Distribution Shape", expanded=False):
        st.caption(
            "Sample skewness and sample excess kurtosis are descriptive moments "
            "and can be highly sensitive to extreme observations."
        )
        st.dataframe(
            _format_historical_table(historical_summary.distribution_shape),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Historical Tail Distribution")
    st.caption(
        "These statistics describe the observed historical return distribution. "
        "They are not forecasts of future losses. Monetary equivalents scale a "
        "historical percentage statistic by ending NAV; they are not realized "
        "historical losses."
    )
    st.dataframe(
        _format_historical_table(historical_summary.tail_distribution),
        width="stretch",
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn(
                "Metric",
                width="medium",
                help=(
                    "A monetary equivalent at ending NAV scales a historical "
                    "percentage statistic by ending NAV; it is not a realized "
                    "historical loss."
                ),
            ),
            "Value": st.column_config.TextColumn("Value", width="small"),
            "Unit": st.column_config.TextColumn("Unit", width="small"),
            "Sample Size": st.column_config.NumberColumn(
                "Sample Size", width="small", format="%d"
            ),
        },
    )

    st.download_button(
        "⬇️ Download historical_risk_summary.csv",
        data=_df_to_csv_bytes(historical_summary.export, include_index=False),
        file_name="historical_risk_summary.csv",
        mime="text/csv",
    )

    with st.expander("📖 Horizon conventions across tabs", expanded=False):
        st.markdown(
            f"""
Different tabs intentionally use different horizon conventions. Same
label, different basis — this table is the reference:

| Where | Convention | Basis |
|---|---|---|
| Risk summary (this tab) | **Historical finalized path** | Net NAV and policy returns |
| Historical Tail Distribution | **Finalized historical return sample** | Policy-specific net returns; data-quality warnings apply |
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
                    f"<br><sup>Historical · "
                    f"{portfolio_path.config.policy.value}</sup>"
                ),
                xlabel=f"{horizon_label} Return",
                extra_var_lines=extra_lines,
                provenance=results["portfolio_path_provenance"],
            )
            _render_plotly_chart(
                fig,
                key="plot_dist_portfolio",
                file_stem="return_distribution_var_cvar",
            )

        if dist_scope in ("Asset-level", "Both"):
            asset_risk_rows = []
            asset_horizon_returns: dict[str, pd.Series] = {}
            asset_risk_levels: dict[str, dict[str, float]] = {}
            for asset in diagnostic_asset_returns.columns:
                a_series = _horizon_returns_cached(
                    diagnostic_asset_returns[asset].dropna(),
                    h,
                    diagnostic_return_method,
                )
                a_var = calculate_var(a_series, "historical", confidence_level)
                a_cvar = calculate_cvar(a_series, "historical", confidence_level)
                asset_horizon_returns[str(asset)] = a_series
                asset_risk_levels[str(asset)] = {"var": a_var, "cvar": a_cvar}
                asset_risk_rows.append(
                    {
                        "Asset": asset,
                        f"{horizon_label} VaR (%)": a_var * 100.0,
                        f"{horizon_label} CVaR (%)": a_cvar * 100.0,
                        f"{horizon_label} Vol (%)": float(a_series.std(ddof=1)) * 100.0,
                    }
                )
            horizon_asset_frame = pd.DataFrame(asset_horizon_returns)
            fig_assets = plot_asset_return_distributions(
                horizon_asset_frame,
                horizon_days=h,
                confidence_level=confidence_level,
                return_method=diagnostic_return_method,
                risk_levels=asset_risk_levels,
                provenance=results["portfolio_path_provenance"],
            )
            _render_plotly_chart(
                fig_assets,
                key="plot_dist_assets",
                file_stem="asset_return_distributions",
            )

            # Asset-level risk table (historical, horizon-matched).
            st.markdown(
                f"**Asset-level historical risk "
                f"({horizon_label}, {confidence_level * 100:.1f}% confidence)**"
            )
            st.dataframe(
                pd.DataFrame(asset_risk_rows).round(2),
                width="stretch",
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
                provenance=results["portfolio_path_provenance"],
            )
            _render_plotly_chart(fig_qq, key="plot_qq", file_stem="historical_qq_plot")

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
    nav_comparison = plot_hold_vs_selected_nav(
        hold_path,
        portfolio_path,
        show_selected_gross=portfolio_path.config.policy.is_rebalanced,
    )
    _render_plotly_chart(
        nav_comparison, key="plot_path_nav", file_stem="portfolio_path_nav"
    )

    weights_figure = plot_weights_drift_and_rebalances(portfolio_path)
    _render_plotly_chart(
        weights_figure, key="plot_path_weights", file_stem="portfolio_path_weights"
    )

    turnover_figure = plot_turnover_and_costs(portfolio_path)
    _render_plotly_chart(
        turnover_figure,
        key="plot_path_turnover",
        file_stem="portfolio_path_turnover_costs",
    )

    gross_net_figure = plot_gross_vs_net_nav(portfolio_path)
    _render_plotly_chart(
        gross_net_figure,
        key="plot_path_gross_net",
        file_stem="portfolio_path_gross_net_nav",
    )
    st.caption(
        "The launch allocation is treated as already established, so no initial "
        "setup cost is charged. Close-price execution is an idealized research mark."
    )

    fig_assets = plot_asset_cumulative_returns(
        asset_returns, provenance=results["portfolio_path_provenance"]
    )
    _render_plotly_chart(
        fig_assets,
        key="plot_asset_cumulative",
        file_stem="asset_cumulative_returns",
    )

with tab_dd:
    drawdown_comparison = plot_comparative_drawdown(hold_path, portfolio_path)
    _render_plotly_chart(
        drawdown_comparison,
        key="plot_path_drawdown",
        file_stem="portfolio_path_drawdown",
    )

    asset_dd = calculate_asset_drawdowns(asset_returns)
    fig_assets_dd = plot_asset_drawdowns(
        asset_dd, provenance=results["portfolio_path_provenance"]
    )
    _render_plotly_chart(
        fig_assets_dd,
        key="plot_asset_drawdowns",
        file_stem="asset_drawdowns",
    )

with tab_data:
    path_provenance = results["portfolio_path_provenance"]
    st.markdown("**Finalized historical prices**")
    st.dataframe(prices.tail(10), width="stretch")
    st.download_button(
        "⬇️ Download prices.csv",
        data=_df_to_csv_bytes(prices),
        file_name="price_data.csv",
        mime="text/csv",
    )
    if not provisional_prices.empty:
        with st.expander("Provisional current/future source rows", expanded=False):
            st.caption(
                "Retained for source audit only. These rows are excluded from all "
                "finalized historical analytics."
            )
            st.dataframe(provisional_prices, width="stretch")
    if not portfolio_path.rebalance_events.empty:
        with st.expander("Rebalance schedule audit", expanded=False):
            st.dataframe(portfolio_path.rebalance_events, width="stretch")

    st.markdown("**Core asset returns (Simple)**")
    st.dataframe(asset_returns.tail(10), width="stretch")
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
    pv_export = _with_portfolio_path_metadata(pv_df, path_provenance)
    st.dataframe(pv_df.tail(10), width="stretch")
    st.download_button(
        "⬇️ Download core_portfolio_returns_simple.csv",
        data=_df_to_csv_bytes(pv_export),
        file_name="core_portfolio_returns_simple.csv",
        mime="text/csv",
    )

    st.markdown("**Audited portfolio path**")
    st.dataframe(portfolio_path.portfolio.tail(20), width="stretch")
    st.download_button(
        "⬇️ Download portfolio_path.csv",
        data=_df_to_csv_bytes(
            _with_portfolio_path_metadata(portfolio_path.portfolio, path_provenance)
        ),
        file_name="portfolio_path.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Download portfolio_path_assets.csv",
        data=_df_to_csv_bytes(
            _with_portfolio_path_metadata(
                portfolio_path.assets.reset_index(), path_provenance
            ),
            include_index=False,
        ),
        file_name="portfolio_path_assets.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Download portfolio_path_provenance.json",
        data=(
            json.dumps(
                results["portfolio_path_provenance"],
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
        file_name="portfolio_path_provenance.json",
        mime="application/json",
    )

    if diagnostic_return_method == "log":
        st.markdown("**Advanced diagnostic returns (Log)**")
        diagnostic_df = diagnostic_asset_returns.copy()
        diagnostic_df["PORTFOLIO"] = diagnostic_portfolio_returns
        st.dataframe(diagnostic_df.tail(10), width="stretch")
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
        st.dataframe(corr_matrix, width="stretch")

        # ── Diversification headline metrics ──────────────────────────────
        n_corr = corr_matrix.shape[0]
        off_diag_mean = float(
            (corr_matrix.to_numpy().sum() - n_corr) / (n_corr * (n_corr - 1))
        )
        met1, met2, met3 = st.columns(3)
        met1.metric("Avg pairwise correlation", f"{off_diag_mean:.3f}")
        try:
            weighted_corr = calculate_weighted_average_correlation(
                corr_matrix, current_policy_weights
            )
            met2.metric(
                "Portfolio-weighted avg correlation",
                f"{weighted_corr:.3f}",
                help=(
                    "Pairwise correlations weighted by the product of the "
                    "current end-of-path weights under the selected policy."
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
            corr_matrix,
            title=f"Historical Asset Return Correlation ({corr_method.title()})",
            provenance={
                **results["portfolio_path_provenance"],
                "correlation_method": corr_method,
            },
        )
        _render_plotly_chart(
            fig_hm, key="plot_corr_heatmap", file_stem="correlation_heatmap"
        )
        st.download_button(
            "⬇️ Download correlation_matrix.csv",
            data=_df_to_csv_bytes(corr_matrix),
            file_name="correlation_matrix.csv",
            mime="text/csv",
            key="dl_corr_csv",
        )

        st.markdown("#### Rolling average pairwise correlation")
        st.caption(
            f"⏱ Rolling correlation **lags by construction**: each point "
            f"averages the past {corr_window} days, so a regime change "
            f"only shows up gradually as new days roll into the window. "
            f"Shorter windows react faster but are noisier."
        )
        if len(asset_returns.dropna()) >= int(corr_window):
            rolling_corr = calculate_rolling_average_correlation(
                asset_returns, window=int(corr_window), method=corr_method
            )
            fig_rc = plot_rolling_average_correlation(
                rolling_corr,
                title="Rolling Average Pairwise Correlation",
                method=corr_method,
                window=int(corr_window),
                provenance=results["portfolio_path_provenance"],
            )
            _render_plotly_chart(
                fig_rc,
                key="plot_rolling_correlation",
                file_stem="rolling_average_correlation",
            )
        else:
            st.info(f"Not enough observations for a {corr_window}-day rolling window.")


# ─── Tab: Robust Assumptions Engine ───────────────────────────────────────

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
        width="stretch",
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
            ra_cov_raw = ra_config.covariance(asset_returns)
            ra_cov, ra_cov_governance = prepare_covariance_matrix(
                ra_cov_raw,
                policy="repair",
            )
        except (ValueError, RuntimeError) as exc:
            st.error(f"Assumption build failed: {exc}")
            st.session_state["assumptions_results"] = None
        else:
            st.session_state["assumptions_results"] = {
                "config": ra_config,
                "table": ra_table,
                "vol_table": ra_vol_table,
                "covariance": ra_cov,
                "covariance_raw": ra_cov_raw,
                "covariance_governance": ra_cov_governance,
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
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "**Final E[r]** is exactly what the optimizer receives when its "
            "estimator is set to *Robust Assumptions Engine*. Large gaps "
            "between mean and median/trimmed columns flag assets whose "
            "average is driven by a few extreme days."
        )

        view_controls = st.columns(2)
        with view_controls[0]:
            view_mode_options = list(EXPECTED_RETURN_VIEW_MODES)
            estimator_view_mode = st.selectbox(
                "View mode",
                view_mode_options,
                index=view_mode_options.index(DEFAULT_EXPECTED_RETURN_VIEW_MODE),
                format_func=lambda value: EXPECTED_RETURN_VIEW_MODES[value],
                key="ra_estimator_view_mode",
            )
        with view_controls[1]:
            comparison_sort = st.selectbox(
                "Asset order",
                ["portfolio", "dispersion"],
                format_func=lambda value: {
                    "portfolio": "Portfolio asset order",
                    "dispersion": "Largest estimator dispersion",
                }[value],
                key="ra_estimator_sort",
            )

        comparison_endpoint = "final_expected_return"
        if estimator_view_mode == "pairwise":
            comparison_options = [
                "median",
                "trimmed_mean",
                "winsorized_mean",
                "shrinkage_to_zero",
            ]
            if ra_state["table"]["manual_view"].notna().any():
                comparison_options.append("manual_view")
            comparison_options.append("final_expected_return")
            comparison_endpoint = st.selectbox(
                "Pairwise estimator",
                comparison_options,
                index=len(comparison_options) - 1,
                format_func=lambda value: EXPECTED_RETURN_COMPARISONS[value],
                key="ra_dumbbell_endpoint",
            )
            st.caption(
                "Raw Historical Mean remains fixed while the selected endpoint "
                "changes. This is an assumption audit, not a forecast."
            )
            dumbbell_figure = build_expected_return_dumbbell(
                ra_state["table"],
                ra_cfg,
                comparison=comparison_endpoint,
                horizon_days=int(h_used),
                asset_order=ra_state["assets"],
                sort_by_dispersion=(comparison_sort == "dispersion"),
            )
            chart_file_stem = "historical_mean_pairwise_comparison"
        else:
            st.caption(
                "Each asset row compares the historical location estimators and "
                "the final value passed downstream. These are model assumptions "
                "derived from the historical sample, not forecasts."
            )
            dumbbell_figure = build_expected_return_estimator_comparison(
                ra_state["table"],
                ra_cfg,
                horizon_days=int(h_used),
                asset_order=ra_state["assets"],
                sort_by_dispersion=(comparison_sort == "dispersion"),
            )
            chart_file_stem = "all_expected_return_estimators"
        _render_plotly_chart(
            dumbbell_figure,
            key="plot_expected_return_dumbbell",
            file_stem=chart_file_stem,
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
            width="stretch",
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
        cov_governance = ra_state.get("covariance_governance", {})
        if cov_governance:
            cov_before = cov_governance["before"]
            cov_after = cov_governance["after"]
            cov_status = (
                "Repaired before use"
                if cov_governance["repaired"]
                else "Valid — no repair needed"
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Covariance status", cov_status)
            c2.metric(
                "Min eigenvalue (before)",
                f"{cov_before['min_eigenvalue']:.3e}",
            )
            c3.metric(
                "Min eigenvalue (used)",
                f"{cov_after['min_eigenvalue']:.3e}",
            )
            condition = cov_after["condition_number"]
            c4.metric(
                "Condition number",
                f"{condition:.3e}" if np.isfinite(condition) else "∞",
            )
            if cov_governance["repaired"]:
                st.warning(
                    "The estimated covariance was not numerically positive "
                    "definite and was repaired before simulation or "
                    "optimization. Marginal variances were preserved. This "
                    "is numerical stabilization, not evidence that the "
                    "covariance estimate is economically correct."
                )
            else:
                st.caption(
                    "The covariance passed symmetry and positive-definiteness "
                    "checks without adjustment."
                )
            with st.expander("Covariance governance details", expanded=False):
                st.json(cov_governance)
        cov_sd = np.sqrt(pd.Series(np.diag(cov_used), index=cov_used.index))
        implied_corr = cov_used / np.outer(cov_sd, cov_sd)
        cov_col1, cov_col2 = st.columns(2)
        with cov_col1:
            st.markdown("**Covariance matrix (daily)**")
            st.dataframe((cov_used * 1e4).round(3), width="stretch")
            st.caption("Values ×10⁻⁴ for readability.")
        with cov_col2:
            fig_ic = plot_correlation_heatmap(
                implied_corr,
                title=f"Implied Correlation — {ra_cfg.covariance_method}",
                provenance={
                    **results["portfolio_path_provenance"],
                    "covariance_method": ra_cfg.covariance_method,
                },
            )
            _render_plotly_chart(
                fig_ic,
                key="plot_implied_correlation",
                file_stem="implied_correlation",
            )
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
                "From",
                value=min_dt,
                min_value=min_dt,
                max_value=max_dt,
                key="bt_start",
            )
            bt_end = st.date_input(
                "To",
                value=max_dt,
                min_value=min_dt,
                max_value=max_dt,
                key="bt_end",
            )

    run_backtest = st.button(
        "▶️ Run Backtest",
        type="primary",
        width="stretch",
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
                        "portfolio_path_provenance": results[
                            "portfolio_path_provenance"
                        ],
                        "portfolio_path_version": PORTFOLIO_PATH_VERSION,
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
                        "portfolio_path_provenance": results[
                            "portfolio_path_provenance"
                        ],
                        "portfolio_path_version": PORTFOLIO_PATH_VERSION,
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
        st.caption(
            _portfolio_path_caption(bt_state["portfolio_path_provenance"])
            + " Forecasts use the sequential realised policy-return history; "
            "risk horizon does not alter its rebalance schedule."
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
        st.dataframe(styled, width="stretch")

        fig_cmp = plot_model_comparison_backtest(
            comparison_df,
            provenance=bt_state["portfolio_path_provenance"],
        )
        _render_plotly_chart(
            fig_cmp,
            key="plot_backtest_model_comparison",
            file_stem="model_comparison_backtest",
        )

        st.download_button(
            "⬇️ Download model_comparison.csv",
            data=_df_to_csv_bytes(
                _with_portfolio_path_metadata(
                    comparison_df, bt_state["portfolio_path_provenance"]
                ),
                include_index=False,
            ),
            file_name="model_comparison.csv",
            mime="text/csv",
        )

        for method_name in ["historical", "gaussian", "cornish_fisher"]:
            label = METHOD_LABELS[method_name]
            with st.expander(f"📊 {label} detail", expanded=False):
                if method_name in forecasts_by_method:
                    fc_df = forecasts_by_method[method_name]
                    fig_a = plot_var_backtest(
                        fc_df,
                        method_name,
                        confidence_used,
                        provenance=bt_state["portfolio_path_provenance"],
                    )
                    _render_plotly_chart(
                        fig_a,
                        key=f"plot_backtest_{method_name}",
                        file_stem=f"var_backtest_{method_name}",
                    )

                    fig_b = plot_breach_timeline(
                        fc_df,
                        method_name,
                        provenance=bt_state["portfolio_path_provenance"],
                    )
                    _render_plotly_chart(
                        fig_b,
                        key=f"plot_breach_timeline_{method_name}",
                        file_stem=f"breach_timeline_{method_name}",
                    )

                    rbw_cmp = min(100, max(2, len(fc_df) // 2))
                    fig_rate_cmp = plot_rolling_breach_rate(
                        calculate_rolling_breach_rate(fc_df, window=rbw_cmp),
                        expected_breach_rate=1.0 - confidence_used,
                        method=method_name,
                        provenance=bt_state["portfolio_path_provenance"],
                    )
                    _render_plotly_chart(
                        fig_rate_cmp,
                        key=f"plot_breach_rate_{method_name}",
                        file_stem=f"rolling_breach_rate_{method_name}",
                    )

                    st.download_button(
                        f"⬇️ Download var_forecasts_{method_name}.csv",
                        data=_df_to_csv_bytes(
                            _with_portfolio_path_metadata(
                                fc_df, bt_state["portfolio_path_provenance"]
                            )
                        ),
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
        st.caption(
            _portfolio_path_caption(bt_state["portfolio_path_provenance"])
            + " Forecasts use the sequential realised policy-return history; "
            "risk horizon does not alter its rebalance schedule."
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
            return f"{value:.4f}" if value is not None and pd.notna(value) else "N/A"

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
            fig_a = plot_var_backtest(
                forecast_df,
                method_used,
                confidence_used,
                provenance=bt_state["portfolio_path_provenance"],
            )
            _render_plotly_chart(
                fig_a,
                key="plot_single_backtest",
                file_stem=f"var_backtesting_exceptions_{method_used}",
            )
        with chart_b:
            fig_b = plot_breach_timeline(
                forecast_df,
                method_used,
                provenance=bt_state["portfolio_path_provenance"],
            )
            _render_plotly_chart(
                fig_b,
                key="plot_single_breach_timeline",
                file_stem=f"breach_timeline_{method_used}",
            )

        with st.expander("📋 Forecast Data", expanded=False):
            st.dataframe(forecast_df.tail(50), width="stretch")

        # ── Rolling breach rate over time ────────────────────────────────
        st.markdown("**📉 Rolling breach rate**")
        rbw = min(100, max(2, len(forecast_df) // 2))
        rolling_rate = calculate_rolling_breach_rate(forecast_df, window=rbw)
        fig_rate = plot_rolling_breach_rate(
            rolling_rate,
            expected_breach_rate=result["expected_breach_rate"],
            method=method_used,
            provenance=bt_state["portfolio_path_provenance"],
        )
        _render_plotly_chart(
            fig_rate,
            key="plot_single_breach_rate",
            file_stem=f"rolling_breach_rate_{method_used}",
        )

        # ── Worst realised horizon losses ────────────────────────────────
        st.markdown("**🔻 Worst realised horizon losses**")
        worst_losses = get_worst_realized_losses(forecast_df, n=10)
        st.dataframe(worst_losses, width="stretch", hide_index=True)
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
            st.dataframe(by_year, width="stretch", hide_index=True)
        except ValueError as exc:
            st.caption(f"Per-period summary unavailable: {exc}")

        dl_a, dl_b = st.columns(2)
        with dl_a:
            st.download_button(
                "⬇️ Download Forecast CSV",
                data=_df_to_csv_bytes(
                    _with_portfolio_path_metadata(
                        forecast_df, bt_state["portfolio_path_provenance"]
                    )
                ),
                file_name=f"var_forecasts_{method_used}.csv",
                mime="text/csv",
            )
        with dl_b:
            result_json = json.dumps(
                {
                    "backtest": {
                        k: (None if isinstance(v, float) and pd.isna(v) else v)
                        for k, v in result.items()
                    },
                    "portfolio_path_provenance": bt_state["portfolio_path_provenance"],
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
        width="stretch",
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
            normal_pf = calculate_portfolio_scenario_returns(
                normal_scen, current_policy_weights
            )
            student_pf = calculate_portfolio_scenario_returns(
                student_scen, current_policy_weights
            )

            pf_mean = float(portfolio_returns.mean())
            pf_vol = float(portfolio_returns.std(ddof=1))
            paths_distribution = (
                "normal" if mc_distribution == "normal" else "student_t"
            )
            paths = simulate_portfolio_paths(
                portfolio_daily_mean=pf_mean,
                portfolio_daily_volatility=pf_vol,
                initial_value=float(risk_base_value),
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
                weights=current_policy_weights,
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
                "current_weights": current_policy_weights.copy(),
                "risk_base_value": risk_base_value,
                "risk_base_date": str(risk_base_date),
                "risk_base_type": risk_base_type,
                "portfolio_path_provenance": results["portfolio_path_provenance"],
                "portfolio_path_version": PORTFOLIO_PATH_VERSION,
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
        money_var = loss_value_to_money(primary_var, mc_state["risk_base_value"])
        money_cvar = loss_value_to_money(primary_cvar, mc_state["risk_base_value"])

        st.caption(
            _portfolio_path_caption(mc_state["portfolio_path_provenance"])
            + " Scenario aggregation uses the current end-of-path weights; "
            "simulated value paths start at current net NAV."
        )

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
            provenance={
                **mc_state["portfolio_path_provenance"],
                "confidence_level": mc_conf_used,
            },
        )
        dist_file_tag = "compare" if dist == "compare" else dist
        _render_plotly_chart(
            fig_dist,
            key="plot_mc_distribution",
            file_stem=f"mc_loss_distribution_{dist_file_tag}",
        )

        if dist == "compare":
            st.markdown("### Normal vs Student-t comparison")
            fig_cmp_dist = plot_normal_vs_student_t_distribution(
                normal_pf,
                student_pf,
                provenance=mc_state["portfolio_path_provenance"],
            )
            _render_plotly_chart(
                fig_cmp_dist,
                key="plot_mc_distribution_comparison",
                file_stem="normal_vs_student_t_distribution",
            )

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
            st.dataframe(pd.DataFrame(cmp_rows), width="stretch", hide_index=True)

        st.markdown("### Portfolio value paths")
        fig_paths = plot_mc_portfolio_paths(
            paths, provenance=mc_state["portfolio_path_provenance"]
        )
        _render_plotly_chart(
            fig_paths,
            key="plot_mc_paths",
            file_stem="mc_portfolio_paths",
        )

        st.markdown("### Method comparison (Historical / Gaussian / CF / MC)")
        st.caption(
            f"All rows are **{mc_horizon_used}-day** VaR/CVaR on the same "
            "basis: Historical / Gaussian / Cornish-Fisher use realised "
            "rolling horizon returns (no √t scaling); the MC rows use "
            "simulated h-day scenarios. These are directly comparable to "
            "each other. They are scenario analyses, not realized performance "
            "or forecasts guaranteed to occur."
        )
        styled_cmp = comparison_all.copy()
        styled_cmp["VaR"] = styled_cmp["VaR"].apply(
            lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "N/A"
        )
        styled_cmp["CVaR"] = styled_cmp["CVaR"].apply(
            lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "N/A"
        )
        st.dataframe(styled_cmp, width="stretch", hide_index=True)

        fig_cmp = plot_var_cvar_method_comparison(
            comparison_all,
            provenance=mc_state["portfolio_path_provenance"],
        )
        _render_plotly_chart(
            fig_cmp,
            key="plot_mc_method_comparison",
            file_stem="var_cvar_method_comparison",
        )

        st.download_button(
            "⬇️ Download model_risk_comparison.csv",
            data=_df_to_csv_bytes(
                _with_portfolio_path_metadata(
                    comparison_all, mc_state["portfolio_path_provenance"]
                ),
                include_index=False,
            ),
            file_name="model_risk_comparison.csv",
            mime="text/csv",
            key="dl_mc_cmp_csv",
        )


# ─── Tab: Portfolio Optimization ──────────────────────────────────────────

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
        "Use robust covariance from the Assumptions engine for MC scenario generation",
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

    with st.expander("📖 Scenario source — how it shapes the result", expanded=False):
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
            disabled=(
                opt_objective
                not in (
                    "max_return_cvar_cap",
                    "compare_all",
                )
            ),
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
            disabled=(
                opt_objective
                not in (
                    "min_cvar_target_return",
                    "compare_all",
                )
            ),
        )
    with opt_e4:
        opt_n_frontier = st.number_input(
            "Frontier points",
            min_value=2,
            max_value=100,
            value=20,
            step=1,
            key="opt_n_frontier",
            disabled=(
                opt_objective
                not in (
                    "efficient_frontier",
                    "compare_all",
                )
            ),
        )

    if opt_long_only and opt_min_weight < 0:
        st.warning("Long-only is enabled — negative min_weight will be clipped to 0.")
    if opt_min_weight > 0:
        st.caption(
            f"ℹ️ Min weight {opt_min_weight:.2f} **forces diversification**: "
            "every asset must be held at least at this weight, including "
            "assets the optimizer would otherwise avoid. That lowers "
            "concentration risk but can reduce expected return and Sharpe, "
            "or raise CVaR — the result panel flags assets pinned at the "
            "minimum."
        )

    with st.expander("📖 Which constraints apply to which objective?", expanded=False):
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

    with st.expander("📖 CVaR cap = your risk budget (regime shifts)", expanded=False):
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
        width="stretch",
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
            robust_cov_governance = None
            if opt_use_robust_cov and opt_source != "historical":
                _ra_state = st.session_state.get("assumptions_results")
                if _ra_state is not None:
                    ra_cov = _ra_state["covariance"]
                    if list(ra_cov.columns) == list(asset_returns.columns):
                        opt_cov_override = ra_cov
                        robust_cov_used = True
                        robust_cov_governance = _ra_state.get("covariance_governance")

            scenarios = _scenario_matrix_cached(
                asset_returns,
                opt_source,
                int(opt_n_scenarios),
                int(opt_horizon),
                float(opt_student_df),
                int(opt_seed),
                covariance_matrix=opt_cov_override,
            )
            scenario_covariance_governance = scenarios.attrs.get(
                "covariance_governance"
            )
            # Cash earns the risk-free rate when an rf mode is active,
            # otherwise the manual cash-return input is used.
            effective_cash_return = (
                rf_per_horizon if rf_mode != "Zero" else float(opt_cash_return)
            )
            current_weights_full = current_policy_weights.copy()
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
                initial_capital=float(risk_base_value),
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
                "risk_base_value": risk_base_value,
                "risk_base_date": str(risk_base_date),
                "risk_base_type": risk_base_type,
                "portfolio_path_provenance": results["portfolio_path_provenance"],
                "portfolio_path_version": PORTFOLIO_PATH_VERSION,
                "optimized_results": optimized_results,
                "comparison": comparison_df,
                "frontier": frontier_df,
                "n_scenarios": int(scenarios.shape[0]),
                "assets": list(scenarios.columns),
                # Optimizer governance
                "expected_returns": expected_returns_vec,
                "estimator_label": estimator_label,
                "robust_cov_used": robust_cov_used,
                "covariance_governance": (
                    robust_cov_governance
                    if robust_cov_governance is not None
                    else scenario_covariance_governance
                ),
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
        st.caption(
            _portfolio_path_caption(opt_state["portfolio_path_provenance"])
            + " The 'Current' comparator uses current end-of-path weights; "
            "optimized portfolios are hypothetical scenario allocations."
        )

        opt_results_map = opt_state["optimized_results"]
        comparison_df = opt_state["comparison"]
        frontier_df = opt_state["frontier"]
        opt_interpretations = opt_state.get("interpretations", {})
        opt_diagnostics = opt_state.get("diagnostics", {})
        opt_bounds = opt_state.get("risk_bounds", {})

        # ── Optimizer input governance ────────────────────────────────
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
            g4.metric("Confidence", f"{opt_state['confidence_level'] * 100:.1f}%")

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
                st.dataframe(mu_table.round(4), width="stretch", hide_index=True)
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
            covariance_governance = opt_state.get("covariance_governance")
            if covariance_governance:
                cov_status = (
                    "repaired"
                    if covariance_governance.get("repaired")
                    else "validated without repair"
                )
                cov_after = covariance_governance.get("after", {})
                st.caption(
                    "Covariance governance — "
                    f"**{cov_status}** · minimum eigenvalue used: "
                    f"`{cov_after.get('min_eigenvalue', float('nan')):.3e}` · "
                    "full diagnostics are available in the Robust "
                    "Assumptions tab."
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
                            f" · maximum achievable E[r]: **{b_max * 100:.3f}%**"
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
                f"KPI cards reflect: **{primary_label}** — {primary.get('message', '')}"
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
                    solver_status = str(result.get("solver_status", "not_run"))
                    validation = result.get("constraint_validation", {})
                    validation_status = validation.get("status", "not_run")
                    max_violation = result.get("max_constraint_violation", float("nan"))
                    st.caption(
                        f"Solver: `{solver_status}` · independent residual "
                        f"check: `{validation_status}` · max violation: "
                        + (
                            f"`{max_violation:.3e}`"
                            if pd.notna(max_violation)
                            else "`N/A`"
                        )
                    )
                    if validation:
                        with st.expander(
                            f"Constraint residuals — {label}", expanded=False
                        ):
                            residual_rows = [
                                {
                                    "Check": key,
                                    "Violation": value,
                                }
                                for key, value in validation.items()
                                if key.endswith("_violation")
                                or key == "budget_residual"
                            ]
                            st.dataframe(
                                pd.DataFrame(residual_rows),
                                width="stretch",
                                hide_index=True,
                            )
                            st.caption(
                                "A result is accepted only when every "
                                "reported violation is within tolerance "
                                f"{validation.get('tolerance', float('nan')):.1e}."
                            )
                    weights_series = result.get("weights")
                    if (
                        isinstance(weights_series, pd.Series)
                        and not weights_series.isna().all()
                    ):
                        st.dataframe(
                            format_weights_table(weights_series),
                            width="stretch",
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
            if (
                primary is not None
                and isinstance(primary.get("weights"), pd.Series)
                and not primary["weights"].isna().all()
            ):
                fig_w = plot_optimized_weights(
                    primary["weights"],
                    title=f"{primary_label} — Optimized Weights",
                    provenance=opt_state["portfolio_path_provenance"],
                )
                optimized_file_stem = (
                    f"optimized_weights_"
                    f"{primary_label.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
                )
                _render_plotly_chart(
                    fig_w,
                    key="plot_optimized_weights",
                    file_stem=optimized_file_stem,
                )

                st.download_button(
                    "⬇️ Download optimized weights CSV",
                    data=_df_to_csv_bytes(
                        _with_portfolio_path_metadata(
                            format_weights_table(primary["weights"]),
                            opt_state["portfolio_path_provenance"],
                        ),
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
        st.dataframe(comp_display, width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Download portfolio_comparison.csv",
            data=_df_to_csv_bytes(
                _with_portfolio_path_metadata(
                    comparison_df, opt_state["portfolio_path_provenance"]
                ),
                include_index=False,
            ),
            file_name="portfolio_comparison.csv",
            mime="text/csv",
            key="dl_opt_cmp_csv",
        )

        fig_cmp = plot_portfolio_comparison(
            comparison_df,
            provenance=opt_state["portfolio_path_provenance"],
        )
        _render_plotly_chart(
            fig_cmp,
            key="plot_optimizer_comparison",
            file_stem="current_vs_optimized_risk",
        )

        # ── Allocation comparison ─────────────────────────────────────
        if len(opt_results_map) >= 1:
            st.markdown("### Allocation comparison")
            weights_dict = {"Current": opt_state["current_weights"]}
            for label, res in opt_results_map.items():
                w = res.get("weights")
                if isinstance(w, pd.Series) and not w.isna().all():
                    weights_dict[label] = w
            fig_alloc = plot_allocation_comparison(
                weights_dict,
                provenance=opt_state["portfolio_path_provenance"],
            )
            _render_plotly_chart(
                fig_alloc,
                key="plot_allocation_comparison",
                file_stem="portfolio_allocation_comparison",
            )

        # ── Efficient frontier ────────────────────────────────────────
        if not frontier_df.empty:
            st.markdown("### CVaR efficient frontier")
            fig_f = plot_cvar_efficient_frontier(
                frontier_df,
                provenance=opt_state["portfolio_path_provenance"],
            )
            _render_plotly_chart(
                fig_f,
                key="plot_cvar_frontier",
                file_stem="cvar_efficient_frontier",
            )

            st.dataframe(frontier_df, width="stretch", hide_index=True)
            st.download_button(
                "⬇️ Download cvar_efficient_frontier.csv",
                data=_df_to_csv_bytes(
                    _with_portfolio_path_metadata(
                        frontier_df, opt_state["portfolio_path_provenance"]
                    ),
                    include_index=False,
                ),
                file_name="cvar_efficient_frontier.csv",
                mime="text/csv",
                key="dl_opt_frontier_csv",
            )
