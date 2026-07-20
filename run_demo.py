"""End-to-end demo for the VaR/CVaR Crypto Portfolio Risk Engine.

Runs the full pipeline in one command:

    python run_demo.py

Step 1 (core risk engine):
    fetch prices → returns → portfolio aggregation
    → VaR / CVaR / drawdown
    → distribution / cumulative / drawdown charts
    → risk_summary table

Step 2 (horizon-aware backtesting & model validation):
    rolling h-day VaR forecasts (h taken from risk.time_horizon_days)
    → Kupiec POF, Christoffersen Independence,
      Christoffersen Conditional Coverage tests
    → Basel III + rate-based traffic light
    → per-method backtest + breach-timeline charts
    → multi-method comparison chart + report table

Step 3 (Monte Carlo scenario engine):
    parameter estimation → Normal & Student-t scenarios
    → scenario VaR / CVaR
    → portfolio value paths
    → cross-method comparison (Historical / Gaussian / Cornish-Fisher /
      Normal MC / Student-t MC)
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from var_cvar_crypto_risk.backtesting import (  # noqa: E402
    backtest_var_model,
    compare_var_models_backtest,
    create_backtesting_report_table,
)
from var_cvar_crypto_risk.config import load_project_config  # noqa: E402
from var_cvar_crypto_risk.cvar_models import calculate_cvar  # noqa: E402
from var_cvar_crypto_risk.data_loader import load_price_data  # noqa: E402
from var_cvar_crypto_risk.export import (  # noqa: E402
    ensure_output_dirs,
    save_dataframe,
    save_json,
    save_series,
)
from var_cvar_crypto_risk.monte_carlo import (  # noqa: E402
    calculate_portfolio_scenario_returns,
    compare_all_risk_methods,
    estimate_return_parameters,
    monte_carlo_risk_summary,
    scenario_cvar,
    scenario_var,
    simulate_normal_returns,
    simulate_portfolio_paths,
    simulate_student_t_returns,
)
from var_cvar_crypto_risk.plotting import (  # noqa: E402
    plot_breach_timeline,
    plot_cumulative_returns,
    plot_drawdown,
    plot_mc_loss_distribution,
    plot_mc_portfolio_paths,
    plot_model_comparison_backtest,
    plot_normal_vs_student_t_distribution,
    plot_return_distribution_with_var_cvar,
    plot_var_backtest,
    plot_var_cvar_method_comparison,
)
from var_cvar_crypto_risk.portfolio import (  # noqa: E402
    calculate_portfolio_returns,
    calculate_portfolio_value,
    get_weights_from_config,
    normalize_weights,
    validate_weights,
)
from var_cvar_crypto_risk.return_conventions import (  # noqa: E402
    resolve_return_policy,
)
from var_cvar_crypto_risk.returns import calculate_returns  # noqa: E402
from var_cvar_crypto_risk.risk_metrics import (  # noqa: E402
    calculate_max_drawdown,
    generate_risk_summary,
)
from var_cvar_crypto_risk.var_models import calculate_var  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
ASSETS_PATH = PROJECT_ROOT / "configs" / "assets.yaml"

METHOD_LABELS = {
    "historical": "Historical",
    "gaussian": "Gaussian",
    "cornish_fisher": "Cornish-Fisher",
}


# ─── Helpers ────────────────────────────────────────────────────────────────


def _format_assets_line(config: dict) -> str:
    parts = [
        f"{symbol} ({float(meta.get('weight', 0)) * 100:.0f}%)"
        for symbol, meta in config["assets"].items()
    ]
    return ", ".join(parts)


def _serializable(result: dict) -> dict:
    out: dict = {}
    for key, value in result.items():
        if isinstance(value, float) and pd.isna(value):
            out[key] = None
        else:
            out[key] = value
    return out


def _backtesting_settings(config: dict) -> dict:
    bt = config.get("backtesting", {}) or {}
    risk = config.get("risk", {}) or {}
    horizon = int(
        bt.get(
            "default_horizon_days",
            risk.get("time_horizon_days", 1),
        )
    )
    return {
        "enabled": bool(bt.get("enabled", True)),
        "confidence_level": float(
            bt.get(
                "default_confidence_level",
                risk.get("confidence_level", 0.95),
            )
        ),
        "window": int(bt.get("default_window", 252)),
        "horizon_days": max(1, horizon),
        "return_method": "simple",
        "traffic_light_mode": str(bt.get("traffic_light_mode", "auto")),
        "methods": list(
            bt.get("methods", ["historical", "gaussian", "cornish_fisher"])
        ),
    }


def _monte_carlo_settings(config: dict) -> dict:
    mc = config.get("monte_carlo", {}) or {}
    risk = config.get("risk", {}) or {}
    return {
        "enabled": bool(mc.get("enabled", True)),
        "default_distribution": str(mc.get("default_distribution", "student_t")),
        "n_scenarios": int(mc.get("n_scenarios", 5000)),
        "horizon_days": int(
            mc.get("horizon_days", risk.get("time_horizon_days", 1))
        ),
        "random_seed": int(mc.get("random_seed", 42)),
        "student_t_df": float(mc.get("student_t_df", 5)),
        "n_paths": int(mc.get("n_paths", 500)),
        "path_horizon_days": int(mc.get("path_horizon_days", 30)),
        "confidence_level": float(risk.get("confidence_level", 0.95)),
    }


# ─── Phase 1: core risk engine ─────────────────────────────────────────────


def _run_core_risk(config: dict, output_dir: str) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    list[str],
]:
    """Compute prices/returns/risk metrics and save Phase 1 artifacts."""
    prices = load_price_data(config)

    configured_diagnostic_method = config["returns"]["method"]
    return_policy = resolve_return_policy(
        "advanced",
        diagnostic_method=configured_diagnostic_method,
    )
    asset_returns = calculate_returns(prices, method=return_policy.portfolio_method)

    weights = get_weights_from_config(config)
    if config["portfolio"]["normalize_weights"]:
        weights = normalize_weights(weights)
    weights = weights.reindex(asset_returns.columns).dropna()
    validate_weights(
        weights,
        assets=list(asset_returns.columns),
        allow_short_selling=config["portfolio"]["allow_short_selling"],
    )

    portfolio_returns = calculate_portfolio_returns(
        asset_returns,
        weights,
        return_method=return_policy.portfolio_method,
    )
    portfolio_value = calculate_portfolio_value(
        portfolio_returns,
        initial_capital=float(config["portfolio"]["initial_capital"]),
        return_method=return_policy.wealth_method,
    )

    risk_summary = generate_risk_summary(
        portfolio_returns=portfolio_returns,
        confidence_level=float(config["risk"]["confidence_level"]),
        initial_capital=float(config["portfolio"]["initial_capital"]),
        var_methods=config["risk"]["var_methods"],
        cvar_methods=config["risk"]["cvar_methods"],
        return_method=return_policy.portfolio_method,
    )

    saved: list[str] = []

    if config["outputs"].get("save_tables", True):
        for path, payload in [
            (f"{output_dir}/tables/price_data.csv", prices),
            (f"{output_dir}/tables/asset_returns.csv", asset_returns),
            (f"{output_dir}/tables/risk_summary.csv", risk_summary),
        ]:
            save_dataframe(payload, path)
            saved.append(path)
        for path, payload in [
            (f"{output_dir}/tables/portfolio_returns.csv", portfolio_returns),
            (f"{output_dir}/tables/portfolio_value.csv", portfolio_value),
        ]:
            save_series(payload, path)
            saved.append(path)

        if return_policy.diagnostic_method == "log":
            diagnostic_asset_returns = calculate_returns(prices, method="log")
            diagnostic_portfolio_returns = calculate_portfolio_returns(
                diagnostic_asset_returns,
                weights,
                return_method="log",
            )
            diagnostic_asset_path = (
                f"{output_dir}/tables/asset_returns_log_diagnostics.csv"
            )
            diagnostic_portfolio_path = (
                f"{output_dir}/tables/portfolio_returns_log_diagnostics.csv"
            )
            save_dataframe(diagnostic_asset_returns, diagnostic_asset_path)
            save_series(diagnostic_portfolio_returns, diagnostic_portfolio_path)
            saved.extend([diagnostic_asset_path, diagnostic_portfolio_path])

    if config["outputs"].get("save_charts", True):
        confidence = float(config["risk"]["confidence_level"])
        primary_var = config["risk"]["var_methods"][0]
        primary_cvar = config["risk"]["cvar_methods"][0]
        var_value = calculate_var(portfolio_returns, primary_var, confidence)
        cvar_value = calculate_cvar(portfolio_returns, primary_cvar, confidence)

        dist_path = f"{output_dir}/charts/return_distribution_var_cvar.png"
        cum_path = f"{output_dir}/charts/cumulative_returns.png"
        dd_path = f"{output_dir}/charts/drawdown.png"

        fig1 = plot_return_distribution_with_var_cvar(
            portfolio_returns,
            var_value=var_value,
            cvar_value=cvar_value,
            confidence_level=confidence,
            output_path=dist_path,
        )
        fig2 = plot_cumulative_returns(portfolio_returns, output_path=cum_path)
        fig3 = plot_drawdown(portfolio_returns, output_path=dd_path)
        for fig in (fig1, fig2, fig3):
            plt.close(fig)
        saved.extend([dist_path, cum_path, dd_path])

    return (
        prices,
        asset_returns,
        portfolio_returns,
        portfolio_value,
        risk_summary,
        weights,
        saved,
    )


def _print_core_summary(
    config: dict,
    prices: pd.DataFrame,
    portfolio_returns: pd.Series,
) -> None:
    var_methods = config["risk"]["var_methods"]
    cvar_methods = config["risk"]["cvar_methods"]
    confidence = float(config["risk"]["confidence_level"])
    initial_capital = float(config["portfolio"]["initial_capital"])

    print("══════════════════════════════════════════════════")
    print(" Flexible VaR/CVaR Crypto Risk Engine")
    print("══════════════════════════════════════════════════")
    print()
    print(f" Assets         : {_format_assets_line(config)}")
    print(
        f" Date Range     : "
        f"{prices.index.min().strftime('%Y-%m-%d')} to "
        f"{prices.index.max().strftime('%Y-%m-%d')}"
    )
    print(f" Observations   : {len(portfolio_returns):,} trading days")
    print(f" Initial Capital: ${initial_capital:,.0f}")
    print(f" Confidence     : {confidence * 100:.1f}%")
    print(" Core Returns   : Simple")
    print(
        f" Diagnostics    : "
        f"{str(config['returns']['method']).title()}"
    )
    print()
    print("──────────────────────────────────────────────────")
    print(" RISK METRICS")
    print("──────────────────────────────────────────────────")
    for method in var_methods:
        var_pct = calculate_var(portfolio_returns, method, confidence)
        label = f"{METHOD_LABELS.get(method, method.title())} VaR"
        print(
            f" {label:<19}:  {var_pct * 100:5.2f}%   "
            f"(${var_pct * initial_capital:,.0f})"
        )
    for method in cvar_methods:
        cvar_pct = calculate_cvar(portfolio_returns, method, confidence)
        label = f"{METHOD_LABELS.get(method, method.title())} CVaR"
        print(
            f" {label:<19}:  {cvar_pct * 100:5.2f}%   "
            f"(${cvar_pct * initial_capital:,.0f})"
        )
    max_dd = calculate_max_drawdown(portfolio_returns)
    print(f" {'Max Drawdown':<19}: {max_dd * 100:6.2f}%")


# ─── Phase 3: backtesting ──────────────────────────────────────────────────


def _run_backtesting(
    portfolio_returns: pd.Series,
    settings: dict,
    output_dir: str,
) -> tuple[dict[str, dict | None], list[str]]:
    """Run rolling backtest for every method and save charts/tables/JSON."""
    saved: list[str] = []
    per_method_results: dict[str, dict | None] = {}

    forecasts_by_method, comparison_df = compare_var_models_backtest(
        portfolio_returns,
        methods=settings["methods"],
        confidence_level=settings["confidence_level"],
        window=settings["window"],
        traffic_light_mode=settings["traffic_light_mode"],
        horizon_days=settings["horizon_days"],
        return_method=settings["return_method"],
    )

    for method in settings["methods"]:
        if method not in forecasts_by_method:
            per_method_results[method] = None
            continue

        forecast_df = forecasts_by_method[method]

        try:
            _fc, result = backtest_var_model(
                portfolio_returns,
                method=method,
                confidence_level=settings["confidence_level"],
                window=settings["window"],
                traffic_light_mode=settings["traffic_light_mode"],
                horizon_days=settings["horizon_days"],
                return_method=settings["return_method"],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  · {method}: result recomputation failed ({exc})")
            per_method_results[method] = None
            continue
        per_method_results[method] = result

        chart_a = f"{output_dir}/charts/var_backtesting_exceptions_{method}.png"
        chart_b = f"{output_dir}/charts/breach_timeline_{method}.png"
        fc_csv = f"{output_dir}/tables/var_forecasts_{method}.csv"
        result_json = f"{output_dir}/tables/backtesting_results_{method}.json"

        fig_a = plot_var_backtest(
            forecast_df,
            method=method,
            confidence_level=settings["confidence_level"],
            output_path=chart_a,
        )
        plt.close(fig_a)

        fig_b = plot_breach_timeline(
            forecast_df, method=method, output_path=chart_b
        )
        plt.close(fig_b)

        save_dataframe(forecast_df, fc_csv)
        save_json(_serializable(result), result_json)
        saved.extend([chart_a, chart_b, fc_csv, result_json])

    cmp_chart = f"{output_dir}/charts/model_comparison_backtest.png"
    cmp_csv = f"{output_dir}/tables/model_comparison.csv"

    fig_cmp = plot_model_comparison_backtest(comparison_df, output_path=cmp_chart)
    plt.close(fig_cmp)

    report_table = create_backtesting_report_table(comparison_df)
    save_dataframe(report_table, cmp_csv)
    saved.extend([cmp_chart, cmp_csv])

    return per_method_results, saved


def _print_backtest_summary(
    settings: dict,
    per_method_results: dict[str, dict | None],
) -> None:
    print()
    print("──────────────────────────────────────────────────")
    print(" BACKTEST RESULTS")
    print("──────────────────────────────────────────────────")
    print(
        f" Window         : {settings['window']}   "
        f"Confidence: {settings['confidence_level'] * 100:.1f}%   "
        f"Horizon: {settings['horizon_days']}d   "
        f"Mode: {settings['traffic_light_mode']}"
    )
    print()
    for method, res in per_method_results.items():
        label = METHOD_LABELS.get(method, method.title())
        if res is None:
            print(f"  {label:<16}: FAILED")
            continue
        kupiec = res.get("kupiec_p_value")
        cc = res.get("cc_p_value")
        kupiec_str = f"{kupiec:.4f}" if kupiec is not None else "N/A"
        cc_str = f"{cc:.4f}" if cc is not None else "N/A"
        light = res.get("traffic_light", "?")
        mode = res.get("traffic_light_mode_used", "?")
        print(
            f"  {label:<16}: "
            f"breaches={res['actual_breaches']:>4} / "
            f"expected={res['expected_breaches']:.1f}  "
            f"Kupiec p={kupiec_str}  CC p={cc_str}  → {light} [{mode}]"
        )


# ─── Phase 4: Monte Carlo ──────────────────────────────────────────────────


def _run_monte_carlo(
    asset_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    weights: pd.Series,
    settings: dict,
    output_dir: str,
) -> tuple[dict, list[str]]:
    """Run Monte Carlo scenario analysis and save Phase 4 artifacts."""
    saved: list[str] = []
    n_scenarios = settings["n_scenarios"]
    horizon = max(1, settings["horizon_days"])
    seed = settings["random_seed"]
    df = settings["student_t_df"]
    confidence = settings["confidence_level"]

    params = estimate_return_parameters(asset_returns)

    normal_scen = simulate_normal_returns(
        mean_vector=params["mean_vector"],
        covariance_matrix=params["covariance_matrix"],
        n_scenarios=n_scenarios,
        horizon_days=horizon,
        random_seed=seed,
    )
    student_scen = simulate_student_t_returns(
        mean_vector=params["mean_vector"],
        covariance_matrix=params["covariance_matrix"],
        df=df,
        n_scenarios=n_scenarios,
        horizon_days=horizon,
        random_seed=seed,
    )

    normal_pf = calculate_portfolio_scenario_returns(normal_scen, weights)
    student_pf = calculate_portfolio_scenario_returns(student_scen, weights)

    normal_var = scenario_var(normal_pf, confidence)
    normal_cvar = scenario_cvar(normal_pf, confidence)
    student_var = scenario_var(student_pf, confidence)
    student_cvar = scenario_cvar(student_pf, confidence)

    normal_summary = monte_carlo_risk_summary(
        normal_pf, confidence_level=confidence, label="Normal MC"
    )
    student_summary = monte_carlo_risk_summary(
        student_pf, confidence_level=confidence, label="Student-t MC"
    )
    mc_summary = pd.concat(
        [normal_summary, student_summary], axis=0, ignore_index=True
    )

    comparison = compare_all_risk_methods(
        portfolio_returns=portfolio_returns,
        asset_returns=asset_returns,
        weights=weights,
        confidence_level=confidence,
        horizon_days=horizon,
        n_scenarios=n_scenarios,
        student_t_df=df,
        random_seed=seed,
        return_method="simple",
    )

    # Portfolio paths use the historical portfolio daily mean/std as inputs.
    pf_mean = float(portfolio_returns.mean())
    pf_vol = float(portfolio_returns.std(ddof=1))
    paths = simulate_portfolio_paths(
        portfolio_daily_mean=pf_mean,
        portfolio_daily_volatility=pf_vol,
        initial_value=100_000.0,
        n_paths=settings["n_paths"],
        horizon_days=settings["path_horizon_days"],
        distribution=settings["default_distribution"],
        df=df,
        random_seed=seed,
        return_method="simple",
    )

    # ── Save tables ──────────────────────────────────────────────────────
    for path, payload in [
        (f"{output_dir}/tables/mc_scenarios_normal.csv", normal_scen),
        (f"{output_dir}/tables/mc_scenarios_student_t.csv", student_scen),
        (f"{output_dir}/tables/model_risk_comparison.csv", comparison),
        (f"{output_dir}/tables/mc_risk_summary.csv", mc_summary),
    ]:
        save_dataframe(payload, path)
        saved.append(path)
    for path, payload in [
        (
            f"{output_dir}/tables/mc_portfolio_returns_normal.csv",
            normal_pf,
        ),
        (
            f"{output_dir}/tables/mc_portfolio_returns_student_t.csv",
            student_pf,
        ),
    ]:
        save_series(payload, path)
        saved.append(path)

    # ── Save charts ──────────────────────────────────────────────────────
    chart_paths = [
        f"{output_dir}/charts/mc_loss_distribution_normal.png",
        f"{output_dir}/charts/mc_loss_distribution_student_t.png",
        f"{output_dir}/charts/mc_portfolio_paths.png",
        f"{output_dir}/charts/normal_vs_student_t_mc.png",
        f"{output_dir}/charts/var_cvar_method_comparison.png",
    ]
    fig_a = plot_mc_loss_distribution(
        normal_pf,
        var_value=normal_var,
        cvar_value=normal_cvar,
        title=f"Normal MC — {horizon}-day Return Distribution",
        output_path=chart_paths[0],
    )
    plt.close(fig_a)
    fig_b = plot_mc_loss_distribution(
        student_pf,
        var_value=student_var,
        cvar_value=student_cvar,
        title=f"Student-t MC — {horizon}-day Return Distribution",
        output_path=chart_paths[1],
    )
    plt.close(fig_b)
    fig_c = plot_mc_portfolio_paths(paths, output_path=chart_paths[2])
    plt.close(fig_c)
    fig_d = plot_normal_vs_student_t_distribution(
        normal_pf, student_pf, output_path=chart_paths[3]
    )
    plt.close(fig_d)
    fig_e = plot_var_cvar_method_comparison(comparison, output_path=chart_paths[4])
    plt.close(fig_e)

    saved.extend(chart_paths)

    summary = {
        "n_scenarios": n_scenarios,
        "horizon_days": horizon,
        "confidence_level": confidence,
        "normal_var": normal_var,
        "normal_cvar": normal_cvar,
        "student_t_var": student_var,
        "student_t_cvar": student_cvar,
        "comparison_table": comparison,
    }
    return summary, saved


def _print_monte_carlo_summary(settings: dict, summary: dict) -> None:
    print()
    print("──────────────────────────────────────────────────")
    print(" MONTE CARLO SCENARIO ENGINE")
    print("──────────────────────────────────────────────────")
    print(
        f" Scenarios     : {summary['n_scenarios']:,}   "
        f"Horizon: {summary['horizon_days']}d   "
        f"Confidence: {summary['confidence_level'] * 100:.1f}%   "
        f"Seed: {settings['random_seed']}"
    )
    print(
        f"  Normal MC      : "
        f"VaR={summary['normal_var'] * 100:.2f}%   "
        f"CVaR={summary['normal_cvar'] * 100:.2f}%"
    )
    print(
        f"  Student-t MC   : "
        f"VaR={summary['student_t_var'] * 100:.2f}%   "
        f"CVaR={summary['student_t_cvar'] * 100:.2f}%   "
        f"(df={settings['student_t_df']})"
    )
    print()
    print(" Method comparison:")
    for _, row in summary["comparison_table"].iterrows():
        var_txt = (
            f"{row['VaR'] * 100:.2f}%" if pd.notna(row["VaR"]) else "N/A"
        )
        cvar_txt = (
            f"{row['CVaR'] * 100:.2f}%" if pd.notna(row["CVaR"]) else "N/A"
        )
        print(
            f"   {str(row['Method']):<22}: VaR={var_txt:>7}   CVaR={cvar_txt:>7}"
        )


def _print_outputs(saved_paths: list[str]) -> None:
    print()
    print("──────────────────────────────────────────────────")
    print(" OUTPUTS SAVED")
    print("──────────────────────────────────────────────────")
    for path in saved_paths:
        print(f" {path}")
    print("══════════════════════════════════════════════════")


# ─── Entry point ───────────────────────────────────────────────────────────


def main() -> int:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found at {CONFIG_PATH}.")
    if not ASSETS_PATH.exists():
        raise FileNotFoundError(f"Assets config not found at {ASSETS_PATH}.")

    config = load_project_config(str(CONFIG_PATH), str(ASSETS_PATH))
    output_dir = config["outputs"]["output_dir"]
    ensure_output_dirs(output_dir)

    print("→ [1/3] Core risk engine: fetching prices, computing VaR / CVaR…")
    (
        prices,
        asset_returns,
        portfolio_returns,
        _portfolio_value,
        _risk_summary,
        weights,
        phase1_saved,
    ) = _run_core_risk(config, output_dir)

    _print_core_summary(config, prices, portfolio_returns)

    bt_settings = _backtesting_settings(config)
    phase3_saved: list[str] = []
    per_method_results: dict[str, dict | None] = {}

    if not bt_settings["enabled"]:
        print()
        print(" Backtesting disabled in config (backtesting.enabled: false) — skipping.")
    elif len(portfolio_returns) <= bt_settings["window"] + bt_settings["horizon_days"]:
        print()
        print(
            f" Skipping backtesting: need more than "
            f"window+horizon={bt_settings['window'] + bt_settings['horizon_days']} "
            f"observations, have {len(portfolio_returns)}."
        )
    else:
        print()
        print(
            "→ [2/3] Backtesting & model validation: "
            f"rolling {bt_settings['horizon_days']}-day forecasts, "
            "Kupiec & Christoffersen tests…"
        )
        per_method_results, phase3_saved = _run_backtesting(
            portfolio_returns, bt_settings, output_dir
        )
        _print_backtest_summary(bt_settings, per_method_results)

    mc_settings = _monte_carlo_settings(config)
    phase4_saved: list[str] = []
    if not mc_settings["enabled"]:
        print()
        print(" Monte Carlo disabled in config (monte_carlo.enabled: false) — skipping.")
    else:
        print()
        print(
            "→ [3/3] Monte Carlo scenario engine: "
            f"{mc_settings['n_scenarios']:,} scenarios × "
            f"{mc_settings['horizon_days']}-day horizon (Normal & Student-t)…"
        )
        try:
            mc_summary, phase4_saved = _run_monte_carlo(
                asset_returns=asset_returns,
                portfolio_returns=portfolio_returns,
                weights=weights,
                settings=mc_settings,
                output_dir=output_dir,
            )
            _print_monte_carlo_summary(mc_settings, mc_summary)
        except Exception as exc:  # noqa: BLE001
            print(f"  Monte Carlo step failed: {exc}")
            traceback.print_exc()

    _print_outputs(phase1_saved + phase3_saved + phase4_saved)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"[UNEXPECTED ERROR] {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
