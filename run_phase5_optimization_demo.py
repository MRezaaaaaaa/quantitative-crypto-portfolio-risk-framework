"""Phase-5 demo: scenario-based CVaR portfolio optimization.

Run from the project root::

    python run_phase5_optimization_demo.py

What this does
--------------
1. Loads the project config and historical crypto prices.
2. Computes asset returns and the user's current portfolio weights.
3. Builds an optimization scenario matrix (historical / Normal MC /
   Student-t MC) using ``optimization.build_optimization_scenarios``.
4. Runs three optimizations on that matrix:

   * Minimum-CVaR portfolio
   * Maximum-return subject to a CVaR cap
   * Minimum-CVaR for a target expected return

   …plus a CVaR efficient frontier sweep.
5. Compares the current portfolio against each optimized portfolio.
6. Saves CSV tables under ``outputs/tables/`` and PNG charts under
   ``outputs/charts/``.
7. Prints a clean console summary.

Heavy lifting lives in :mod:`var_cvar_crypto_risk.optimization`. This
script is intentionally thin orchestration only.
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

from var_cvar_crypto_risk.config import load_project_config  # noqa: E402
from var_cvar_crypto_risk.data_loader import load_price_data  # noqa: E402
from var_cvar_crypto_risk.export import (  # noqa: E402
    ensure_output_dirs,
    save_dataframe,
)
from var_cvar_crypto_risk.optimization import (  # noqa: E402
    add_cash_asset,
    build_optimization_scenarios,
    calculate_portfolio_scenario_metrics,
    compare_current_vs_optimized,
    estimate_expected_returns,
    format_weights_table,
    generate_cvar_efficient_frontier,
    maximize_return_with_cvar_constraint,
    minimize_cvar,
    minimize_cvar_for_target_return,
)
from var_cvar_crypto_risk.plotting import (  # noqa: E402
    plot_allocation_comparison,
    plot_cvar_efficient_frontier,
    plot_optimized_weights,
    plot_portfolio_comparison,
)
from var_cvar_crypto_risk.portfolio import (  # noqa: E402
    get_weights_from_config,
    normalize_weights,
    validate_weights,
)
from var_cvar_crypto_risk.returns import calculate_returns  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
ASSETS_PATH = PROJECT_ROOT / "configs" / "assets.yaml"


# ─── Helpers ────────────────────────────────────────────────────────────────


def _optimization_settings(config: dict) -> dict:
    opt = config.get("optimization", {}) or {}
    risk = config.get("risk", {}) or {}
    return {
        "enabled": bool(opt.get("enabled", True)),
        "scenario_source": str(
            opt.get("default_scenario_source", "historical")
        ),
        "confidence_level": float(
            opt.get("confidence_level", risk.get("confidence_level", 0.95))
        ),
        "horizon_days": int(
            opt.get("horizon_days", risk.get("time_horizon_days", 1))
        ),
        "long_only": bool(opt.get("long_only", True)),
        "min_weight": (
            None if opt.get("min_weight") is None else float(opt.get("min_weight"))
        ),
        "max_weight": (
            None if opt.get("max_weight") is None else float(opt.get("max_weight"))
        ),
        "include_cash": bool(opt.get("include_cash", False)),
        "cash_return": float(opt.get("cash_return", 0.0)),
        "cvar_limit": float(opt.get("cvar_limit", 0.10)),
        "target_return": float(opt.get("target_return", 0.02)),
        "n_frontier_points": int(opt.get("n_frontier_points", 20)),
        "expected_returns_method": str(
            opt.get("expected_returns_method", "mean")
        ),
        "n_scenarios": int(opt.get("n_scenarios", 5000)),
        "student_t_df": float(opt.get("student_t_df", 5)),
        "random_seed": (
            None
            if opt.get("random_seed") is None
            else int(opt.get("random_seed"))
        ),
        "solver": opt.get("solver"),  # None or string
        "initial_capital": float(
            config.get("portfolio", {}).get("initial_capital", 100_000.0)
        ),
    }


def _scenario_label(source: str) -> str:
    return {
        "historical": "Historical",
        "normal_mc": "Normal Monte Carlo",
        "student_t_mc": "Student-t Monte Carlo",
    }.get(source, source)


def _fmt_pct(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def _print_header(config: dict, settings: dict, asset_returns: pd.DataFrame) -> None:
    print("══════════════════════════════════════════════════")
    print(" Flexible VaR/CVaR Crypto Risk Engine")
    print(" Phase 5 — CVaR Portfolio Optimization Demo")
    print("══════════════════════════════════════════════════")
    print()
    print(f" Scenario Source : {_scenario_label(settings['scenario_source'])}")
    print(
        f" Confidence Lvl  : {settings['confidence_level'] * 100:.1f}%   "
        f"Horizon: {settings['horizon_days']}d"
    )
    print(f" Assets          : {', '.join(asset_returns.columns)}")
    print(
        f" Constraints     : long_only={settings['long_only']}   "
        f"min={settings['min_weight']}   max={settings['max_weight']}   "
        f"cash={settings['include_cash']}"
    )
    print(
        f" CVaR Cap        : {settings['cvar_limit'] * 100:.1f}%   "
        f"Target Return: {settings['target_return'] * 100:.2f}%   "
        f"Frontier Points: {settings['n_frontier_points']}"
    )


def _print_portfolio_block(name: str, metrics: dict, extras: dict | None = None) -> None:
    print()
    print(f" {name}:")
    print(f"   Status         : {metrics.get('status', 'n/a')}")
    print(f"   Expected Return: {_fmt_pct(metrics.get('expected_return'))}")
    print(f"   Volatility     : {_fmt_pct(metrics.get('volatility'))}")
    print(f"   VaR            : {_fmt_pct(metrics.get('VaR'))}")
    print(f"   CVaR           : {_fmt_pct(metrics.get('CVaR'))}")
    if extras:
        for k, v in extras.items():
            print(f"   {k}")


# ─── Main demo ──────────────────────────────────────────────────────────────


def main() -> int:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found at {CONFIG_PATH}.")
    if not ASSETS_PATH.exists():
        raise FileNotFoundError(f"Assets config not found at {ASSETS_PATH}.")

    config = load_project_config(str(CONFIG_PATH), str(ASSETS_PATH))
    output_dir = config["outputs"]["output_dir"]
    ensure_output_dirs(output_dir)

    settings = _optimization_settings(config)
    if not settings["enabled"]:
        print("Optimization disabled in config (optimization.enabled: false).")
        return 0

    print("→ Loading prices and computing asset returns…")
    prices = load_price_data(config)
    returns_method = config["returns"]["method"]
    asset_returns = calculate_returns(prices, method=returns_method)

    current_weights = get_weights_from_config(config)
    if config["portfolio"]["normalize_weights"]:
        current_weights = normalize_weights(current_weights)
    current_weights = current_weights.reindex(asset_returns.columns).dropna()
    validate_weights(
        current_weights,
        assets=list(asset_returns.columns),
        allow_short_selling=config["portfolio"]["allow_short_selling"],
    )

    _print_header(config, settings, asset_returns)

    # ── Build scenario matrix ────────────────────────────────────────────
    print()
    print(
        f"→ Building scenario matrix "
        f"({_scenario_label(settings['scenario_source'])})…"
    )
    scenarios = build_optimization_scenarios(
        asset_returns=asset_returns,
        source=settings["scenario_source"],
        n_scenarios=settings["n_scenarios"],
        horizon_days=settings["horizon_days"],
        student_t_df=settings["student_t_df"],
        random_seed=settings["random_seed"],
    )

    # If we include cash, add it BEFORE optimization so it's also part of
    # the comparison table.
    if settings["include_cash"]:
        scenarios = add_cash_asset(scenarios, cash_return=settings["cash_return"])
        # And include CASH=0 in the current-weights series.
        if "CASH" not in current_weights.index:
            current_weights = pd.concat(
                [current_weights, pd.Series({"CASH": 0.0})]
            )

    expected_returns = estimate_expected_returns(
        scenarios, method=settings["expected_returns_method"]
    )

    print(
        f"   Scenarios shape : "
        f"{scenarios.shape[0]:,} × {scenarios.shape[1]}"
    )

    saved: list[str] = []

    # ── Current portfolio metrics ────────────────────────────────────────
    current_metrics = calculate_portfolio_scenario_metrics(
        scenarios,
        current_weights,
        confidence_level=settings["confidence_level"],
        initial_capital=settings["initial_capital"],
    )
    current_metrics["status"] = "current"

    # ── Optimization 1: minimum CVaR ─────────────────────────────────────
    print()
    print("→ Optimization 1/3: Minimum-CVaR portfolio…")
    min_cvar_res = minimize_cvar(
        scenarios,
        confidence_level=settings["confidence_level"],
        long_only=settings["long_only"],
        min_weight=settings["min_weight"],
        max_weight=settings["max_weight"],
        include_cash=False,
        solver=settings["solver"],
    )

    # ── Optimization 2: max return under CVaR cap ───────────────────────
    print(
        f"→ Optimization 2/3: Max return under CVaR ≤ "
        f"{settings['cvar_limit'] * 100:.1f}%…"
    )
    max_ret_res = maximize_return_with_cvar_constraint(
        scenarios,
        expected_returns=expected_returns,
        cvar_limit=settings["cvar_limit"],
        confidence_level=settings["confidence_level"],
        long_only=settings["long_only"],
        min_weight=settings["min_weight"],
        max_weight=settings["max_weight"],
        include_cash=False,
        solver=settings["solver"],
    )

    # ── Optimization 3: minimum CVaR for target return ──────────────────
    print(
        f"→ Optimization 3/3: Min CVaR for target return ≥ "
        f"{settings['target_return'] * 100:.2f}%…"
    )
    target_res = minimize_cvar_for_target_return(
        scenarios,
        expected_returns=expected_returns,
        target_return=settings["target_return"],
        confidence_level=settings["confidence_level"],
        long_only=settings["long_only"],
        min_weight=settings["min_weight"],
        max_weight=settings["max_weight"],
        include_cash=False,
        solver=settings["solver"],
    )

    # ── CVaR efficient frontier ─────────────────────────────────────────
    print(
        f"→ Generating CVaR efficient frontier "
        f"({settings['n_frontier_points']} points)…"
    )
    try:
        frontier = generate_cvar_efficient_frontier(
            scenarios,
            expected_returns=expected_returns,
            confidence_level=settings["confidence_level"],
            n_points=settings["n_frontier_points"],
            long_only=settings["long_only"],
            min_weight=settings["min_weight"],
            max_weight=settings["max_weight"],
            include_cash=False,
            solver=settings["solver"],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"   Frontier generation failed: {exc}")
        frontier = pd.DataFrame()

    # ── Comparison table ────────────────────────────────────────────────
    optimized_results = {
        "Min CVaR": min_cvar_res,
        "Max Return (CVaR Cap)": max_ret_res,
        "Min CVaR (Target Return)": target_res,
    }
    comparison = compare_current_vs_optimized(
        scenarios,
        current_weights,
        optimized_results,
        confidence_level=settings["confidence_level"],
        initial_capital=settings["initial_capital"],
    )

    # ── Optimization summary (status + key metrics per portfolio) ───────
    summary_rows = [
        {
            "Portfolio": "Current",
            "Status": "current",
            "Expected Return": current_metrics["expected_return"],
            "Volatility": current_metrics["volatility"],
            "VaR": current_metrics["VaR"],
            "CVaR": current_metrics["CVaR"],
            "Notes": "Current portfolio (no optimization).",
        }
    ]
    for label, res in optimized_results.items():
        summary_rows.append(
            {
                "Portfolio": label,
                "Status": res.get("status", "unknown"),
                "Expected Return": res.get("expected_return", float("nan")),
                "Volatility": res.get("volatility", float("nan")),
                "VaR": res.get("VaR", float("nan")),
                "CVaR": res.get("CVaR", float("nan")),
                "Notes": res.get("message", ""),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    # ── Save tables ─────────────────────────────────────────────────────
    print()
    print("→ Saving outputs…")
    tables = [
        (
            f"{output_dir}/tables/optimized_weights_min_cvar.csv",
            format_weights_table(min_cvar_res["weights"]),
        ),
        (
            f"{output_dir}/tables/optimized_weights_target_return.csv",
            format_weights_table(target_res["weights"]),
        ),
        (
            f"{output_dir}/tables/optimized_weights_cvar_cap.csv",
            format_weights_table(max_ret_res["weights"]),
        ),
        (f"{output_dir}/tables/optimization_summary.csv", summary_df),
        (f"{output_dir}/tables/portfolio_comparison.csv", comparison),
        (f"{output_dir}/tables/cvar_efficient_frontier.csv", frontier),
    ]
    for path, df in tables:
        save_dataframe(df, path)
        saved.append(path)

    # ── Save charts ─────────────────────────────────────────────────────
    chart_paths = [
        f"{output_dir}/charts/optimized_weights_min_cvar.png",
        f"{output_dir}/charts/current_vs_optimized_risk.png",
        f"{output_dir}/charts/cvar_efficient_frontier.png",
        f"{output_dir}/charts/portfolio_allocation_comparison.png",
    ]

    fig_w = plot_optimized_weights(
        min_cvar_res["weights"],
        title="Minimum-CVaR Portfolio Weights",
        output_path=chart_paths[0],
    )
    plt.close(fig_w)

    fig_c = plot_portfolio_comparison(comparison, output_path=chart_paths[1])
    plt.close(fig_c)

    fig_f = plot_cvar_efficient_frontier(frontier, output_path=chart_paths[2])
    plt.close(fig_f)

    weights_dict = {
        "Current": current_weights,
    }
    if isinstance(min_cvar_res["weights"], pd.Series):
        weights_dict["Min CVaR"] = min_cvar_res["weights"]
    if isinstance(target_res["weights"], pd.Series):
        weights_dict["Target Return"] = target_res["weights"]
    if isinstance(max_ret_res["weights"], pd.Series):
        weights_dict["CVaR Cap"] = max_ret_res["weights"]
    fig_a = plot_allocation_comparison(weights_dict, output_path=chart_paths[3])
    plt.close(fig_a)

    saved.extend(chart_paths)

    # ── Console summary ─────────────────────────────────────────────────
    print()
    print("──────────────────────────────────────────────────")
    print(" OPTIMIZATION RESULTS")
    print("──────────────────────────────────────────────────")
    _print_portfolio_block("Current Portfolio", current_metrics)
    _print_portfolio_block(
        "Minimum CVaR Portfolio",
        {**min_cvar_res, "status": min_cvar_res.get("status")},
    )
    _print_portfolio_block(
        "Max Return under CVaR Cap",
        {**max_ret_res, "status": max_ret_res.get("status")},
        extras={
            f"CVaR Limit: {settings['cvar_limit'] * 100:.1f}%": None,
        },
    )
    _print_portfolio_block(
        "Min CVaR (Target Return)",
        {**target_res, "status": target_res.get("status")},
        extras={
            f"Target Return: {settings['target_return'] * 100:.2f}%": None,
        },
    )

    print()
    print("──────────────────────────────────────────────────")
    print(" OUTPUTS SAVED")
    print("──────────────────────────────────────────────────")
    for path in saved:
        print(f" {path}")
    print("══════════════════════════════════════════════════")
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
