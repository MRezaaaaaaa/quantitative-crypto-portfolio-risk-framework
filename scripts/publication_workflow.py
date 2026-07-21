"""Deterministic, offline publication-artifact workflow.

The workflow intentionally uses only a repository-local dataset with an
explicit hash and cutoff.  It is a methodology demonstration, not a market
performance backtest.  Generated manifests record the exact inputs,
assumptions, source revision, solver diagnostics, and artifact hashes.
"""

from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from var_cvar_crypto_risk.assumptions import (
    estimate_covariance_robust,
    estimate_expected_returns_robust,
)
from var_cvar_crypto_risk.backtesting import (
    christoffersen_cc_test,
    christoffersen_independence_test,
    kupiec_pof_test,
    rolling_var_forecast,
)
from var_cvar_crypto_risk.cvar_models import calculate_cvar
from var_cvar_crypto_risk.data_loader import validate_price_data
from var_cvar_crypto_risk.optimization import (
    build_optimization_scenarios,
    calculate_portfolio_scenario_metrics,
    minimize_cvar,
)
from var_cvar_crypto_risk.portfolio import (
    calculate_portfolio_returns,
    validate_weights,
)
from var_cvar_crypto_risk.returns import calculate_simple_returns
from var_cvar_crypto_risk.risk_metrics import generate_risk_summary
from var_cvar_crypto_risk.var_models import calculate_var


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ALLOWED_CLAIMS_BOUNDARY = "synthetic_methodology_only"
_GENERATED_FILENAMES = (
    "risk_summary.csv",
    "expected_returns.csv",
    "covariance.csv",
    "backtesting_forecasts.csv",
    "backtesting_summary.csv",
    "allocation_comparison.csv",
    "portfolio_comparison.csv",
    "var_cvar_comparison.svg",
    "allocation_comparison.svg",
    "experiment_summary.json",
)


class PublicationWorkflowError(RuntimeError):
    """Raised when a publication control or integrity check fails."""


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(data: Any) -> str:
    return json.dumps(
        _json_safe(data),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).date().isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(_canonical_json(data), encoding="utf-8", newline="\n")


def _write_csv(path: Path, frame: pd.DataFrame, decimals: int) -> None:
    frame.to_csv(
        path,
        index=False,
        float_format=f"%.{decimals}f",
        lineterminator="\n",
    )


def _resolve_repo_path(relative_path: str) -> Path:
    candidate = (PROJECT_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise PublicationWorkflowError(
            f"Path must remain inside the repository: {relative_path}"
        ) from exc
    return candidate


def _validate_publication_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise PublicationWorkflowError("Publication config must be a YAML mapping.")
    required = {
        "experiment",
        "data",
        "portfolio",
        "returns",
        "risk",
        "assumptions",
        "backtesting",
        "optimization",
        "publication",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise PublicationWorkflowError(f"Missing config sections: {missing}")
    if config["experiment"].get("claims_boundary") != _ALLOWED_CLAIMS_BOUNDARY:
        raise PublicationWorkflowError(
            "This workflow only permits claims_boundary="
            f"'{_ALLOWED_CLAIMS_BOUNDARY}'."
        )
    if config["data"].get("source") != "local_csv":
        raise PublicationWorkflowError("Publication generation must be offline/local_csv.")
    if config["returns"].get("method") != "simple":
        raise PublicationWorkflowError(
            "The publication workflow requires simple returns for consistent "
            "portfolio aggregation and optimization."
        )
    if config["optimization"].get("interpretation") != "in_sample_methodology_demo":
        raise PublicationWorkflowError(
            "Optimization must be labelled as an in-sample methodology demo."
        )
    if config["optimization"].get("objective") != "minimum_cvar":
        raise PublicationWorkflowError(
            "This experiment runner currently supports objective='minimum_cvar' only."
        )
    if config["optimization"].get("scenario_source") != "historical":
        raise PublicationWorkflowError(
            "This deterministic methodology experiment requires historical scenarios."
        )
    if config["assumptions"].get("covariance_frequency") != "daily":
        raise PublicationWorkflowError("Publication covariance must be labelled daily.")
    return config


def load_publication_config(config_path: Path) -> dict[str, Any]:
    """Load and minimally validate a publication experiment configuration."""
    resolved = config_path.resolve()
    if not resolved.is_file():
        raise PublicationWorkflowError(f"Config does not exist: {config_path}")
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise PublicationWorkflowError("Config must be inside the repository.") from exc

    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    return _validate_publication_config(config)


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty_at_generation_start": bool(status),
    }


def _source_tree_sha256() -> str:
    """Hash calculation sources and locked dependency inputs, path-aware."""
    candidates = sorted((PROJECT_ROOT / "src").rglob("*.py"))
    candidates += sorted((PROJECT_ROOT / "scripts").glob("*.py"))
    candidates += [PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "uv.lock"]
    digest = hashlib.sha256()
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        digest.update(relative + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _load_prices(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    data_config = config["data"]
    data_path = _resolve_repo_path(str(data_config["path"]))
    if not data_path.is_file():
        raise PublicationWorkflowError(f"Dataset does not exist: {data_config['path']}")
    actual_hash = sha256_file(data_path)
    expected_hash = str(data_config["expected_sha256"])
    if actual_hash != expected_hash:
        raise PublicationWorkflowError(
            "Dataset hash mismatch; refusing to publish results from an "
            "unreviewed input."
        )

    date_column = str(data_config["date_column"])
    raw = pd.read_csv(data_path)
    if date_column not in raw.columns:
        raise PublicationWorkflowError(f"Missing date column '{date_column}'.")
    if raw[date_column].duplicated().any():
        raise PublicationWorkflowError("Dataset contains duplicate dates.")
    raw[date_column] = pd.to_datetime(raw[date_column], errors="raise")
    raw = raw.set_index(date_column)
    if not raw.index.is_monotonic_increasing:
        raise PublicationWorkflowError("Dataset dates must be sorted ascending.")

    assets = list(config["portfolio"]["weights"])
    missing = sorted(set(assets).difference(raw.columns))
    if missing:
        raise PublicationWorkflowError(f"Dataset is missing assets: {missing}")
    source_prices = raw.loc[:, assets].astype(float)
    validate_price_data(source_prices)

    cutoff = pd.Timestamp(data_config["cutoff_date"])
    prices = source_prices.loc[source_prices.index <= cutoff].copy()
    if prices.empty:
        raise PublicationWorkflowError("No observations exist on or before cutoff_date.")
    if prices.index.max() > cutoff:
        raise PublicationWorkflowError("Look-ahead control failed: data exceeds cutoff.")
    if len(prices) < 50:
        raise PublicationWorkflowError("At least 50 price observations are required.")

    metadata = {
        "path": data_path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": actual_hash,
        "dataset_type": data_config["dataset_type"],
        "license": data_config["license"],
        "attribution": data_config["attribution"],
        "frequency": data_config["frequency"],
        "configured_cutoff_date": cutoff.date().isoformat(),
        "source_start_date": source_prices.index.min().date().isoformat(),
        "source_end_date": source_prices.index.max().date().isoformat(),
        "used_start_date": prices.index.min().date().isoformat(),
        "used_end_date": prices.index.max().date().isoformat(),
        "price_observations": int(len(prices)),
        "assets": assets,
    }
    return prices, metadata


def _test_summary(
    name: str,
    result: dict[str, Any],
    *,
    observations: int,
    breaches: int,
) -> dict[str, Any]:
    return {
        "test": name,
        "observations": result.get("observations", observations),
        "breaches": result.get("breaches", breaches),
        "lr_statistic": result.get("lr_statistic", result.get("lr_cc")),
        "p_value": result.get("p_value"),
        "pass_test": result.get("pass_test"),
        "interpretation": result.get("interpretation"),
    }


def _bar_svg(
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    title: str,
    y_label: str,
) -> str:
    """Return deterministic grouped-bar SVG without environment-dependent fonts."""
    width, height = 900, 520
    left, right, top, bottom = 90, 30, 70, 105
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [value for _, items, _ in series for value in items]
    upper = max(values) if values else 1.0
    upper = upper if upper > 0 else 1.0
    group_width = plot_width / max(len(labels), 1)
    bar_width = min(52.0, group_width * 0.72 / max(len(series), 1))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:22px;font-weight:700}.axis{font-size:13px}.label{font-size:12px}.legend{font-size:13px}</style>',
        f'<text class="title" x="{width / 2:.1f}" y="34" text-anchor="middle">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#556070"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#556070"/>',
    ]
    for tick in range(6):
        value = upper * tick / 5
        y = top + plot_height - (plot_height * tick / 5)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e9f0"/>',
                f'<text class="axis" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{value:.2%}</text>',
            ]
        )
    for group_index, label in enumerate(labels):
        group_center = left + group_width * (group_index + 0.5)
        total_width = bar_width * len(series)
        for series_index, (series_name, series_values, color) in enumerate(series):
            value = series_values[group_index]
            bar_height = plot_height * max(value, 0.0) / upper
            x = group_center - total_width / 2 + series_index * bar_width
            y = top + plot_height - bar_height
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 4:.2f}" height="{bar_height:.2f}" fill="{color}"/>'
            )
        parts.append(
            f'<text class="label" x="{group_center:.2f}" y="{top + plot_height + 24}" text-anchor="middle">{html.escape(label)}</text>'
        )
    legend_x = left
    legend_y = height - 28
    for index, (series_name, _, color) in enumerate(series):
        x = legend_x + index * 190
        parts.extend(
            [
                f'<rect x="{x}" y="{legend_y - 11}" width="14" height="14" fill="{color}"/>',
                f'<text class="legend" x="{x + 21}" y="{legend_y}">{html.escape(series_name)}</text>',
            ]
        )
    parts.append(
        f'<text class="axis" transform="translate(22 {top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>'
    )
    parts.append("</svg>\n")
    return "".join(parts)


def _artifact_record(path: Path, app_section: str | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if app_section is not None:
        record["app_section"] = app_section
    return record


def generate_publication_bundle(
    config_path: Path,
    output_dir: Path,
    *,
    allow_dirty: bool = False,
    overwrite: bool = False,
) -> Path:
    """Generate a deterministic publication bundle and return manifest path."""
    config = _validate_publication_config(load_publication_config(config_path))
    git = _git_metadata()
    if git["dirty_at_generation_start"] and not allow_dirty:
        raise PublicationWorkflowError(
            "Repository is dirty. Commit/revert changes, or use --allow-dirty "
            "only for a non-publication preview."
        )

    destination = output_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        existing_names = {path.name for path in destination.iterdir()}
        known_names = {*_GENERATED_FILENAMES, "manifest.json"}
        unexpected_names = sorted(existing_names.difference(known_names))
        if unexpected_names:
            raise PublicationWorkflowError(
                "Output directory contains unexpected files; refusing to mix "
                f"them into a publication bundle: {unexpected_names}"
            )
        if not overwrite:
            raise PublicationWorkflowError(
                f"Output directory is not empty: {output_dir}. Use --overwrite explicitly."
            )
    destination.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in (*_GENERATED_FILENAMES, "manifest.json"):
            candidate = destination / name
            if candidate.is_file():
                candidate.unlink()

    prices, data_metadata = _load_prices(config)
    portfolio_config = config["portfolio"]
    weights = pd.Series(portfolio_config["weights"], dtype=float, name="weight")
    validate_weights(weights, list(prices.columns), allow_short_selling=False)

    returns = calculate_simple_returns(prices)
    portfolio_returns = calculate_portfolio_returns(returns, weights)
    confidence = float(config["risk"]["confidence_level"])
    decimals = int(config["publication"]["decimal_places"])
    optimization_config = config["optimization"]
    scenarios = build_optimization_scenarios(
        returns,
        source=str(optimization_config["scenario_source"]),
        horizon_days=int(optimization_config["horizon_days"]),
        return_method="simple",
    )

    risk_summary = generate_risk_summary(
        portfolio_returns,
        confidence_level=confidence,
        initial_capital=float(portfolio_config["initial_capital"]),
        var_methods=list(config["risk"]["var_methods"]),
        cvar_methods=list(config["risk"]["cvar_methods"]),
        periods_per_year=int(config["returns"]["periods_per_year"]),
        return_method="simple",
    )
    _write_csv(destination / "risk_summary.csv", risk_summary, decimals)

    assumptions = config["assumptions"]
    expected_horizon = int(assumptions["expected_return_horizon_days"])
    if expected_horizon != int(optimization_config["horizon_days"]):
        expected_return_input = build_optimization_scenarios(
            returns,
            source="historical",
            horizon_days=expected_horizon,
            return_method="simple",
        )
    else:
        expected_return_input = scenarios
    expected_returns = estimate_expected_returns_robust(
        expected_return_input,
        method=str(assumptions["expected_return_method"]),
        trim_proportion=float(assumptions["trim_proportion"]),
        winsor_proportion=float(assumptions["winsor_proportion"]),
        shrinkage_weight=float(assumptions["shrinkage_weight"]),
    )
    expected_frame = (
        expected_returns.rename(f"Expected Return ({expected_horizon}-day)")
        .rename_axis("Asset")
        .reset_index()
    )
    _write_csv(destination / "expected_returns.csv", expected_frame, decimals)

    covariance = estimate_covariance_robust(
        returns,
        method=str(assumptions["covariance_method"]),
        decay_lambda=float(assumptions["ewma_lambda"]),
        shrinkage_delta=float(assumptions["shrinkage_delta"]),
        shrinkage_target=str(assumptions["shrinkage_target"]),
    )
    covariance_frame = covariance.rename_axis("Asset").reset_index()
    _write_csv(destination / "covariance.csv", covariance_frame, decimals)

    backtest_config = config["backtesting"]
    backtest = rolling_var_forecast(
        portfolio_returns,
        method=str(backtest_config["method"]),
        confidence_level=confidence,
        window=int(backtest_config["window"]),
        horizon_days=int(backtest_config["horizon_days"]),
        return_method="simple",
        backtest_mode=str(backtest_config["mode"]),
    )
    backtest_export = backtest.reset_index().rename(columns={backtest.index.name or "index": "Date"})
    if "Date" in backtest_export:
        backtest_export["Date"] = pd.to_datetime(backtest_export["Date"]).dt.date.astype(str)
    _write_csv(destination / "backtesting_forecasts.csv", backtest_export, decimals)

    kupiec = kupiec_pof_test(backtest["breach"], confidence)
    independence = christoffersen_independence_test(backtest["breach"])
    conditional = christoffersen_cc_test(backtest["breach"], confidence)
    forecast_count = len(backtest)
    breach_count = int(backtest["breach"].sum())
    backtesting_summary = pd.DataFrame(
        [
            _test_summary(
                "Kupiec POF",
                kupiec,
                observations=forecast_count,
                breaches=breach_count,
            ),
            _test_summary(
                "Christoffersen Independence",
                independence,
                observations=forecast_count,
                breaches=breach_count,
            ),
            _test_summary(
                "Christoffersen Conditional Coverage",
                conditional,
                observations=forecast_count,
                breaches=breach_count,
            ),
        ]
    )
    _write_csv(destination / "backtesting_summary.csv", backtesting_summary, decimals)

    optimized = minimize_cvar(
        scenarios,
        confidence_level=confidence,
        long_only=bool(optimization_config["long_only"]),
        min_weight=float(optimization_config["min_weight"]),
        max_weight=float(optimization_config["max_weight"]),
        solver=optimization_config.get("solver"),
    )
    if optimized["status"] not in {"optimal", "optimal_inaccurate"}:
        raise PublicationWorkflowError(
            f"Optimizer did not produce an accepted solution: {optimized['status']}"
        )
    if not optimized["constraint_validation"]["passed"]:
        raise PublicationWorkflowError("Optimizer residual validation failed.")
    optimized_weights = optimized["weights"].reindex(weights.index)

    allocation = pd.DataFrame(
        {
            "Asset": weights.index,
            "Current Weight": weights.to_numpy(dtype=float),
            "Minimum-CVaR Weight": optimized_weights.to_numpy(dtype=float),
        }
    )
    _write_csv(destination / "allocation_comparison.csv", allocation, decimals)

    current_metrics = calculate_portfolio_scenario_metrics(
        scenarios,
        weights,
        confidence_level=confidence,
        initial_capital=float(portfolio_config["initial_capital"]),
    )
    optimized_metrics = calculate_portfolio_scenario_metrics(
        scenarios,
        optimized_weights,
        confidence_level=confidence,
        initial_capital=float(portfolio_config["initial_capital"]),
    )
    metric_names = (
        "expected_return",
        "volatility",
        "VaR",
        "CVaR",
        "worst_return",
        "best_return",
        "sharpe_ratio",
        "money_VaR",
        "money_CVaR",
    )
    comparison = pd.DataFrame(
        [
            {
                "Portfolio": "Current",
                "Horizon Days": int(optimization_config["horizon_days"]),
                **{name: current_metrics[name] for name in metric_names},
            },
            {
                "Portfolio": "Minimum CVaR",
                "Horizon Days": int(optimization_config["horizon_days"]),
                **{name: optimized_metrics[name] for name in metric_names},
            },
        ]
    )
    _write_csv(destination / "portfolio_comparison.csv", comparison, decimals)

    chart_methods = ["historical", "gaussian"]
    var_values = [calculate_var(portfolio_returns, method, confidence) for method in chart_methods]
    cvar_values = [calculate_cvar(portfolio_returns, method, confidence) for method in chart_methods]
    (destination / "var_cvar_comparison.svg").write_text(
        _bar_svg(
            [method.replace("_", " ").title() for method in chart_methods],
            [("VaR", var_values, "#3157a4"), ("CVaR", cvar_values, "#d4553f")],
            f"Synthetic Portfolio Daily VaR and CVaR ({confidence:.0%})",
            "Loss as return",
        ),
        encoding="utf-8",
        newline="\n",
    )
    (destination / "allocation_comparison.svg").write_text(
        _bar_svg(
            list(weights.index),
            [
                ("Current", list(weights.astype(float)), "#3157a4"),
                ("Minimum CVaR", list(optimized_weights.astype(float)), "#2b8a6e"),
            ],
            "Synthetic In-Sample Allocation Comparison",
            "Portfolio weight",
        ),
        encoding="utf-8",
        newline="\n",
    )

    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment"]["id"],
        "claims_boundary": config["experiment"]["claims_boundary"],
        "warning": (
            "Synthetic, in-sample methodology demonstration. These results do "
            "not represent real-market performance or an investable strategy."
        ),
        "data": data_metadata,
        "portfolio": {
            "current_weights": weights.to_dict(),
            "optimized_weights": optimized_weights.to_dict(),
        },
        "risk": {
            "confidence_level": confidence,
            "historical_var": calculate_var(portfolio_returns, "historical", confidence),
            "historical_cvar": calculate_cvar(portfolio_returns, "historical", confidence),
            "gaussian_var": calculate_var(portfolio_returns, "gaussian", confidence),
            "gaussian_cvar": calculate_cvar(portfolio_returns, "gaussian", confidence),
        },
        "backtesting": {
            "forecast_count": forecast_count,
            "breach_count": breach_count,
            "tests": [_json_safe(item) for item in backtesting_summary.to_dict("records")],
        },
        "optimization": {
            "interpretation": optimization_config["interpretation"],
            "scenario_count": len(scenarios),
            "status": optimized["status"],
            "solver": optimized["solver"],
            "max_constraint_violation": optimized["max_constraint_violation"],
            "current_metrics": current_metrics,
            "optimized_metrics": optimized_metrics,
        },
        "input_usage": {
            "expected_returns": (
                "Diagnostic robust-assumption output; not consumed by the "
                "minimum-CVaR objective."
            ),
            "covariance": (
                "Diagnostic daily EWMA output; not consumed by historical "
                "minimum-CVaR scenarios."
            ),
            "optimizer": (
                "Consumes rolling historical simple-return scenarios at the "
                f"{optimization_config['horizon_days']}-day horizon."
            ),
        },
    }
    _write_json(destination / "experiment_summary.json", summary)

    config_resolved = config_path.resolve()
    article_mapping = config["publication"].get("article_mapping", {})
    artifact_records = {
        name: _artifact_record(destination / name, article_mapping.get(name))
        for name in _GENERATED_FILENAMES
    }
    manifest = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "generation": {
            "offline": True,
            "deterministic": True,
            "config_path": config_resolved.relative_to(PROJECT_ROOT).as_posix(),
            "config_sha256": sha256_file(config_resolved),
            "uv_lock_sha256": sha256_file(PROJECT_ROOT / "uv.lock"),
            "source_tree_sha256": _source_tree_sha256(),
            "git": git,
            "python_version": platform.python_version(),
            "package_version": importlib.metadata.version(
                "var-cvar-crypto-risk-engine"
            ),
        },
        "data": data_metadata,
        "bias_controls": {
            "look_ahead": "Dataset is truncated at cutoff before all calculations.",
            "backtesting": "Each forecast uses prior-window observations only.",
            "overlap": f"Backtest mode is {backtest_config['mode']}.",
            "optimization": (
                "Optimization is in-sample on rolling historical horizon scenarios; "
                "no out-of-sample performance claim is permitted."
            ),
            "survivorship": (
                "Synthetic fixed assets avoid vendor-universe drift but cannot "
                "demonstrate real-market survivorship-bias control."
            ),
        },
        "assumptions": {
            "returns": config["returns"],
            "risk": config["risk"],
            "robust_estimators": assumptions,
            "backtesting": backtest_config,
            "optimization": optimization_config,
            "input_usage": summary["input_usage"],
        },
        "solver_validation": {
            "status": optimized["status"],
            "solver_status": optimized["solver_status"],
            "solver": optimized["solver"],
            "constraint_validation": optimized["constraint_validation"],
        },
        "artifacts": artifact_records,
    }
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def verify_publication_manifest(manifest_path: Path) -> dict[str, Any]:
    """Verify current sources, inputs, cutoff, revision, and artifact hashes."""
    resolved = manifest_path.resolve()
    if not resolved.is_file():
        raise PublicationWorkflowError(f"Manifest does not exist: {manifest_path}")
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    errors: list[str] = []
    generation = manifest.get("generation", {})
    git = _git_metadata()
    if generation.get("git", {}).get("commit") != git["commit"]:
        errors.append("Git commit does not match the manifest.")
    if generation.get("source_tree_sha256") != _source_tree_sha256():
        errors.append("Source-tree hash does not match the manifest.")

    config_path = _resolve_repo_path(str(generation.get("config_path", "")))
    if not config_path.is_file() or sha256_file(config_path) != generation.get("config_sha256"):
        errors.append("Publication config hash does not match the manifest.")
    lock_path = PROJECT_ROOT / "uv.lock"
    if not lock_path.is_file() or sha256_file(lock_path) != generation.get("uv_lock_sha256"):
        errors.append("uv.lock hash does not match the manifest.")

    data = manifest.get("data", {})
    dataset_path = _resolve_repo_path(str(data.get("path", "")))
    if not dataset_path.is_file() or sha256_file(dataset_path) != data.get("sha256"):
        errors.append("Dataset hash does not match the manifest.")
    try:
        if pd.Timestamp(data["used_end_date"]) > pd.Timestamp(data["configured_cutoff_date"]):
            errors.append("Manifest records data after the configured cutoff.")
    except (KeyError, TypeError, ValueError):
        errors.append("Manifest cutoff metadata is missing or invalid.")

    for name, record in manifest.get("artifacts", {}).items():
        artifact = resolved.parent / name
        if not artifact.is_file():
            errors.append(f"Artifact is missing: {name}")
            continue
        if sha256_file(artifact) != record.get("sha256"):
            errors.append(f"Artifact hash mismatch: {name}")
        if artifact.stat().st_size != record.get("size_bytes"):
            errors.append(f"Artifact size mismatch: {name}")

    expected_names = {"manifest.json", *manifest.get("artifacts", {})}
    actual_names = {path.name for path in resolved.parent.iterdir()}
    unexpected_names = sorted(actual_names.difference(expected_names))
    if unexpected_names:
        errors.append(f"Bundle contains unexpected artifact files: {unexpected_names}")

    if errors:
        raise PublicationWorkflowError("Manifest verification failed: " + " ".join(errors))
    return {
        "verified": True,
        "experiment_id": manifest["experiment"]["id"],
        "artifact_count": len(manifest["artifacts"]),
        "git_commit": git["commit"],
    }
