"""Pure, auditable portfolio evolution with explicit rebalancing policies.

Trades are decided and valued at a complete UTC close.  Post-trade quantities
become effective for the following return interval.  The launch allocation is
treated as already established and therefore carries no setup transaction cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd


PORTFOLIO_PATH_VERSION = "portfolio-path-v1"
_WEIGHT_TOLERANCE = 1e-8
_ACCOUNTING_TOLERANCE = 1e-8


class PortfolioEvolutionPolicy(str, Enum):
    """Supported V1 portfolio-evolution policies."""

    BUY_AND_HOLD = "buy_and_hold"
    DAILY_REBALANCE = "daily_rebalance"
    WEEKLY_REBALANCE = "weekly_rebalance"
    MONTHLY_REBALANCE = "monthly_rebalance"
    QUARTERLY_REBALANCE = "quarterly_rebalance"

    @property
    def is_rebalanced(self) -> bool:
        return self is not PortfolioEvolutionPolicy.BUY_AND_HOLD


@dataclass(frozen=True)
class PortfolioPathConfig:
    """Inputs that define one portfolio path and its transaction-cost model."""

    initial_capital: float
    policy: PortfolioEvolutionPolicy = PortfolioEvolutionPolicy.BUY_AND_HOLD
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    missing_price_policy: str = "defer_rebalance"
    methodology_version: str = PORTFOLIO_PATH_VERSION

    def __post_init__(self) -> None:
        capital = float(self.initial_capital)
        commission = float(self.commission_bps)
        slippage = float(self.slippage_bps)
        if not math.isfinite(capital) or capital <= 0.0:
            raise ValueError("initial_capital must be finite and positive")
        if not math.isfinite(commission) or commission < 0.0:
            raise ValueError("commission_bps must be finite and non-negative")
        if not math.isfinite(slippage) or slippage < 0.0:
            raise ValueError("slippage_bps must be finite and non-negative")
        if commission + slippage >= 10_000.0:
            raise ValueError(
                "combined commission and slippage must be below 10,000 bps"
            )
        if self.missing_price_policy != "defer_rebalance":
            raise ValueError("V1 supports missing_price_policy='defer_rebalance' only")
        if not str(self.methodology_version).strip():
            raise ValueError("methodology_version is required")
        object.__setattr__(self, "initial_capital", capital)
        object.__setattr__(self, "commission_bps", commission)
        object.__setattr__(self, "slippage_bps", slippage)
        object.__setattr__(self, "policy", PortfolioEvolutionPolicy(self.policy))

    @property
    def cost_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 10_000.0

    @property
    def execution_convention(self) -> str:
        return (
            "complete UTC close; post-trade quantities apply to the next interval; "
            "launch allocation is pre-established and has no setup cost"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "initial_capital": self.initial_capital,
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
            "combined_cost_rate": self.cost_rate,
            "missing_price_policy": self.missing_price_policy,
            "execution_convention": self.execution_convention,
            "methodology_version": self.methodology_version,
        }


@dataclass(frozen=True)
class PortfolioPathResult:
    """Portfolio- and asset-level time series plus frozen provenance."""

    portfolio: pd.DataFrame
    assets: pd.DataFrame
    rebalance_events: pd.DataFrame
    target_weights: pd.Series
    config: PortfolioPathConfig
    warnings: tuple[str, ...] = ()

    @property
    def net_returns(self) -> pd.Series:
        result = self.portfolio.loc[
            self.portfolio["finalized"] & ~self.portfolio["is_launch"], "net_return"
        ].dropna()
        result.name = "portfolio_return"
        return result

    @property
    def gross_returns(self) -> pd.Series:
        result = self.portfolio.loc[
            self.portfolio["finalized"] & ~self.portfolio["is_launch"], "gross_return"
        ].dropna()
        result.name = "gross_portfolio_return"
        return result

    @property
    def latest_net_nav(self) -> float:
        complete = self.portfolio.loc[self.portfolio["finalized"], "net_nav"].dropna()
        if complete.empty:
            raise ValueError("portfolio path has no finalized NAV")
        return float(complete.iloc[-1])

    @property
    def latest_date(self) -> pd.Timestamp:
        complete = self.portfolio.index[self.portfolio["finalized"]]
        if complete.empty:
            raise ValueError("portfolio path has no finalized date")
        return pd.Timestamp(complete[-1])

    @property
    def latest_weights(self) -> pd.Series:
        frame = self.assets.xs(self.latest_date, level="date")
        result = frame["post_trade_weight"].astype(float)
        result.index.name = None
        return result.rename("current_weight")

    def provenance(self) -> dict[str, Any]:
        event_status_counts = {
            str(key): int(value)
            for key, value in self.rebalance_events.get(
                "status", pd.Series(dtype="object")
            )
            .value_counts()
            .items()
        }
        return {
            **self.config.to_dict(),
            "target_weights": {
                key: float(value) for key, value in self.target_weights.items()
            },
            "launch_date": self.portfolio.index[0].date().isoformat(),
            "latest_finalized_date": self.latest_date.date().isoformat(),
            "risk_base_value": self.latest_net_nav,
            "risk_base_date": self.latest_date.date().isoformat(),
            "risk_base_type": "current_net_nav",
            "rebalance_event_status_counts": event_status_counts,
            "warnings": list(self.warnings),
        }


def _validated_inputs(
    prices: pd.DataFrame, weights: pd.Series | Mapping[str, float]
) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must use a DatetimeIndex")
    frame = prices.copy()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    normalized = index.normalize()
    if normalized.duplicated().any():
        raise ValueError("prices contain duplicate UTC dates")
    frame.index = normalized
    frame = frame.sort_index()
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    if len(frame.columns) != len(set(frame.columns)):
        raise ValueError("price columns must be unique after symbol normalization")
    values = frame.apply(pd.to_numeric, errors="coerce")
    non_missing = values.stack()
    if non_missing.empty or not np.isfinite(non_missing.to_numpy(dtype=float)).all():
        raise ValueError("non-missing prices must be finite")
    if (non_missing <= 0.0).any():
        raise ValueError("non-missing prices must be positive")

    target = pd.Series(weights, dtype=float).copy()
    target.index = [str(asset).strip().upper() for asset in target.index]
    if len(target) != len(set(target.index)):
        raise ValueError("target weights contain duplicate symbols")
    target = target.reindex(values.columns)
    if target.isna().any():
        raise ValueError("target weights must match every price column exactly")
    if not np.isfinite(target.to_numpy()).all():
        raise ValueError("target weights must be finite")
    if (target < 0.0).any():
        raise ValueError(
            "stateful portfolio policies require long-only weights; short-sale "
            "financing, margin, borrow fees and cash flows are not modeled"
        )
    if not math.isclose(
        float(target.sum()), 1.0, rel_tol=0.0, abs_tol=_WEIGHT_TOLERANCE
    ):
        raise ValueError("target weights must sum to 1.0 without leverage")
    if values.iloc[0].isna().any():
        missing = values.columns[values.iloc[0].isna()].tolist()
        raise ValueError("launch prices are incomplete for: " + ", ".join(missing))
    return values.astype(float), target.astype(float)


def _scheduled_dates(
    index: pd.DatetimeIndex, policy: PortfolioEvolutionPolicy
) -> set[pd.Timestamp]:
    """Return theoretical UTC calendar boundaries, independent of observations."""
    if policy is PortfolioEvolutionPolicy.BUY_AND_HOLD:
        return set()
    if policy is PortfolioEvolutionPolicy.DAILY_REBALANCE:
        return set(index[1:])

    launch = pd.Timestamp(index.min()).normalize()
    terminal = index.max()
    if policy is PortfolioEvolutionPolicy.WEEKLY_REBALANCE:
        periods = pd.period_range(start=launch, end=terminal, freq="W-SUN")
    elif policy is PortfolioEvolutionPolicy.MONTHLY_REBALANCE:
        periods = pd.period_range(start=launch, end=terminal, freq="M")
    elif policy is PortfolioEvolutionPolicy.QUARTERLY_REBALANCE:
        periods = pd.period_range(start=launch, end=terminal, freq="Q-DEC")
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"unsupported policy {policy.value}")
    return {
        pd.Timestamp(period.end_time).normalize()
        for period in periods
        if launch < pd.Timestamp(period.end_time).normalize() <= terminal
    }


def _solve_post_cost_nav(
    pre_trade_nav: float,
    current_values: np.ndarray,
    target_weights: np.ndarray,
    cost_rate: float,
) -> tuple[float, np.ndarray, float]:
    """Solve ``post_nav = pre_nav - rate * |target*post_nav-current|_1``."""
    if cost_rate == 0.0:
        post_nav = pre_trade_nav
    else:
        low, high = 0.0, pre_trade_nav
        for _ in range(160):
            midpoint = 0.5 * (low + high)
            gross_notional = float(
                np.abs(target_weights * midpoint - current_values).sum()
            )
            value = midpoint + cost_rate * gross_notional - pre_trade_nav
            if value > 0.0:
                high = midpoint
            else:
                low = midpoint
        post_nav = 0.5 * (low + high)
    trades = target_weights * post_nav - current_values
    gross_notional = float(np.abs(trades).sum())
    transaction_cost = cost_rate * gross_notional
    if not math.isclose(
        post_nav + transaction_cost,
        pre_trade_nav,
        rel_tol=1e-11,
        abs_tol=max(_ACCOUNTING_TOLERANCE, pre_trade_nav * 1e-11),
    ):
        raise RuntimeError("post-cost NAV equation did not converge")
    return post_nav, trades, transaction_cost


def build_portfolio_path(
    prices: pd.DataFrame,
    weights: pd.Series | Mapping[str, float],
    config: PortfolioPathConfig,
) -> PortfolioPathResult:
    """Build one deterministic portfolio path from daily close observations."""
    frame, target = _validated_inputs(prices, weights)
    schedule = _scheduled_dates(frame.index, config.policy)
    target_values = target.to_numpy(dtype=float)
    launch_prices = frame.iloc[0].to_numpy(dtype=float)
    net_quantities = config.initial_capital * target_values / launch_prices
    gross_quantities = net_quantities.copy()
    previous_net_nav = config.initial_capital
    previous_gross_nav = config.initial_capital
    cumulative_cost = 0.0
    net_peak = config.initial_capital
    gross_peak = config.initial_capital
    pending_schedules: list[pd.Timestamp] = []
    recognized_schedules: set[pd.Timestamp] = set()
    event_records: dict[pd.Timestamp, dict[str, Any]] = {}
    warnings: list[str] = []
    portfolio_rows: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []

    for position, (timestamp, row) in enumerate(frame.iterrows()):
        date_key = pd.Timestamp(timestamp)
        prices_today = row.to_numpy(dtype=float)
        complete = bool(np.isfinite(prices_today).all())
        newly_due = sorted(
            boundary
            for boundary in schedule
            if boundary <= date_key and boundary not in recognized_schedules
        )
        for boundary in newly_due:
            recognized_schedules.add(boundary)
            pending_schedules.append(boundary)
            if date_key > boundary:
                status = "pending_missing_boundary"
                reason = "scheduled boundary absent from the observed price index"
                warnings.append(
                    f"Rebalance scheduled for {boundary.date()} was deferred because "
                    "the boundary date was absent from the price index."
                )
            elif not complete:
                status = "pending_incomplete_prices"
                reason = "scheduled boundary has incomplete cross-asset prices"
                warnings.append(
                    f"Rebalance scheduled for {boundary.date()} was deferred because "
                    "prices were incomplete."
                )
            else:
                status = "scheduled_complete"
                reason = ""
            event_records[boundary] = {
                "scheduled_rebalance_date": boundary,
                "effective_rebalance_date": pd.NaT,
                "status": status,
                "defer_reason": reason,
                "transaction_cost_charged": 0.0,
            }

        scheduled_date = pending_schedules[0] if pending_schedules else pd.NaT
        effective_rebalance = bool(
            position > 0
            and complete
            and pending_schedules
            and config.policy.is_rebalanced
        )

        if not complete:
            for index, asset in enumerate(frame.columns):
                price = prices_today[index]
                asset_rows.append(
                    {
                        "date": date_key,
                        "asset": asset,
                        "price": price if math.isfinite(price) else np.nan,
                        "quantity": net_quantities[index],
                        "market_value": (
                            net_quantities[index] * price
                            if math.isfinite(price)
                            else np.nan
                        ),
                        "pre_trade_weight": np.nan,
                        "post_trade_weight": np.nan,
                        "target_weight": target_values[index],
                        "pre_trade_weight_drift": np.nan,
                        "weight_drift": np.nan,
                        "trade_quantity": 0.0,
                        "trade_value": 0.0,
                    }
                )
            portfolio_rows.append(
                {
                    "date": date_key,
                    "finalized": False,
                    "is_launch": False,
                    "gross_nav": np.nan,
                    "net_nav": np.nan,
                    "gross_return": np.nan,
                    "net_return": np.nan,
                    "cumulative_gross_return": np.nan,
                    "cumulative_net_return": np.nan,
                    "running_peak": np.nan,
                    "gross_running_peak": np.nan,
                    "drawdown": np.nan,
                    "gross_drawdown": np.nan,
                    "turnover": 0.0,
                    "gross_turnover": 0.0,
                    "one_way_turnover": 0.0,
                    "gross_traded_notional": 0.0,
                    "transaction_cost": 0.0,
                    "cumulative_transaction_cost": cumulative_cost,
                    "rebalance_flag": False,
                    "scheduled_rebalance_date": scheduled_date,
                    "effective_rebalance_date": pd.NaT,
                    "total_pre_trade_drift": np.nan,
                    "total_post_trade_drift": np.nan,
                    "data_quality_status": (
                        "rebalance_deferred" if pending_schedules else "incomplete"
                    ),
                }
            )
            continue

        net_pre_values = net_quantities * prices_today
        gross_pre_values = gross_quantities * prices_today
        net_pre_nav = float(net_pre_values.sum())
        gross_pre_nav = float(gross_pre_values.sum())
        if net_pre_nav <= 0.0 or gross_pre_nav <= 0.0:
            raise ValueError(f"portfolio NAV is non-positive on {date_key.date()}")
        pre_weights = net_pre_values / net_pre_nav
        trade_values = np.zeros(len(target), dtype=float)
        transaction_cost = 0.0
        gross_notional = 0.0

        if effective_rebalance:
            executing_schedules = tuple(pending_schedules)
            pending_was_deferred = any(
                event_records[boundary]["status"] != "scheduled_complete"
                for boundary in executing_schedules
            )
            net_nav, trade_values, transaction_cost = _solve_post_cost_nav(
                net_pre_nav, net_pre_values, target_values, config.cost_rate
            )
            net_quantities = (net_pre_values + trade_values) / prices_today
            gross_quantities = target_values * gross_pre_nav / prices_today
            gross_notional = float(np.abs(trade_values).sum())
            effective_date: pd.Timestamp | pd.NaTType = date_key
            for event_position, boundary in enumerate(executing_schedules):
                event = event_records[boundary]
                event["effective_rebalance_date"] = date_key
                if event_position == 0:
                    event["status"] = (
                        "executed_on_schedule"
                        if boundary == date_key and not pending_was_deferred
                        else "deferred_executed"
                    )
                    event["transaction_cost_charged"] = transaction_cost
                else:
                    event["status"] = "coalesced_without_additional_trade"
            if len(executing_schedules) > 1:
                warnings.append(
                    f"{len(executing_schedules)} pending rebalance boundaries were "
                    f"coalesced into one execution on {date_key.date()}; transaction "
                    "cost was charged once."
                )
            pending_schedules.clear()
        else:
            net_nav = net_pre_nav
            pending_was_deferred = False
            effective_date = pd.NaT

        net_post_values = net_quantities * prices_today
        post_weights = net_post_values / net_nav
        if not math.isclose(
            float(net_post_values.sum()),
            net_nav,
            rel_tol=1e-11,
            abs_tol=max(_ACCOUNTING_TOLERANCE, net_nav * 1e-11),
        ):
            raise RuntimeError(f"net accounting identity failed on {date_key.date()}")
        if not math.isclose(float(post_weights.sum()), 1.0, abs_tol=1e-10):
            raise RuntimeError(
                f"post-trade weights do not sum to one on {date_key.date()}"
            )
        if not math.isclose(
            float((gross_quantities * prices_today).sum()),
            gross_pre_nav,
            rel_tol=1e-11,
            abs_tol=max(_ACCOUNTING_TOLERANCE, gross_pre_nav * 1e-11),
        ):
            raise RuntimeError(f"gross accounting identity failed on {date_key.date()}")

        cumulative_cost += transaction_cost
        gross_return = (
            0.0 if position == 0 else gross_pre_nav / previous_gross_nav - 1.0
        )
        net_return = 0.0 if position == 0 else net_nav / previous_net_nav - 1.0
        gross_peak = max(gross_peak, gross_pre_nav)
        net_peak = max(net_peak, net_nav)
        gross_turnover = gross_notional / net_pre_nav
        total_pre_drift = 0.5 * float(np.abs(pre_weights - target_values).sum())
        total_post_drift = 0.5 * float(np.abs(post_weights - target_values).sum())

        for index, asset in enumerate(frame.columns):
            asset_rows.append(
                {
                    "date": date_key,
                    "asset": asset,
                    "price": prices_today[index],
                    "quantity": net_quantities[index],
                    "market_value": net_post_values[index],
                    "pre_trade_weight": pre_weights[index],
                    "post_trade_weight": post_weights[index],
                    "target_weight": target_values[index],
                    "pre_trade_weight_drift": pre_weights[index] - target_values[index],
                    "weight_drift": post_weights[index] - target_values[index],
                    "trade_quantity": trade_values[index] / prices_today[index],
                    "trade_value": trade_values[index],
                }
            )
        portfolio_rows.append(
            {
                "date": date_key,
                "finalized": True,
                "is_launch": position == 0,
                "gross_nav": gross_pre_nav,
                "net_nav": net_nav,
                "gross_return": gross_return,
                "net_return": net_return,
                "cumulative_gross_return": gross_pre_nav / config.initial_capital - 1.0,
                "cumulative_net_return": net_nav / config.initial_capital - 1.0,
                "running_peak": net_peak,
                "gross_running_peak": gross_peak,
                "drawdown": net_nav / net_peak - 1.0,
                "gross_drawdown": gross_pre_nav / gross_peak - 1.0,
                "turnover": gross_turnover,
                "gross_turnover": gross_turnover,
                "one_way_turnover": 0.5 * gross_turnover,
                "gross_traded_notional": gross_notional,
                "transaction_cost": transaction_cost,
                "cumulative_transaction_cost": cumulative_cost,
                "rebalance_flag": effective_rebalance,
                "scheduled_rebalance_date": scheduled_date,
                "effective_rebalance_date": effective_date,
                "total_pre_trade_drift": total_pre_drift,
                "total_post_trade_drift": total_post_drift,
                "data_quality_status": (
                    "deferred_rebalance_executed"
                    if pending_was_deferred
                    else "complete"
                ),
            }
        )
        previous_net_nav = net_nav
        previous_gross_nav = gross_pre_nav

    portfolio = pd.DataFrame(portfolio_rows).set_index("date")
    portfolio.index.name = "date"
    assets = pd.DataFrame(asset_rows).set_index(["date", "asset"]).sort_index()
    rebalance_events = pd.DataFrame(
        list(event_records.values()),
        columns=[
            "scheduled_rebalance_date",
            "effective_rebalance_date",
            "status",
            "defer_reason",
            "transaction_cost_charged",
        ],
    )
    return PortfolioPathResult(
        portfolio=portfolio,
        assets=assets,
        rebalance_events=rebalance_events,
        target_weights=target.rename("target_weight"),
        config=config,
        warnings=tuple(warnings),
    )


def drawdown_statistics(nav: pd.Series) -> dict[str, Any]:
    """Return maximum drawdown, longest underwater duration and max-DD recovery."""
    clean = nav.dropna().astype(float)
    if clean.empty:
        return {
            "maximum_drawdown": 0.0,
            "trough_date": None,
            "longest_underwater_days": 0,
            "recovery_date": None,
            "last_episode_recovery_date": None,
        }
    peak = clean.cummax()
    drawdown = clean / peak - 1.0
    longest = 0
    current_start: pd.Timestamp | None = None
    last_episode_recovery: pd.Timestamp | None = None
    for timestamp, value in drawdown.items():
        when = pd.Timestamp(timestamp)
        if value < -1e-14 and current_start is None:
            current_start = when
        elif value >= -1e-14 and current_start is not None:
            longest = max(longest, (when - current_start).days)
            last_episode_recovery = when
            current_start = None
    if current_start is not None:
        longest = max(longest, (pd.Timestamp(clean.index[-1]) - current_start).days)
        last_episode_recovery = None

    maximum_drawdown = float(drawdown.min())
    trough_date = pd.Timestamp(drawdown.idxmin())
    recovery: pd.Timestamp | None = None
    if maximum_drawdown < -1e-14:
        peak_value = float(peak.loc[trough_date])
        after_trough = clean.loc[trough_date:]
        recovered = after_trough[after_trough >= peak_value * (1.0 - 1e-14)]
        if not recovered.empty:
            recovery = pd.Timestamp(recovered.index[0])
    return {
        "maximum_drawdown": maximum_drawdown,
        "trough_date": trough_date,
        "longest_underwater_days": int(longest),
        "recovery_date": recovery,
        "last_episode_recovery_date": last_episode_recovery,
    }


__all__ = [
    "PORTFOLIO_PATH_VERSION",
    "PortfolioEvolutionPolicy",
    "PortfolioPathConfig",
    "PortfolioPathResult",
    "build_portfolio_path",
    "drawdown_statistics",
]
