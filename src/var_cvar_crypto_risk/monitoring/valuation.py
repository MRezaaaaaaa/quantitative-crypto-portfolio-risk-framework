"""Deterministic fixed-quantity valuation for model portfolios.

The functions in this module are pure: they do not reveal historical data
sequentially, fetch live prices, rebalance, or write a database.  Batch 4 and
Batch 5 orchestration will control which already-normalized observations may be
passed here.
"""

from __future__ import annotations

from datetime import date
import math

import pandas as pd

from .domain import (
    DailyAssetState,
    DailyPortfolioState,
    DataQualityStatus,
    DomainValidationError,
    Experiment,
    OptimizationSnapshot,
    SnapshotAllocation,
)
from .prices import NormalizedPriceData, missing_symbols_on_date
from .recipes import CashPolicy


def cash_value_at_date(
    initial_cash: float,
    *,
    launch_date: date,
    state_date: date,
    policy: CashPolicy,
) -> float:
    """Derive cash directly from launch and elapsed calendar days.

    No previous stored value is used, so repeating a calculation cannot
    double-compound cash.
    """
    initial = float(initial_cash)
    if not math.isfinite(initial):
        raise DomainValidationError("initial_cash must be finite")
    elapsed = (state_date - launch_date).days
    if elapsed < 0:
        raise DomainValidationError("state_date must not precede launch_date")
    return initial * (1.0 + policy.horizon_return(elapsed))


def _validate_snapshot_for_valuation(
    experiment: Experiment, snapshot: OptimizationSnapshot, cash_policy: CashPolicy
) -> tuple[tuple[SnapshotAllocation, ...], tuple[SnapshotAllocation, ...]]:
    if snapshot.experiment_id != experiment.experiment_id:
        raise DomainValidationError("snapshot belongs to a different experiment")
    if snapshot.activated_at is None:
        raise DomainValidationError("snapshot must be activated before valuation")
    if experiment.launch_date is None:
        raise DomainValidationError("experiment launch_date is required")
    allocations = snapshot.allocations
    market = tuple(item for item in allocations if not item.is_cash)
    cash = tuple(item for item in allocations if item.is_cash)
    if len(cash) > 1:
        raise DomainValidationError("snapshot may contain at most one cash allocation")
    if bool(cash) != cash_policy.enabled:
        raise DomainValidationError(
            "snapshot cash allocation does not match the frozen cash policy"
        )
    if cash and cash[0].asset != cash_policy.symbol:
        raise DomainValidationError(
            "cash allocation symbol does not match the frozen cash policy"
        )
    total_initial = math.fsum(item.initial_value for item in allocations)
    if not math.isclose(
        total_initial,
        experiment.initial_capital,
        rel_tol=0.0,
        abs_tol=max(1e-8, experiment.initial_capital * 1e-9),
    ):
        raise DomainValidationError(
            "snapshot initial allocation values do not equal initial capital"
        )
    for allocation in market:
        if allocation.launch_price is None:
            raise DomainValidationError(
                f"market allocation {allocation.asset} lacks launch_price"
            )
        expected_quantity = allocation.initial_value / allocation.launch_price
        if not math.isclose(
            expected_quantity,
            allocation.quantity,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise DomainValidationError(
                f"market allocation {allocation.asset} quantity is inconsistent"
            )
    return market, cash


def _benchmark_values(
    *,
    normalized: NormalizedPriceData,
    benchmark_symbol: str | None,
    launch_date: date,
    state_date: date,
    initial_capital: float,
    previous_nav: float | None,
    previous_date: date | None,
) -> tuple[float | None, float | None, int | None]:
    if benchmark_symbol is None:
        return None, None, None
    symbol = benchmark_symbol.strip().upper()
    if symbol not in normalized.prices.columns:
        return None, None, None
    launch_key = pd.Timestamp(launch_date)
    state_key = pd.Timestamp(state_date)
    if launch_key not in normalized.prices.index or state_key not in normalized.prices.index:
        return None, None, None
    launch_price = normalized.prices.at[launch_key, symbol]
    current_price = normalized.prices.at[state_key, symbol]
    if pd.isna(launch_price) or pd.isna(current_price):
        return None, None, None
    nav = initial_capital * float(current_price) / float(launch_price)
    if state_date == launch_date:
        return nav, 0.0, 0
    result_return = None if previous_nav is None else nav / previous_nav - 1.0
    interval_days = None if previous_date is None else (state_date - previous_date).days
    return nav, result_return, interval_days


def value_fixed_holdings(
    *,
    experiment: Experiment,
    snapshot: OptimizationSnapshot,
    normalized: NormalizedPriceData,
    cash_policy: CashPolicy,
    calculation_version: str,
) -> tuple[DailyPortfolioState, ...]:
    """Calculate deterministic states for every supplied date from launch.

    The caller owns information revelation. Passing a full future frame here is
    acceptable only for deterministic valuation, never for re-estimation or
    forecast construction. Missing market prices produce a non-finalized state
    with no fabricated NAV and do not alter fixed quantities.
    """
    market_allocations, cash_allocations = _validate_snapshot_for_valuation(
        experiment, snapshot, cash_policy
    )
    launch_date = experiment.launch_date
    assert launch_date is not None
    launch_key = pd.Timestamp(launch_date)
    if launch_key not in normalized.prices.index:
        raise DomainValidationError("launch date is absent from monitoring prices")
    symbols = tuple(item.asset for item in market_allocations)
    missing_at_launch = missing_symbols_on_date(normalized, launch_date, symbols)
    if missing_at_launch:
        raise DomainValidationError(
            "launch observation is incomplete; missing: "
            + ", ".join(missing_at_launch)
        )
    if experiment.benchmark_symbol is not None:
        missing_benchmark = missing_symbols_on_date(
            normalized, launch_date, (experiment.benchmark_symbol,)
        )
        if missing_benchmark:
            raise DomainValidationError(
                "launch benchmark observation is incomplete; missing: "
                + ", ".join(missing_benchmark)
            )
    for allocation in market_allocations:
        actual = float(normalized.prices.at[launch_key, allocation.asset])
        assert allocation.launch_price is not None
        if not math.isclose(actual, allocation.launch_price, rel_tol=1e-12, abs_tol=1e-12):
            raise DomainValidationError(
                f"launch price mismatch for {allocation.asset}: snapshot provenance "
                "does not match valuation input"
            )

    rows = normalized.prices.loc[normalized.prices.index >= launch_key]
    if rows.empty:
        raise DomainValidationError("no monitoring observations exist at or after launch")

    initial_cash = cash_allocations[0].initial_value if cash_allocations else 0.0
    previous_complete_date: date | None = None
    previous_complete_nav: float | None = None
    running_peak: float | None = None
    maximum_drawdown = 0.0
    previous_benchmark_nav: float | None = None
    previous_benchmark_date: date | None = None
    eligible_daily_returns: list[float] = []
    states: list[DailyPortfolioState] = []

    for timestamp, row in rows.iterrows():
        state_date = timestamp.date()
        cash_value = cash_value_at_date(
            initial_cash,
            launch_date=launch_date,
            state_date=state_date,
            policy=cash_policy,
        )
        missing_assets = tuple(
            allocation.asset
            for allocation in market_allocations
            if pd.isna(row.get(allocation.asset))
        )
        benchmark_nav, benchmark_return, benchmark_interval_days = _benchmark_values(
            normalized=normalized,
            benchmark_symbol=experiment.benchmark_symbol,
            launch_date=launch_date,
            state_date=state_date,
            initial_capital=experiment.initial_capital,
            previous_nav=previous_benchmark_nav,
            previous_date=previous_benchmark_date,
        )
        if benchmark_nav is not None:
            previous_benchmark_nav = benchmark_nav
            previous_benchmark_date = state_date

        if missing_assets:
            incomplete_assets: list[DailyAssetState] = []
            for allocation in market_allocations:
                raw_price = row.get(allocation.asset)
                price = None if pd.isna(raw_price) else float(raw_price)
                value = None if price is None else allocation.quantity * price
                incomplete_assets.append(
                    DailyAssetState(
                        experiment_id=experiment.experiment_id,
                        state_date=state_date,
                        asset=allocation.asset,
                        price=price,
                        quantity=allocation.quantity,
                        market_value=value,
                        target_weight=allocation.target_weight,
                    )
                )
            if cash_allocations:
                cash_allocation = cash_allocations[0]
                incomplete_assets.append(
                    DailyAssetState(
                        experiment_id=experiment.experiment_id,
                        state_date=state_date,
                        asset=cash_allocation.asset,
                        quantity=cash_allocation.quantity,
                        market_value=cash_value,
                        target_weight=cash_allocation.target_weight,
                        is_cash=True,
                    )
                )
            states.append(
                DailyPortfolioState(
                    experiment_id=experiment.experiment_id,
                    state_date=state_date,
                    data_quality_status=DataQualityStatus.INCOMPLETE,
                    calculation_version=calculation_version,
                    finalized=False,
                    asset_states=tuple(incomplete_assets),
                    benchmark_nav=benchmark_nav,
                    benchmark_return=benchmark_return,
                    quality_metadata={
                        "missing_assets": list(missing_assets),
                        "source": normalized.source,
                        "source_fingerprint": normalized.fingerprint,
                        "nav_fabricated": False,
                        "benchmark_return_interval_days": benchmark_interval_days,
                    },
                )
            )
            continue

        raw_values = {
            allocation.asset: allocation.quantity * float(row[allocation.asset])
            for allocation in market_allocations
        }
        nav = cash_value + math.fsum(raw_values.values())
        if not math.isfinite(nav) or nav <= 0.0:
            raise DomainValidationError(
                f"portfolio NAV is non-positive or non-finite on {state_date}"
            )
        interval_days = (
            0
            if previous_complete_date is None
            else (state_date - previous_complete_date).days
        )
        if state_date == launch_date:
            daily_return = 0.0
        elif previous_complete_nav is None:
            daily_return = None
        else:
            daily_return = nav / previous_complete_nav - 1.0
        if daily_return is None:
            raise DomainValidationError(
                "complete post-launch state lacks a previous complete valuation"
            )
        if interval_days == 1:
            eligible_daily_returns.append(daily_return)
        realized_volatility = (
            float(pd.Series(eligible_daily_returns, dtype=float).std(ddof=1))
            * math.sqrt(365.0)
            if len(eligible_daily_returns) >= 2
            else None
        )
        running_peak = nav if running_peak is None else max(running_peak, nav)
        drawdown = nav / running_peak - 1.0
        maximum_drawdown = min(maximum_drawdown, drawdown)

        complete_assets: list[DailyAssetState] = []
        for allocation in market_allocations:
            market_value = raw_values[allocation.asset]
            current_weight = market_value / nav
            complete_assets.append(
                DailyAssetState(
                    experiment_id=experiment.experiment_id,
                    state_date=state_date,
                    asset=allocation.asset,
                    price=float(row[allocation.asset]),
                    quantity=allocation.quantity,
                    market_value=market_value,
                    target_weight=allocation.target_weight,
                    current_weight=current_weight,
                    drift_percentage_points=current_weight
                    - allocation.target_weight,
                )
            )
        if cash_allocations:
            allocation = cash_allocations[0]
            current_weight = cash_value / nav
            complete_assets.append(
                DailyAssetState(
                    experiment_id=experiment.experiment_id,
                    state_date=state_date,
                    asset=allocation.asset,
                    quantity=allocation.quantity,
                    market_value=cash_value,
                    target_weight=allocation.target_weight,
                    current_weight=current_weight,
                    drift_percentage_points=current_weight
                    - allocation.target_weight,
                    is_cash=True,
                )
            )
        total_drift = 0.5 * math.fsum(
            abs(float(item.current_weight) - item.target_weight)
            for item in complete_assets
        )
        state = DailyPortfolioState(
            experiment_id=experiment.experiment_id,
            state_date=state_date,
            data_quality_status=DataQualityStatus.COMPLETE,
            calculation_version=calculation_version,
            finalized=True,
            asset_states=tuple(complete_assets),
            nav=nav,
            base_100_nav=100.0 * nav / experiment.initial_capital,
            cash_value=cash_value,
            daily_return=daily_return,
            cumulative_return=nav / experiment.initial_capital - 1.0,
            realized_volatility=realized_volatility,
            running_peak=running_peak,
            drawdown=drawdown,
            maximum_drawdown=maximum_drawdown,
            total_drift=total_drift,
            return_interval_days=interval_days,
            benchmark_nav=benchmark_nav,
            benchmark_return=benchmark_return,
            quality_metadata={
                "missing_assets": [],
                "source": normalized.source,
                "source_fingerprint": normalized.fingerprint,
                "return_interval_days": interval_days,
                "is_one_day_return": interval_days in {0, 1},
                "benchmark_return_interval_days": benchmark_interval_days,
            },
        )
        states.append(state)
        previous_complete_date = state_date
        previous_complete_nav = nav

    launch_state = next((item for item in states if item.state_date == launch_date), None)
    if launch_state is None or launch_state.nav is None or not math.isclose(
        launch_state.nav,
        experiment.initial_capital,
        rel_tol=0.0,
        abs_tol=max(1e-8, experiment.initial_capital * 1e-9),
    ):
        raise DomainValidationError("launch NAV must equal initial capital")
    return tuple(states)


__all__ = ["cash_value_at_date", "value_fixed_holdings"]
