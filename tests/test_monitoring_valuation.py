"""Fixed-quantity NAV, drift, drawdown, cash, and quality tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from var_cvar_crypto_risk.monitoring.domain import (
    DataQualityStatus,
    DomainValidationError,
    Experiment,
    ExperimentMode,
    OptimizationSnapshot,
    SnapshotAllocation,
)
from var_cvar_crypto_risk.monitoring.prices import normalize_monitoring_prices
from var_cvar_crypto_risk.monitoring.recipes import CashPolicy
from var_cvar_crypto_risk.monitoring.valuation import (
    cash_value_at_date,
    value_fixed_holdings,
)


ACTIVATED_AT = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def _experiment(*, benchmark: str | None = "BENCH") -> Experiment:
    return Experiment.create(
        name="valuation",
        mode=ExperimentMode.HISTORICAL_OOS,
        base_currency="USD",
        initial_capital=1_000.0,
        benchmark_symbol=benchmark,
        training_start=date(2025, 1, 1),
        training_end=date(2025, 12, 31),
        optimization_as_of=date(2025, 12, 31),
        launch_date=date(2026, 1, 1),
        historical_evaluation_end=date(2026, 12, 31),
    )


def _snapshot(experiment: Experiment, *, with_cash: bool = True) -> OptimizationSnapshot:
    allocations = [
        SnapshotAllocation(
            asset="BTC",
            asset_type="crypto",
            target_weight=0.6 if with_cash else 0.7,
            launch_price=100.0,
            initial_value=600.0 if with_cash else 700.0,
            quantity=6.0 if with_cash else 7.0,
        ),
        SnapshotAllocation(
            asset="ETH",
            asset_type="crypto",
            target_weight=0.3,
            launch_price=50.0,
            initial_value=300.0,
            quantity=6.0,
        ),
    ]
    if with_cash:
        allocations.append(
            SnapshotAllocation(
                asset="CASH",
                asset_type="cash",
                target_weight=0.1,
                launch_price=None,
                initial_value=100.0,
                quantity=100.0,
                is_cash=True,
            )
        )
    snapshot = OptimizationSnapshot(
        snapshot_id=uuid4(),
        experiment_id=experiment.experiment_id,
        package_version="1.0.0",
        code_version="abc123",
        objective="min_cvar",
        solver="CLARABEL",
        solver_status="optimal",
        source_data_hash="a" * 64,
        assumption_recipe_hash="b" * 64,
        assumptions={},
        constraints={},
        launch_forecast={},
        scenario_metadata={},
        return_policy={"wealth": "simple"},
        loss_convention={"name": "signed_loss_space"},
        residual_validation={"passed": True},
        allocations=tuple(allocations),
    )
    return snapshot.activate(at=ACTIVATED_AT)


def _prices():
    frame = pd.DataFrame(
        {
            "BTC": [100.0, 110.0, 120.0, 90.0],
            "ETH": [50.0, 45.0, np.nan, 60.0],
            "BENCH": [200.0, 210.0, 220.0, 205.0],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="D"),
    )
    return normalize_monitoring_prices(
        frame, source="fixture", retrieved_at=ACTIVATED_AT
    )


def test_launch_nav_quantities_weights_and_return_contract() -> None:
    experiment = _experiment()
    states = value_fixed_holdings(
        experiment=experiment,
        snapshot=_snapshot(experiment),
        normalized=_prices(),
        cash_policy=CashPolicy(enabled=True),
        calculation_version="valuation-v1",
    )
    launch = states[0]
    assert launch.nav == pytest.approx(1_000.0)
    assert launch.base_100_nav == pytest.approx(100.0)
    assert launch.daily_return == 0.0
    assert launch.cumulative_return == 0.0
    assert launch.return_interval_days == 0
    assert sum(item.current_weight for item in launch.asset_states) == pytest.approx(1)
    assert {item.asset: item.quantity for item in launch.asset_states} == {
        "BTC": 6.0,
        "ETH": 6.0,
        "CASH": 100.0,
    }


def test_market_moves_change_values_and_weights_but_not_quantities() -> None:
    experiment = _experiment()
    states = value_fixed_holdings(
        experiment=experiment,
        snapshot=_snapshot(experiment),
        normalized=_prices(),
        cash_policy=CashPolicy(enabled=True),
        calculation_version="valuation-v1",
    )
    second = states[1]
    assert second.nav == pytest.approx(1_030.0)
    assert second.daily_return == pytest.approx(0.03)
    assert second.return_interval_days == 1
    assert second.benchmark_nav == pytest.approx(1_050.0)
    assert second.benchmark_return == pytest.approx(0.05)
    assert [item.quantity for item in second.asset_states] == [6.0, 6.0, 100.0]
    expected_drift = 0.5 * sum(
        abs(item.current_weight - item.target_weight) for item in second.asset_states
    )
    assert second.total_drift == pytest.approx(expected_drift)


def test_missing_asset_records_incomplete_state_without_fabricated_nav() -> None:
    experiment = _experiment()
    states = value_fixed_holdings(
        experiment=experiment,
        snapshot=_snapshot(experiment),
        normalized=_prices(),
        cash_policy=CashPolicy(enabled=True),
        calculation_version="valuation-v1",
    )
    incomplete = states[2]
    assert incomplete.data_quality_status is DataQualityStatus.INCOMPLETE
    assert incomplete.finalized is False
    assert incomplete.nav is None
    assert incomplete.quality_metadata["missing_assets"] == ["ETH"]
    eth = next(item for item in incomplete.asset_states if item.asset == "ETH")
    assert eth.price is None and eth.market_value is None


def test_post_gap_return_uses_previous_complete_nav_and_labels_interval() -> None:
    experiment = _experiment()
    states = value_fixed_holdings(
        experiment=experiment,
        snapshot=_snapshot(experiment),
        normalized=_prices(),
        cash_policy=CashPolicy(enabled=True),
        calculation_version="valuation-v1",
    )
    final = states[3]
    assert final.nav == pytest.approx(1_000.0)
    assert final.daily_return == pytest.approx(1_000.0 / 1_030.0 - 1.0)
    assert final.return_interval_days == 2
    assert final.quality_metadata["is_one_day_return"] is False
    assert final.running_peak == pytest.approx(1_030.0)
    assert final.drawdown == pytest.approx(1_000.0 / 1_030.0 - 1.0)
    assert final.maximum_drawdown == pytest.approx(final.drawdown)
    assert final.realized_volatility is None


def test_realized_volatility_is_expanding_origin_safe_and_annualized() -> None:
    experiment = _experiment()
    frame = _prices().prices.copy()
    frame.loc["2026-01-03", "ETH"] = 55.0
    normalized = normalize_monitoring_prices(
        frame, source="fixture", retrieved_at=ACTIVATED_AT
    )
    states = value_fixed_holdings(
        experiment=experiment,
        snapshot=_snapshot(experiment),
        normalized=normalized,
        cash_policy=CashPolicy(enabled=True),
        calculation_version="valuation-v1",
    )

    assert states[0].realized_volatility is None
    assert states[1].realized_volatility is None
    eligible = pd.Series([states[1].daily_return, states[2].daily_return])
    assert states[2].realized_volatility == pytest.approx(
        eligible.std(ddof=1) * np.sqrt(365.0)
    )
    all_eligible = pd.Series([item.daily_return for item in states[1:]])
    assert states[3].realized_volatility == pytest.approx(
        all_eligible.std(ddof=1) * np.sqrt(365.0)
    )


def test_cash_is_derived_from_launch_and_is_repeatable() -> None:
    policy = CashPolicy(enabled=True, mode="annual_rate", annual_rate=0.10)
    value = cash_value_at_date(
        100.0,
        launch_date=date(2025, 1, 1),
        state_date=date(2026, 1, 1),
        policy=policy,
    )
    assert value == pytest.approx(110.0)
    assert cash_value_at_date(
        100.0,
        launch_date=date(2025, 1, 1),
        state_date=date(2026, 1, 1),
        policy=policy,
    ) == pytest.approx(value)


def test_snapshot_cash_presence_must_match_frozen_policy() -> None:
    experiment = _experiment()
    with pytest.raises(DomainValidationError, match="cash allocation"):
        value_fixed_holdings(
            experiment=experiment,
            snapshot=_snapshot(experiment),
            normalized=_prices(),
            cash_policy=CashPolicy(enabled=False),
            calculation_version="valuation-v1",
        )


def test_missing_benchmark_at_launch_blocks_activation_valuation() -> None:
    experiment = _experiment()
    frame = _prices().prices.drop(columns="BENCH")
    normalized = normalize_monitoring_prices(
        frame, source="fixture", retrieved_at=ACTIVATED_AT
    )
    with pytest.raises(DomainValidationError, match="launch benchmark"):
        value_fixed_holdings(
            experiment=experiment,
            snapshot=_snapshot(experiment),
            normalized=normalized,
            cash_policy=CashPolicy(enabled=True),
            calculation_version="valuation-v1",
        )


def test_snapshot_launch_price_mismatch_is_rejected() -> None:
    experiment = _experiment()
    snapshot = _snapshot(experiment)
    bad_allocation = replace(snapshot.allocations[0], launch_price=99.0)
    snapshot = replace(snapshot, allocations=(bad_allocation, *snapshot.allocations[1:]))
    with pytest.raises(DomainValidationError, match="quantity is inconsistent"):
        value_fixed_holdings(
            experiment=experiment,
            snapshot=snapshot,
            normalized=_prices(),
            cash_policy=CashPolicy(enabled=True),
            calculation_version="valuation-v1",
        )
