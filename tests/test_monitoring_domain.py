"""Domain and hashing tests for persistent portfolio experiments."""

from __future__ import annotations

from datetime import date, datetime, timezone
import math
from uuid import uuid4

import pytest

from var_cvar_crypto_risk.monitoring.domain import (
    DomainValidationError,
    Experiment,
    ExperimentMode,
    ExperimentStatus,
    ImmutableRecordError,
    InvalidTransitionError,
    OptimizationSnapshot,
    SnapshotAllocation,
    validate_date_boundaries,
)
from var_cvar_crypto_risk.monitoring.hashing import (
    canonical_json,
    sha256_fingerprint,
)


def _allocation(weight: float = 1.0) -> SnapshotAllocation:
    return SnapshotAllocation(
        asset="BTC",
        asset_type="crypto",
        target_weight=weight,
        launch_price=50_000.0,
        initial_value=100_000.0,
        quantity=2.0,
    )


def _snapshot(**overrides) -> OptimizationSnapshot:
    values = {
        "snapshot_id": uuid4(),
        "experiment_id": uuid4(),
        "package_version": "1.0.0",
        "code_version": "abc123",
        "objective": "min_cvar",
        "solver": "CLARABEL",
        "solver_status": "optimal",
        "source_data_hash": "a" * 64,
        "assumption_recipe_hash": "b" * 64,
        "assumptions": {"confidence": 0.95},
        "constraints": {"long_only": True},
        "launch_forecast": {"cvar": 0.1},
        "scenario_metadata": {"source": "historical"},
        "return_policy": {"wealth": "simple"},
        "loss_convention": {"losses": "positive"},
        "residual_validation": {"passed": True},
        "allocations": (_allocation(),),
    }
    values.update(overrides)
    return OptimizationSnapshot(**values)


def test_experiment_identity_name_and_currency_are_normalized() -> None:
    experiment = Experiment.create(
        name="  OOS test  ",
        mode=ExperimentMode.HISTORICAL_OOS,
        base_currency=" usd ",
        initial_capital=100_000.0,
    )
    assert experiment.name == "OOS test"
    assert experiment.base_currency == "USD"
    assert experiment.status is ExperimentStatus.DRAFT
    assert experiment.experiment_id.version == 4


@pytest.mark.parametrize("capital", [0.0, -1.0, math.inf, math.nan])
def test_experiment_rejects_invalid_initial_capital(capital: float) -> None:
    with pytest.raises(DomainValidationError, match="initial_capital"):
        Experiment.create(
            name="invalid",
            mode=ExperimentMode.LIVE_FORWARD,
            base_currency="USD",
            initial_capital=capital,
        )


def test_lifecycle_transition_and_archive_are_explicit() -> None:
    experiment = Experiment.create(
        name="lifecycle",
        mode=ExperimentMode.LIVE_FORWARD,
        base_currency="USD",
        initial_capital=1_000.0,
    )
    active = experiment.transition(ExperimentStatus.ACTIVE)
    archived = active.transition(ExperimentStatus.ARCHIVED)
    assert archived.archived_at == archived.updated_at
    with pytest.raises(InvalidTransitionError):
        archived.transition(ExperimentStatus.ACTIVE)


def test_invalid_lifecycle_jump_is_rejected() -> None:
    experiment = Experiment.create(
        name="invalid jump",
        mode=ExperimentMode.HISTORICAL_OOS,
        base_currency="USD",
        initial_capital=1_000.0,
    )
    with pytest.raises(InvalidTransitionError):
        experiment.transition(ExperimentStatus.COMPLETED)


def test_point_in_time_date_boundaries_are_strict() -> None:
    validate_date_boundaries(
        mode=ExperimentMode.HISTORICAL_OOS,
        training_start=date(2024, 1, 1),
        training_end=date(2024, 12, 31),
        optimization_as_of=date(2025, 1, 1),
        launch_date=date(2025, 1, 2),
        historical_evaluation_end=date(2025, 6, 30),
        live_tracking_end=None,
        require_complete=True,
    )
    with pytest.raises(DomainValidationError, match="launch_date"):
        validate_date_boundaries(
            mode=ExperimentMode.HISTORICAL_OOS,
            training_start=date(2024, 1, 1),
            training_end=date(2024, 12, 31),
            optimization_as_of=date(2025, 1, 2),
            launch_date=date(2025, 1, 2),
            historical_evaluation_end=date(2025, 6, 30),
            live_tracking_end=None,
            require_complete=True,
        )


def test_hybrid_live_end_cannot_precede_historical_boundary() -> None:
    with pytest.raises(DomainValidationError, match="historical boundary"):
        validate_date_boundaries(
            mode=ExperimentMode.HYBRID,
            training_start=date(2024, 1, 1),
            training_end=date(2024, 12, 31),
            optimization_as_of=date(2025, 1, 1),
            launch_date=date(2025, 1, 2),
            historical_evaluation_end=date(2025, 6, 30),
            live_tracking_end=date(2025, 6, 1),
            require_complete=True,
        )


def test_canonical_hash_is_order_independent_but_value_sensitive() -> None:
    left = {"confidence": 0.95, "assets": ["BTC", "ETH"]}
    right = {"assets": ["BTC", "ETH"], "confidence": 0.95}
    changed = {"confidence": 0.99, "assets": ["BTC", "ETH"]}
    assert canonical_json(left) == canonical_json(right)
    assert sha256_fingerprint(left) == sha256_fingerprint(right)
    assert sha256_fingerprint(left) != sha256_fingerprint(changed)
    assert len(sha256_fingerprint(left)) == 64


def test_canonical_hash_normalizes_utc_and_rejects_ambiguous_values() -> None:
    aware = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    assert canonical_json({"as_of": aware}) == '{"as_of":"2026-01-01T12:00:00Z"}'
    with pytest.raises(DomainValidationError, match="timezone-aware"):
        canonical_json({"as_of": datetime(2026, 1, 1, 12)})
    with pytest.raises(DomainValidationError, match="finite"):
        canonical_json({"value": math.nan})
    with pytest.raises(DomainValidationError, match="keys"):
        canonical_json({1: "not portable"})


def test_snapshot_activation_requires_solved_validated_unit_sum() -> None:
    draft = _snapshot()
    activated = draft.activate(at=datetime(2026, 1, 2, tzinfo=timezone.utc))
    assert activated.activated_at is not None
    with pytest.raises(ImmutableRecordError):
        activated.activate()
    with pytest.raises(DomainValidationError, match="solved status"):
        _snapshot(solver_status="failed").activate()
    with pytest.raises(DomainValidationError, match="residual validation"):
        _snapshot(residual_validation={"passed": False}).activate()
    with pytest.raises(DomainValidationError, match="sum to one"):
        _snapshot(allocations=(_allocation(0.8),)).activate()


def test_snapshot_rejects_duplicate_assets_and_invalid_hashes() -> None:
    with pytest.raises(DomainValidationError, match="unique"):
        _snapshot(allocations=(_allocation(0.5), _allocation(0.5)))
    with pytest.raises(DomainValidationError, match="SHA-256"):
        _snapshot(source_data_hash="not-a-hash")


def test_cash_allocation_does_not_accept_market_launch_price() -> None:
    with pytest.raises(DomainValidationError, match="cash allocation"):
        SnapshotAllocation(
            asset="CASH",
            asset_type="cash",
            target_weight=1.0,
            launch_price=1.0,
            initial_value=1_000.0,
            quantity=1_000.0,
            is_cash=True,
        )
