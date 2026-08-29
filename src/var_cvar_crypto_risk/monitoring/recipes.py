"""Serializable, secret-free recipes for portfolio-monitoring experiments.

These values are deliberately independent from Streamlit and SQLAlchemy.  They
capture the full construction recipe that must be frozen before a model
portfolio is launched and can be converted back to the existing analytics
objects without changing financial formulas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit

from ..assumptions import (
    COVARIANCE_METHODS,
    EXPECTED_RETURN_METHODS,
    SHRINKAGE_TARGETS,
    AssumptionConfig,
)
from .domain import DomainValidationError
from .hashing import sha256_fingerprint


_SCENARIO_SOURCES = frozenset({"historical", "normal_mc", "student_t_mc"})
_OBJECTIVES = frozenset(
    {
        "min_cvar",
        "max_return_cvar",
        "min_cvar_target_return",
        "max_sharpe",
    }
)
_CASH_MODES = frozenset({"zero", "annual_rate"})
_EVALUATION_MODES = frozenset({"overlapping", "non_overlapping"})
_FORBIDDEN_METADATA_KEYS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "database_url",
    "password",
    "private_key",
    "secret",
    "token",
)


def _finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise DomainValidationError(f"{name} must be finite")
    return converted


def _probability(value: float, name: str) -> float:
    converted = _finite(value, name)
    if not 0.0 < converted < 1.0:
        raise DomainValidationError(f"{name} must be in (0, 1)")
    return converted


def _unit_interval(value: float, name: str) -> float:
    converted = _finite(value, name)
    if not 0.0 <= converted <= 1.0:
        raise DomainValidationError(f"{name} must be in [0, 1]")
    return converted


def _tail_proportion(value: float, name: str) -> float:
    converted = _finite(value, name)
    if not 0.0 <= converted < 0.5:
        raise DomainValidationError(f"{name} must be in [0, 0.5)")
    return converted


def _freeze_string_float_mapping(
    value: Mapping[str, float], name: str
) -> Mapping[str, float]:
    normalized: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip().upper()
        if not key:
            raise DomainValidationError(f"{name} contains an empty key")
        if key in normalized:
            raise DomainValidationError(f"{name} contains duplicate key {key!r}")
        normalized[key] = _finite(raw_value, f"{name}[{key}]")
    return MappingProxyType(normalized)


def _freeze_string_mapping(
    value: Mapping[str, str], name: str
) -> Mapping[str, str]:
    normalized: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip().upper()
        mapped = str(raw_value).strip()
        if not key or not mapped:
            raise DomainValidationError(f"{name} keys and values must be non-empty")
        if key in normalized:
            raise DomainValidationError(f"{name} contains duplicate key {key!r}")
        normalized[key] = mapped
    return MappingProxyType(normalized)


def _assert_secret_free(value: Any, path: str = "metadata") -> None:
    """Reject key names that could serialize credentials into public artifacts."""
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            if any(fragment in key for fragment in _FORBIDDEN_METADATA_KEYS):
                raise DomainValidationError(
                    f"{path}.{raw_key} is not allowed in a persisted recipe"
                )
            _assert_secret_free(item, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_secret_free(item, f"{path}[{index}]")
    elif isinstance(value, str) and "://" in value:
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            raise DomainValidationError(
                f"{path} contains a credential-bearing URL"
            )


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursively plain mapping suitable for canonical JSON."""
    output: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            output[str(key)] = _plain_mapping(item)
        elif isinstance(item, tuple):
            output[str(key)] = list(item)
        else:
            output[str(key)] = item
    return output


@dataclass(frozen=True)
class AssumptionRecipe:
    """Frozen configuration for the existing robust-assumptions engine."""

    expected_return_method: str = "mean"
    trim_proportion: float = 0.10
    winsor_proportion: float = 0.05
    shrinkage_weight: float = 0.5
    manual_views: Mapping[str, float] = field(default_factory=dict)
    view_blend_weight: float = 1.0
    covariance_method: str = "sample"
    shrinkage_delta: float = 0.2
    shrinkage_target: str = "constant_correlation"
    decay_lambda: float = 0.94

    def __post_init__(self) -> None:
        expected_method = self.expected_return_method.strip().lower()
        covariance_method = self.covariance_method.strip().lower()
        shrinkage_target = self.shrinkage_target.strip().lower()
        if expected_method not in EXPECTED_RETURN_METHODS:
            raise DomainValidationError(
                f"unsupported expected-return method {expected_method!r}"
            )
        if covariance_method not in COVARIANCE_METHODS:
            raise DomainValidationError(
                f"unsupported covariance method {covariance_method!r}"
            )
        if shrinkage_target not in SHRINKAGE_TARGETS:
            raise DomainValidationError(
                f"unsupported shrinkage target {shrinkage_target!r}"
            )
        decay = _finite(self.decay_lambda, "decay_lambda")
        if not 0.0 < decay < 1.0:
            raise DomainValidationError("decay_lambda must be in (0, 1)")
        object.__setattr__(self, "expected_return_method", expected_method)
        object.__setattr__(self, "covariance_method", covariance_method)
        object.__setattr__(self, "shrinkage_target", shrinkage_target)
        object.__setattr__(
            self,
            "trim_proportion",
            _tail_proportion(self.trim_proportion, "trim_proportion"),
        )
        object.__setattr__(
            self,
            "winsor_proportion",
            _tail_proportion(self.winsor_proportion, "winsor_proportion"),
        )
        object.__setattr__(
            self,
            "shrinkage_weight",
            _unit_interval(self.shrinkage_weight, "shrinkage_weight"),
        )
        object.__setattr__(
            self,
            "view_blend_weight",
            _unit_interval(self.view_blend_weight, "view_blend_weight"),
        )
        object.__setattr__(
            self,
            "shrinkage_delta",
            _unit_interval(self.shrinkage_delta, "shrinkage_delta"),
        )
        object.__setattr__(self, "decay_lambda", decay)
        object.__setattr__(
            self,
            "manual_views",
            _freeze_string_float_mapping(self.manual_views, "manual_views"),
        )

    def to_assumption_config(self) -> AssumptionConfig:
        """Reconstruct the existing analytics configuration exactly."""
        return AssumptionConfig(
            expected_return_method=self.expected_return_method,
            trim_proportion=self.trim_proportion,
            winsor_proportion=self.winsor_proportion,
            shrinkage_weight=self.shrinkage_weight,
            manual_views=dict(self.manual_views),
            view_blend_weight=self.view_blend_weight,
            covariance_method=self.covariance_method,
            shrinkage_delta=self.shrinkage_delta,
            shrinkage_target=self.shrinkage_target,
            decay_lambda=self.decay_lambda,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_return_method": self.expected_return_method,
            "trim_proportion": self.trim_proportion,
            "winsor_proportion": self.winsor_proportion,
            "shrinkage_weight": self.shrinkage_weight,
            "manual_views": dict(self.manual_views),
            "view_blend_weight": self.view_blend_weight,
            "covariance_method": self.covariance_method,
            "shrinkage_delta": self.shrinkage_delta,
            "shrinkage_target": self.shrinkage_target,
            "decay_lambda": self.decay_lambda,
        }


@dataclass(frozen=True)
class ScenarioRecipe:
    """Scenario-source and horizon recipe used by optimization."""

    source: str = "historical"
    horizon_days: int = 1
    n_scenarios: int = 5_000
    student_t_df: float = 5.0
    random_seed: int | None = 42
    covariance_policy: str = "repair"

    def __post_init__(self) -> None:
        source = self.source.strip().lower()
        policy = self.covariance_policy.strip().lower()
        if source not in _SCENARIO_SOURCES:
            raise DomainValidationError(f"unsupported scenario source {source!r}")
        if self.horizon_days < 1:
            raise DomainValidationError("horizon_days must be at least one")
        if self.n_scenarios < 2:
            raise DomainValidationError("n_scenarios must be at least two")
        degrees = _finite(self.student_t_df, "student_t_df")
        if degrees <= 2.0:
            raise DomainValidationError("student_t_df must exceed two")
        if policy not in {"repair", "strict"}:
            raise DomainValidationError(
                "covariance_policy must be 'repair' or 'strict'"
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "student_t_df", degrees)
        object.__setattr__(self, "covariance_policy", policy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "horizon_days": self.horizon_days,
            "n_scenarios": self.n_scenarios,
            "student_t_df": self.student_t_df,
            "random_seed": self.random_seed,
            "covariance_policy": self.covariance_policy,
        }


@dataclass(frozen=True)
class OptimizerRecipe:
    """Objective, solver, and portfolio-constraint recipe."""

    objective: str = "min_cvar"
    confidence_level: float = 0.95
    long_only: bool = True
    min_weight: float | None = None
    max_weight: float | None = 1.0
    cvar_limit: float | None = None
    target_return: float | None = None
    risk_free_rate: float = 0.0
    solver: str | None = None
    sharpe_grid_points: int = 25
    accept_optimal_inaccurate: bool = False

    def __post_init__(self) -> None:
        objective = self.objective.strip().lower()
        if objective not in _OBJECTIVES:
            raise DomainValidationError(f"unsupported optimizer objective {objective!r}")
        minimum = (
            None
            if self.min_weight is None
            else _finite(self.min_weight, "min_weight")
        )
        maximum = (
            None
            if self.max_weight is None
            else _finite(self.max_weight, "max_weight")
        )
        if minimum is not None and maximum is not None and minimum > maximum:
            raise DomainValidationError("min_weight must not exceed max_weight")
        cvar_limit = (
            None
            if self.cvar_limit is None
            else _finite(self.cvar_limit, "cvar_limit")
        )
        if objective == "max_return_cvar" and (
            cvar_limit is None or cvar_limit <= 0.0
        ):
            raise DomainValidationError(
                "max_return_cvar requires a positive cvar_limit"
            )
        target_return = (
            None
            if self.target_return is None
            else _finite(self.target_return, "target_return")
        )
        if objective == "min_cvar_target_return" and target_return is None:
            raise DomainValidationError(
                "min_cvar_target_return requires target_return"
            )
        if self.sharpe_grid_points < 2:
            raise DomainValidationError("sharpe_grid_points must be at least two")
        solver = self.solver.strip() if self.solver is not None else None
        if solver == "":
            solver = None
        object.__setattr__(self, "objective", objective)
        object.__setattr__(
            self,
            "confidence_level",
            _probability(self.confidence_level, "confidence_level"),
        )
        object.__setattr__(self, "min_weight", minimum)
        object.__setattr__(self, "max_weight", maximum)
        object.__setattr__(self, "cvar_limit", cvar_limit)
        object.__setattr__(self, "target_return", target_return)
        object.__setattr__(
            self, "risk_free_rate", _finite(self.risk_free_rate, "risk_free_rate")
        )
        object.__setattr__(self, "solver", solver)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "confidence_level": self.confidence_level,
            "long_only": self.long_only,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "cvar_limit": self.cvar_limit,
            "target_return": self.target_return,
            "risk_free_rate": self.risk_free_rate,
            "solver": self.solver,
            "sharpe_grid_points": self.sharpe_grid_points,
            "accept_optimal_inaccurate": self.accept_optimal_inaccurate,
        }


@dataclass(frozen=True)
class RiskMonitoringRecipe:
    """Origin-safe risk-forecast settings saved before monitoring begins."""

    var_method: str = "historical"
    cvar_method: str = "historical"
    confidence_level: float = 0.95
    horizon_days: int = 1
    estimation_window: int = 252
    evaluation_mode: str = "overlapping"
    convention_version: str = "1"

    def __post_init__(self) -> None:
        evaluation_mode = self.evaluation_mode.strip().lower()
        if evaluation_mode not in _EVALUATION_MODES:
            raise DomainValidationError(
                f"unsupported evaluation mode {evaluation_mode!r}"
            )
        if self.horizon_days < 1:
            raise DomainValidationError("risk horizon_days must be at least one")
        if self.estimation_window < 2:
            raise DomainValidationError("estimation_window must be at least two")
        if not self.var_method.strip() or not self.cvar_method.strip():
            raise DomainValidationError("VaR and CVaR methods are required")
        if not self.convention_version.strip():
            raise DomainValidationError("convention_version is required")
        object.__setattr__(self, "var_method", self.var_method.strip().lower())
        object.__setattr__(self, "cvar_method", self.cvar_method.strip().lower())
        object.__setattr__(self, "evaluation_mode", evaluation_mode)
        object.__setattr__(
            self,
            "confidence_level",
            _probability(self.confidence_level, "risk confidence_level"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "var_method": self.var_method,
            "cvar_method": self.cvar_method,
            "confidence_level": self.confidence_level,
            "horizon_days": self.horizon_days,
            "estimation_window": self.estimation_window,
            "evaluation_mode": self.evaluation_mode,
            "convention_version": self.convention_version,
        }


@dataclass(frozen=True)
class CashPolicy:
    """Explicit deterministic cash policy for fixed-holdings valuation."""

    enabled: bool = False
    mode: str = "zero"
    annual_rate: float = 0.0
    symbol: str = "CASH"

    def __post_init__(self) -> None:
        mode = self.mode.strip().lower()
        symbol = self.symbol.strip().upper()
        rate = _finite(self.annual_rate, "annual_rate")
        if mode not in _CASH_MODES:
            raise DomainValidationError(f"unsupported cash mode {mode!r}")
        if not symbol:
            raise DomainValidationError("cash symbol is required")
        if rate <= -1.0:
            raise DomainValidationError("annual_rate must exceed -100%")
        if mode == "zero" and not math.isclose(rate, 0.0, abs_tol=0.0):
            raise DomainValidationError("zero cash mode requires annual_rate=0")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "annual_rate", rate)

    def horizon_return(self, horizon_days: int) -> float:
        if horizon_days < 0:
            raise DomainValidationError("horizon_days must be non-negative")
        if self.mode == "zero":
            return 0.0
        return (1.0 + self.annual_rate) ** (horizon_days / 365.0) - 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "annual_rate": self.annual_rate,
            "day_count": 365,
            "symbol": self.symbol,
        }


@dataclass(frozen=True)
class SourceRecipe:
    """Refresh/provenance mapping with credentials explicitly excluded."""

    provider: str
    quote_currency: str = "USD"
    symbol_mapping: Mapping[str, str] = field(default_factory=dict)
    refreshable: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider = self.provider.strip()
        quote = self.quote_currency.strip().upper()
        if not provider or not quote:
            raise DomainValidationError("provider and quote_currency are required")
        _assert_secret_free(self.metadata)
        metadata = _plain_mapping(self.metadata)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "quote_currency", quote)
        object.__setattr__(
            self,
            "symbol_mapping",
            _freeze_string_mapping(self.symbol_mapping, "symbol_mapping"),
        )
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "quote_currency": self.quote_currency,
            "symbol_mapping": dict(self.symbol_mapping),
            "refreshable": self.refreshable,
            "metadata": _plain_mapping(self.metadata),
        }


@dataclass(frozen=True)
class OptimizationRecipe:
    """Complete construction and monitoring recipe frozen with a snapshot."""

    assumptions: AssumptionRecipe = field(default_factory=AssumptionRecipe)
    scenario: ScenarioRecipe = field(default_factory=ScenarioRecipe)
    optimizer: OptimizerRecipe = field(default_factory=OptimizerRecipe)
    risk: RiskMonitoringRecipe = field(default_factory=RiskMonitoringRecipe)
    cash: CashPolicy = field(default_factory=CashPolicy)
    source: SourceRecipe = field(
        default_factory=lambda: SourceRecipe(provider="uploaded_csv")
    )
    recipe_version: str = "1"

    def __post_init__(self) -> None:
        if not self.recipe_version.strip():
            raise DomainValidationError("recipe_version is required")
        if self.risk.horizon_days != self.scenario.horizon_days:
            raise DomainValidationError(
                "risk and optimization scenario horizons must match"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_version": self.recipe_version,
            "assumptions": self.assumptions.to_dict(),
            "scenario": self.scenario.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "risk": self.risk.to_dict(),
            "cash": self.cash.to_dict(),
            "source": self.source.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        return sha256_fingerprint(self.to_dict())


def optimization_recipe_from_dict(value: Mapping[str, Any]) -> OptimizationRecipe:
    """Reconstruct and revalidate a persisted secret-free recipe.

    Persistence is deliberately JSON-shaped.  Rebuilding the dataclasses at the
    service boundary reruns every validation rule instead of trusting mutable
    database JSON blindly.
    """
    if not isinstance(value, Mapping):
        raise DomainValidationError("persisted optimization recipe must be a mapping")
    try:
        cash_values = dict(value["cash"])
        cash_values.pop("day_count", None)
        return OptimizationRecipe(
            assumptions=AssumptionRecipe(**dict(value["assumptions"])),
            scenario=ScenarioRecipe(**dict(value["scenario"])),
            optimizer=OptimizerRecipe(**dict(value["optimizer"])),
            risk=RiskMonitoringRecipe(**dict(value["risk"])),
            cash=CashPolicy(**cash_values),
            source=SourceRecipe(**dict(value["source"])),
            recipe_version=str(value["recipe_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DomainValidationError(
            "persisted optimization recipe is incomplete or invalid"
        ) from exc


__all__ = [
    "AssumptionRecipe",
    "CashPolicy",
    "OptimizationRecipe",
    "OptimizerRecipe",
    "RiskMonitoringRecipe",
    "ScenarioRecipe",
    "SourceRecipe",
    "optimization_recipe_from_dict",
]
