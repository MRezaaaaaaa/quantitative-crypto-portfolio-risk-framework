"""Central return-convention policy for the application and demos."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReturnPolicy:
    """Resolved return conventions for each calculation boundary.

    Portfolio construction, wealth paths, scenarios, and optimization use
    simple returns because those operations require arithmetic aggregation.
    Log returns are available only as an advanced diagnostic convention.
    """

    handling_mode: str
    diagnostic_method: str
    portfolio_method: str = "simple"
    wealth_method: str = "simple"
    scenario_method: str = "simple"
    optimization_method: str = "simple"


def resolve_return_policy(
    handling_mode: str = "automatic",
    diagnostic_method: str = "simple",
) -> ReturnPolicy:
    """Resolve a deterministic Simple/Log routing policy.

    ``automatic`` uses simple returns throughout. ``advanced`` permits log
    returns for distribution diagnostics only; all portfolio, wealth,
    scenario, and optimization calculations remain in simple-return space.
    """
    if handling_mode not in {"automatic", "advanced"}:
        raise ValueError(
            f"handling_mode must be 'automatic' or 'advanced', got {handling_mode!r}."
        )
    if diagnostic_method not in {"simple", "log"}:
        raise ValueError(
            f"diagnostic_method must be 'simple' or 'log', got {diagnostic_method!r}."
        )

    resolved_diagnostic = (
        "simple" if handling_mode == "automatic" else diagnostic_method
    )
    return ReturnPolicy(
        handling_mode=handling_mode,
        diagnostic_method=resolved_diagnostic,
    )
