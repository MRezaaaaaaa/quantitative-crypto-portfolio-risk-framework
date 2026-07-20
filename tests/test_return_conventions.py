"""Tests for deterministic Simple/Log return-routing policy."""

from __future__ import annotations

import pytest

from var_cvar_crypto_risk.return_conventions import resolve_return_policy


def test_automatic_policy_uses_simple_everywhere() -> None:
    policy = resolve_return_policy("automatic", diagnostic_method="log")
    assert policy.handling_mode == "automatic"
    assert policy.diagnostic_method == "simple"
    assert policy.portfolio_method == "simple"
    assert policy.wealth_method == "simple"
    assert policy.scenario_method == "simple"
    assert policy.optimization_method == "simple"


def test_advanced_policy_allows_log_for_diagnostics_only() -> None:
    policy = resolve_return_policy("advanced", diagnostic_method="log")
    assert policy.handling_mode == "advanced"
    assert policy.diagnostic_method == "log"
    assert policy.portfolio_method == "simple"
    assert policy.wealth_method == "simple"
    assert policy.scenario_method == "simple"
    assert policy.optimization_method == "simple"


@pytest.mark.parametrize("mode", ["manual", "", "AUTO"])
def test_return_policy_rejects_unknown_mode(mode: str) -> None:
    with pytest.raises(ValueError, match="handling_mode"):
        resolve_return_policy(mode)


def test_return_policy_rejects_unknown_diagnostic_method() -> None:
    with pytest.raises(ValueError, match="diagnostic_method"):
        resolve_return_policy("advanced", diagnostic_method="arithmetic")
