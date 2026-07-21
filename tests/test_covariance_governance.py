"""Tests for covariance diagnostics and deterministic PSD repair."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from var_cvar_crypto_risk.covariance import (
    covariance_diagnostics,
    prepare_covariance_matrix,
)
from var_cvar_crypto_risk.monte_carlo import simulate_normal_returns
from var_cvar_crypto_risk.optimization import build_optimization_scenarios


def _indefinite_covariance() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.04, 0.072], [0.072, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )


def test_covariance_diagnostics_detect_indefinite_matrix() -> None:
    diagnostics = covariance_diagnostics(_indefinite_covariance())
    assert diagnostics["is_psd"] is False
    assert diagnostics["min_eigenvalue"] < 0.0
    assert diagnostics["is_symmetric"] is True


def test_covariance_repair_is_pd_and_preserves_variances() -> None:
    original = _indefinite_covariance()
    repaired, report = prepare_covariance_matrix(original, policy="repair")

    np.testing.assert_allclose(np.diag(repaired), np.diag(original), rtol=1e-12)
    np.testing.assert_allclose(repaired, repaired.T, atol=1e-15)
    assert np.linalg.eigvalsh(repaired.to_numpy()).min() > 0.0
    assert report["repaired"] is True
    assert report["before"]["is_psd"] is False
    assert report["after"]["is_positive_definite"] is True
    assert report["relative_frobenius_adjustment"] > 0.0


def test_covariance_strict_policy_rejects_indefinite_matrix() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        prepare_covariance_matrix(_indefinite_covariance(), policy="strict")


def test_covariance_repair_leaves_valid_matrix_unchanged() -> None:
    valid = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    prepared, report = prepare_covariance_matrix(valid, policy="repair")
    pd.testing.assert_frame_equal(prepared, valid)
    assert report["repaired"] is False


def test_covariance_governance_rejects_invalid_labels_and_values() -> None:
    bad_labels = pd.DataFrame(np.eye(2), index=["A", "B"], columns=["A", "C"])
    with pytest.raises(ValueError, match="labels"):
        prepare_covariance_matrix(bad_labels)

    nonfinite = pd.DataFrame(
        [[1.0, np.nan], [np.nan, 1.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    with pytest.raises(ValueError, match="finite"):
        prepare_covariance_matrix(nonfinite)


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.eye(2), "pandas DataFrame"),
        (pd.DataFrame(), "empty"),
        (pd.DataFrame([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), "square"),
        (
            pd.DataFrame(
                [[1.0, 0.0], [0.0, 1.0]],
                index=["A", "A"],
                columns=["A", "A"],
            ),
            "unique",
        ),
    ],
)
def test_covariance_governance_rejects_invalid_frame_contracts(
    covariance,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        covariance_diagnostics(covariance)


def test_covariance_governance_rejects_invalid_numerical_controls() -> None:
    covariance = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
    with pytest.raises(ValueError, match="tolerance must be"):
        covariance_diagnostics(covariance, tolerance=0.0)
    with pytest.raises(ValueError, match="policy must be"):
        prepare_covariance_matrix(covariance, policy="clip")
    with pytest.raises(ValueError, match="eigenvalue_floor"):
        prepare_covariance_matrix(covariance, eigenvalue_floor=0.0)


def test_covariance_governance_rejects_negative_variance() -> None:
    covariance = pd.DataFrame([[-1.0]], index=["A"], columns=["A"])
    with pytest.raises(ValueError, match="negative variance"):
        prepare_covariance_matrix(covariance)


def test_covariance_repair_reports_asymmetry_reason() -> None:
    covariance = pd.DataFrame(
        [[1.0, 0.8], [0.2, 1.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    repaired, report = prepare_covariance_matrix(covariance)

    assert report["repaired"] is True
    assert "asymmetry" in report["reasons"]
    np.testing.assert_allclose(repaired, repaired.T)


def test_monte_carlo_attaches_covariance_repair_report() -> None:
    scenarios = simulate_normal_returns(
        mean_vector=pd.Series({"A": 0.0, "B": 0.0}),
        covariance_matrix=_indefinite_covariance(),
        n_scenarios=100,
        random_seed=7,
    )
    report = scenarios.attrs["covariance_governance"]
    assert report["repaired"] is True
    assert report["after"]["is_positive_definite"] is True
    assert report["cholesky_jitter"] == 0.0


def test_monte_carlo_strict_covariance_policy_fails_loudly() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        simulate_normal_returns(
            mean_vector=pd.Series({"A": 0.0, "B": 0.0}),
            covariance_matrix=_indefinite_covariance(),
            covariance_policy="strict",
        )


def test_simulation_rejects_nonfinite_mean_and_zero_variance() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    with pytest.raises(ValueError, match="finite"):
        simulate_normal_returns(
            mean_vector=pd.Series({"A": np.nan, "B": 0.0}),
            covariance_matrix=covariance,
        )

    singular = covariance.copy()
    singular.loc["B", "B"] = 0.0
    with pytest.raises(ValueError, match="zero-variance"):
        simulate_normal_returns(
            mean_vector=pd.Series({"A": 0.0, "B": 0.0}),
            covariance_matrix=singular,
        )


def test_optimization_scenario_builder_propagates_governance() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.03, -0.01],
            "B": [0.02, -0.01, 0.01, -0.03],
        }
    )
    scenarios = build_optimization_scenarios(
        asset_returns=returns,
        source="normal_mc",
        n_scenarios=50,
        covariance_matrix=_indefinite_covariance(),
        random_seed=11,
    )
    assert scenarios.attrs["covariance_governance"]["repaired"] is True

    with pytest.raises(ValueError, match="positive definite"):
        build_optimization_scenarios(
            asset_returns=returns,
            source="normal_mc",
            covariance_matrix=_indefinite_covariance(),
            covariance_policy="strict",
        )
