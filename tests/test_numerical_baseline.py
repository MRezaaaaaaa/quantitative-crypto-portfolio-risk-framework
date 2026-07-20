"""Golden numerical regression tests for publication-critical calculations."""

from __future__ import annotations

import json
from numbers import Integral, Real

import numpy as np

from tests.numerical_baseline import GOLDEN_PATH, compute_numerical_baseline


def _assert_same(actual, expected, path: str = "root") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected a mapping"
        assert actual.keys() == expected.keys(), f"{path}: key mismatch"
        for key in expected:
            _assert_same(actual[key], expected[key], f"{path}.{key}")
        return

    if isinstance(expected, list):
        assert actual == expected, f"{path}: list mismatch"
        return

    if isinstance(expected, bool):
        assert actual is expected, f"{path}: boolean mismatch"
        return

    if isinstance(expected, Integral):
        assert actual == expected, f"{path}: integer mismatch"
        return

    if isinstance(expected, Real):
        rtol = 1e-7 if ".monte_carlo." in path else 1e-10
        atol = 1e-9 if ".monte_carlo." in path else 1e-12
        np.testing.assert_allclose(
            float(actual),
            float(expected),
            rtol=rtol,
            atol=atol,
            err_msg=path,
        )
        return

    assert actual == expected, f"{path}: value mismatch"


def test_numerical_golden_baseline() -> None:
    """Current public calculations must match the reviewed baseline."""
    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    actual = compute_numerical_baseline()
    _assert_same(actual, expected)
