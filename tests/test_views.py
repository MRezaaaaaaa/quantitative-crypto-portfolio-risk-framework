"""Tests for the Phase 5.5 manual expected-return views layer."""

from __future__ import annotations

import pandas as pd
import pytest

from var_cvar_crypto_risk.views import (
    AssetReturnView,
    apply_manual_expected_return_views,
)


def _base() -> pd.Series:
    return pd.Series({"BTC": 0.001, "ETH": 0.002, "SOL": 0.003})


def test_full_replace_when_blend_one() -> None:
    views = [AssetReturnView("BTC", 0.05)]
    out = apply_manual_expected_return_views(_base(), views, blend_weight=1.0)
    assert out["BTC"] == pytest.approx(0.05)
    # untouched assets stay the same
    assert out["ETH"] == pytest.approx(0.002)
    assert out["SOL"] == pytest.approx(0.003)


def test_half_blend() -> None:
    views = [AssetReturnView("ETH", 0.05)]
    out = apply_manual_expected_return_views(_base(), views, blend_weight=0.5)
    assert out["ETH"] == pytest.approx(0.5 * 0.05 + 0.5 * 0.002)


def test_blend_zero_keeps_base() -> None:
    views = [AssetReturnView("SOL", 0.99)]
    out = apply_manual_expected_return_views(_base(), views, blend_weight=0.0)
    assert out["SOL"] == pytest.approx(0.003)


def test_unknown_asset_raises() -> None:
    with pytest.raises(ValueError):
        apply_manual_expected_return_views(
            _base(), [AssetReturnView("DOGE", 0.01)], blend_weight=1.0
        )


def test_does_not_mutate_input() -> None:
    base = _base()
    apply_manual_expected_return_views(base, [AssetReturnView("BTC", 0.5)])
    assert base["BTC"] == pytest.approx(0.001)


def test_invalid_blend_weight_raises() -> None:
    with pytest.raises(ValueError):
        apply_manual_expected_return_views(
            _base(), [AssetReturnView("BTC", 0.01)], blend_weight=1.5
        )
