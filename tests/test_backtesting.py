"""Tests for var_cvar_crypto_risk.backtesting (Phase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from var_cvar_crypto_risk.backtesting import (
    assign_traffic_light_status,
    backtest_var_model,
    calculate_realized_horizon_returns,
    calculate_rolling_breach_rate,
    calculate_var_breaches,
    christoffersen_cc_test,
    christoffersen_independence_test,
    compare_var_models_backtest,
    create_backtesting_report_table,
    get_worst_realized_losses,
    interpret_traffic_light_status,
    kupiec_pof_test,
    rolling_var_forecast,
    summarize_backtest_by_period,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_returns_300() -> pd.Series:
    """300 observations, Student-t df=4, scale=0.02. seed=42."""
    rng = np.random.default_rng(seed=42)
    samples = rng.standard_t(df=4, size=300) * 0.02
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    return pd.Series(samples, index=dates, name="returns")


@pytest.fixture
def clean_returns_1000() -> pd.Series:
    """1000 observations, Student-t df=4, scale=0.02. seed=99."""
    rng = np.random.default_rng(seed=99)
    samples = rng.standard_t(df=4, size=1000) * 0.02
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    return pd.Series(samples, index=dates, name="returns")


@pytest.fixture
def all_breach_series() -> np.ndarray:
    """100-element boolean array, all True."""
    return np.ones(100, dtype=bool)


@pytest.fixture
def no_breach_series() -> np.ndarray:
    """100-element boolean array, all False."""
    return np.zeros(100, dtype=bool)


@pytest.fixture
def alternating_breach_series() -> np.ndarray:
    """100-element boolean array, alternating True/False."""
    arr = np.zeros(100, dtype=bool)
    arr[::2] = True
    return arr


@pytest.fixture
def random_breach_series() -> np.ndarray:
    """500-element i.i.d. Bernoulli(0.05) breach series — truly independent."""
    rng = np.random.default_rng(seed=2025)
    return (rng.uniform(size=500) < 0.05).astype(bool)


# ── rolling_var_forecast ────────────────────────────────────────────────────


def test_rolling_var_forecast_output_length():
    """Returns 300, window 100 → output length 200."""
    rng = np.random.default_rng(seed=1)
    series = pd.Series(rng.normal(0, 0.02, size=300))
    series.index = pd.date_range("2023-01-01", periods=300, freq="D")
    out = rolling_var_forecast(series, method="historical", window=100)
    assert len(out) == 200


def test_rolling_var_forecast_output_length_1000(clean_returns_1000):
    """Returns 1000, window 252 → output length 748."""
    out = rolling_var_forecast(
        clean_returns_1000, method="historical", window=252
    )
    assert len(out) == 748


def test_rolling_var_forecast_columns(clean_returns_300):
    """Output must include actual_return, var_forecast, breach, horizon_days, method."""
    out = rolling_var_forecast(clean_returns_300, method="historical", window=60)
    assert {
        "actual_return",
        "var_forecast",
        "breach",
        "horizon_days",
        "method",
    }.issubset(set(out.columns))
    assert (out["horizon_days"] == 1).all()
    assert (out["method"] == "historical").all()


def test_rolling_var_forecast_no_lookahead():
    """For each row, var_forecast must be computed from BEFORE that date.

    A monotonically increasing series produces a deterministic check: the
    historical VaR computed using only past values must equal -min(window).
    The realised value at t must NOT be used in computing var_forecast.
    """
    n = 200
    values = np.linspace(-1.0, 1.0, n)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    series = pd.Series(values, index=dates)
    window = 50
    out = rolling_var_forecast(
        series, method="historical", confidence_level=0.95, window=window
    )

    for i, t in enumerate(out.index):
        pos = series.index.get_loc(t)
        past_window = series.iloc[pos - window:pos].to_numpy()
        expected_var = -float(np.quantile(past_window, 0.05))
        assert out["var_forecast"].iloc[i] == pytest.approx(expected_var)
        # Confirm forecast does NOT depend on the realised value at t:
        assert series.iloc[pos] not in past_window or pos == 0


def test_rolling_var_forecast_breach_detection():
    """Manually crafted series produces the expected breach pattern."""
    window = 30
    base = np.full(window, 0.0)  # zero history → VaR = 0
    after = np.array([-0.03, 0.01])
    values = np.concatenate([base, after])
    series = pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="D"))
    out = rolling_var_forecast(series, method="historical", window=window)
    assert len(out) == 2
    assert bool(out["breach"].iloc[0]) is True
    assert bool(out["breach"].iloc[1]) is False


@pytest.mark.parametrize("method", ["historical", "gaussian", "cornish_fisher"])
def test_rolling_var_forecast_all_methods(method, clean_returns_300):
    """All three methods return valid DataFrames."""
    out = rolling_var_forecast(
        clean_returns_300, method=method, confidence_level=0.95, window=100
    )
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 200
    assert out["var_forecast"].notna().all()
    assert (out["var_forecast"] > -1.0).all()


def test_rolling_var_forecast_window_too_large_raises(clean_returns_300):
    """window >= len(returns) → ValueError."""
    with pytest.raises(ValueError):
        rolling_var_forecast(clean_returns_300, method="historical", window=300)


def test_rolling_var_forecast_empty_raises():
    """Empty returns → ValueError."""
    empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError):
        rolling_var_forecast(empty, method="historical", window=30)


def test_rolling_var_forecast_invalid_method_raises(clean_returns_300):
    """Unknown method → ValueError."""
    with pytest.raises(ValueError):
        rolling_var_forecast(
            clean_returns_300, method="garch_unknown", window=100
        )


def test_rolling_var_forecast_invalid_confidence_raises(clean_returns_300):
    """confidence_level=0.0 → ValueError. confidence_level=1.0 → ValueError."""
    with pytest.raises(ValueError):
        rolling_var_forecast(
            clean_returns_300, method="historical", confidence_level=0.0, window=100
        )
    with pytest.raises(ValueError):
        rolling_var_forecast(
            clean_returns_300, method="historical", confidence_level=1.0, window=100
        )


# ── calculate_var_breaches ──────────────────────────────────────────────────


def test_calculate_var_breaches_counts(clean_returns_300):
    """observations, actual_breaches, expected_breaches are numerically correct."""
    out = rolling_var_forecast(clean_returns_300, method="historical", window=100)
    summary = calculate_var_breaches(out, confidence_level=0.95)
    assert summary["observations"] == 200
    assert summary["actual_breaches"] == int(out["breach"].sum())
    assert summary["expected_breaches"] == pytest.approx(200 * 0.05)


def test_calculate_var_breaches_rates(clean_returns_300):
    """actual_breach_rate = actual_breaches / observations."""
    out = rolling_var_forecast(clean_returns_300, method="historical", window=100)
    summary = calculate_var_breaches(out, confidence_level=0.95)
    expected = summary["actual_breaches"] / summary["observations"]
    assert summary["actual_breach_rate"] == pytest.approx(expected)


def test_calculate_var_breaches_ratio(clean_returns_300):
    """breach_ratio = actual_breaches / expected_breaches."""
    out = rolling_var_forecast(clean_returns_300, method="historical", window=100)
    summary = calculate_var_breaches(out, confidence_level=0.95)
    expected_ratio = summary["actual_breaches"] / summary["expected_breaches"]
    assert summary["breach_ratio"] == pytest.approx(expected_ratio)


def test_calculate_var_breaches_zero_expected():
    """If expected_breaches == 0, breach_ratio == np.nan."""
    df = pd.DataFrame(
        {
            "actual_return": [0.01, -0.02, 0.0],
            "var_forecast": [0.03, 0.03, 0.03],
            "breach": [False, False, False],
        }
    )
    summary = calculate_var_breaches(df, confidence_level=1.0 - 0.0)
    assert np.isnan(summary["breach_ratio"])


# ── kupiec_pof_test ─────────────────────────────────────────────────────────


def _breaches_from(returns):
    out = rolling_var_forecast(returns, method="historical", window=100)
    return out["breach"].to_numpy(dtype=bool)


def test_kupiec_returns_dict(clean_returns_300):
    """Result is a dict with all required keys."""
    breaches = _breaches_from(clean_returns_300)
    res = kupiec_pof_test(breaches, confidence_level=0.95)
    expected_keys = {
        "test_name", "observations", "breaches",
        "expected_breach_probability", "observed_breach_probability",
        "lr_statistic", "p_value", "pass_test", "interpretation",
    }
    assert expected_keys.issubset(res.keys())


def test_kupiec_p_value_in_range(clean_returns_300):
    """p_value is in [0, 1]."""
    breaches = _breaches_from(clean_returns_300)
    res = kupiec_pof_test(breaches, confidence_level=0.95)
    assert 0.0 <= res["p_value"] <= 1.0


def test_kupiec_zero_breaches_no_crash(no_breach_series):
    """x=0 must not raise or return NaN p_value."""
    res = kupiec_pof_test(no_breach_series, confidence_level=0.95)
    assert np.isfinite(res["p_value"])
    assert isinstance(res["pass_test"], bool)


def test_kupiec_all_breaches_no_crash(all_breach_series):
    """x=n must not raise or return NaN p_value."""
    res = kupiec_pof_test(all_breach_series, confidence_level=0.95)
    assert np.isfinite(res["p_value"])
    assert isinstance(res["pass_test"], bool)


def test_kupiec_pass_test_type(clean_returns_300):
    """pass_test must be bool."""
    breaches = _breaches_from(clean_returns_300)
    res = kupiec_pof_test(breaches, confidence_level=0.95)
    assert isinstance(res["pass_test"], bool)


# ── christoffersen_independence_test ────────────────────────────────────────


def test_christoffersen_returns_transition_counts(clean_returns_300):
    """Result dict contains n00, n01, n10, n11."""
    breaches = _breaches_from(clean_returns_300)
    res = christoffersen_independence_test(breaches)
    for key in ("n00", "n01", "n10", "n11"):
        assert key in res
    total = res["n00"] + res["n01"] + res["n10"] + res["n11"]
    assert total == len(breaches) - 1


def test_christoffersen_p_value_in_range(clean_returns_300):
    """p_value is in [0, 1]."""
    breaches = _breaches_from(clean_returns_300)
    res = christoffersen_independence_test(breaches)
    assert 0.0 <= res["p_value"] <= 1.0


def test_christoffersen_alternating_rejected(alternating_breach_series):
    """A perfectly alternating series violates Markov independence (pi01=1, pi11=0)
    and should be rejected. This pins the test to the right intuition: the
    Christoffersen test detects ANY serial dependence, not just clustering."""
    res = christoffersen_independence_test(alternating_breach_series)
    assert res["pass_test"] is False
    assert res["p_value"] < 0.05


def test_christoffersen_random_passes(random_breach_series):
    """A truly i.i.d. Bernoulli series should not be rejected for independence."""
    res = christoffersen_independence_test(random_breach_series)
    assert res["pass_test"] is True
    assert res["p_value"] > 0.05


def test_christoffersen_all_same_no_crash(all_breach_series, no_breach_series):
    """All-True or all-False input must not crash."""
    r1 = christoffersen_independence_test(all_breach_series)
    r2 = christoffersen_independence_test(no_breach_series)
    assert np.isfinite(r1["p_value"])
    assert np.isfinite(r2["p_value"])


def test_christoffersen_short_series_graceful():
    """Series of length 1 → pass_test is None, no crash."""
    res = christoffersen_independence_test(np.array([True]))
    assert res["pass_test"] is None
    assert np.isnan(res["p_value"])


# ── christoffersen_cc_test ──────────────────────────────────────────────────


def test_cc_test_lr_equals_sum(clean_returns_300):
    """lr_cc == lr_pof + lr_ind (within 1e-10 tolerance)."""
    breaches = _breaches_from(clean_returns_300)
    pof = kupiec_pof_test(breaches, confidence_level=0.95)
    ind = christoffersen_independence_test(breaches)
    cc = christoffersen_cc_test(breaches, confidence_level=0.95)
    assert cc["lr_cc"] == pytest.approx(
        pof["lr_statistic"] + ind["lr_statistic"], abs=1e-10
    )


def test_cc_test_p_value_in_range(clean_returns_300):
    """p_value is in [0, 1]."""
    breaches = _breaches_from(clean_returns_300)
    res = christoffersen_cc_test(breaches, confidence_level=0.95)
    assert 0.0 <= res["p_value"] <= 1.0


def test_cc_test_uses_df2():
    """Manually verify p_value is computed with chi2(df=2), not df=1."""
    rng = np.random.default_rng(seed=12345)
    breaches = (rng.uniform(size=400) < 0.06).astype(int)
    cc = christoffersen_cc_test(breaches, confidence_level=0.95)
    expected_p_df2 = float(1.0 - stats.chi2.cdf(cc["lr_cc"], df=2))
    expected_p_df1 = float(1.0 - stats.chi2.cdf(cc["lr_cc"], df=1))
    assert cc["p_value"] == pytest.approx(expected_p_df2, abs=1e-12)
    if cc["lr_cc"] > 1e-6:
        assert cc["p_value"] != pytest.approx(expected_p_df1, abs=1e-6)


# ── assign_traffic_light_status ─────────────────────────────────────────────


def test_traffic_light_green_rate_based():
    """breach_ratio=1.0 → Green in rate_based mode."""
    status = assign_traffic_light_status(
        actual_breach_rate=0.05,
        expected_breach_rate=0.05,
        actual_breaches=25,
        n_observations=500,
        mode="rate_based",
    )
    assert status == "Green"


def test_traffic_light_red_rate_based():
    """breach_ratio=3.0 → Red in rate_based mode."""
    status = assign_traffic_light_status(
        actual_breach_rate=0.15,
        expected_breach_rate=0.05,
        actual_breaches=75,
        n_observations=500,
        mode="rate_based",
    )
    assert status == "Red"


def test_traffic_light_yellow_rate_based():
    """breach_ratio=1.6 → Yellow in rate_based mode."""
    status = assign_traffic_light_status(
        actual_breach_rate=0.08,
        expected_breach_rate=0.05,
        actual_breaches=40,
        n_observations=500,
        mode="rate_based",
    )
    assert status == "Yellow"


def test_traffic_light_basel3_green():
    """3 breaches, 250 observations, mode=basel3 → Green."""
    status = assign_traffic_light_status(
        actual_breach_rate=3 / 250,
        expected_breach_rate=0.01,
        actual_breaches=3,
        n_observations=250,
        mode="basel3",
    )
    assert status == "Green"


def test_traffic_light_basel3_yellow():
    """7 breaches, 250 observations, mode=basel3 → Yellow."""
    status = assign_traffic_light_status(
        actual_breach_rate=7 / 250,
        expected_breach_rate=0.01,
        actual_breaches=7,
        n_observations=250,
        mode="basel3",
    )
    assert status == "Yellow"


def test_traffic_light_basel3_red():
    """12 breaches, 250 observations, mode=basel3 → Red."""
    status = assign_traffic_light_status(
        actual_breach_rate=12 / 250,
        expected_breach_rate=0.01,
        actual_breaches=12,
        n_observations=250,
        mode="basel3",
    )
    assert status == "Red"


def test_traffic_light_auto_selects_basel3_at_250():
    """mode=auto with n=250 and 12 breaches uses Basel3 (Red)."""
    status = assign_traffic_light_status(
        actual_breach_rate=12 / 250,
        expected_breach_rate=0.01,
        actual_breaches=12,
        n_observations=250,
        mode="auto",
    )
    assert status == "Red"


def test_traffic_light_auto_selects_rate_based_at_500():
    """mode=auto with n=500 ignores Basel3 thresholds and uses ratios.

    12 breaches at n=500 with expected_rate=0.05 yields ratio ≈ 0.48,
    which falls outside Basel3's "Red" cutoff (≥10) but lands in rate-based
    "Red" (ratio < 0.50). Either signal lands on "Red", so we instead
    check a case where the two systems disagree: 25 breaches, ratio=1.0.
    Basel3 would mark Red (>=10); rate_based marks Green.
    """
    status = assign_traffic_light_status(
        actual_breach_rate=0.05,
        expected_breach_rate=0.05,
        actual_breaches=25,
        n_observations=500,
        mode="auto",
    )
    assert status == "Green"


def test_traffic_light_invalid_mode_raises():
    """Unknown mode → ValueError."""
    with pytest.raises(ValueError):
        assign_traffic_light_status(
            actual_breach_rate=0.05,
            expected_breach_rate=0.05,
            actual_breaches=12,
            n_observations=250,
            mode="bogus",
        )


def test_interpret_traffic_light_status_known():
    for status in ("Green", "Yellow", "Red"):
        msg = interpret_traffic_light_status(status)
        assert isinstance(msg, str) and len(msg) > 0


def test_interpret_traffic_light_status_unknown_raises():
    with pytest.raises(ValueError):
        interpret_traffic_light_status("Purple")


# ── backtest_var_model ───────────────────────────────────────────────────────


def test_backtest_var_model_returns_tuple(clean_returns_1000):
    """Returns (pd.DataFrame, dict)."""
    forecast_df, result = backtest_var_model(
        clean_returns_1000, method="historical", confidence_level=0.95, window=252
    )
    assert isinstance(forecast_df, pd.DataFrame)
    assert isinstance(result, dict)


def test_backtest_var_model_result_keys(clean_returns_1000):
    """Result dict contains all 19 required keys."""
    _, result = backtest_var_model(
        clean_returns_1000, method="historical", confidence_level=0.95, window=252
    )
    expected = {
        "method", "confidence_level", "window", "horizon_days", "observations",
        "actual_breaches", "expected_breaches", "expected_breach_rate",
        "actual_breach_rate", "breach_ratio",
        "kupiec_lr_statistic", "kupiec_p_value", "kupiec_pass",
        "christoffersen_lr_statistic", "christoffersen_p_value", "christoffersen_pass",
        "cc_lr_statistic", "cc_p_value", "cc_pass",
        "traffic_light", "traffic_light_mode_used", "interpretation",
    }
    assert expected.issubset(result.keys())


def test_backtest_var_model_traffic_light_valid(clean_returns_1000):
    """traffic_light is one of Green, Yellow, Red."""
    _, result = backtest_var_model(
        clean_returns_1000, method="historical", confidence_level=0.95, window=252
    )
    assert result["traffic_light"] in {"Green", "Yellow", "Red"}


# ── compare_var_models_backtest ──────────────────────────────────────────────


def test_compare_returns_forecasts_dict(clean_returns_1000):
    """First return value is dict with method keys."""
    forecasts, _ = compare_var_models_backtest(
        clean_returns_1000, methods=["historical", "gaussian"], window=252
    )
    assert isinstance(forecasts, dict)
    assert "historical" in forecasts
    assert "gaussian" in forecasts


def test_compare_returns_comparison_df(clean_returns_1000):
    """Second return value is DataFrame with one row per method."""
    _, comparison = compare_var_models_backtest(
        clean_returns_1000,
        methods=["historical", "gaussian", "cornish_fisher"],
        window=252,
    )
    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) == 3


def test_compare_includes_all_methods(clean_returns_1000):
    """All three methods appear in comparison_df."""
    _, comparison = compare_var_models_backtest(
        clean_returns_1000,
        methods=["historical", "gaussian", "cornish_fisher"],
        window=252,
    )
    assert set(comparison["method"].tolist()) == {
        "historical", "gaussian", "cornish_fisher"
    }


def test_compare_does_not_crash_on_bad_method(clean_returns_1000):
    """Invalid method → comparison continues, error column populated."""
    forecasts, comparison = compare_var_models_backtest(
        clean_returns_1000,
        methods=["historical", "nonexistent", "gaussian"],
        window=252,
    )
    assert "historical" in forecasts
    assert "gaussian" in forecasts
    assert "nonexistent" not in forecasts
    bad_row = comparison.loc[comparison["method"] == "nonexistent"].iloc[0]
    assert bad_row["error"] is not None
    assert isinstance(bad_row["error"], str)


# ── create_backtesting_report_table ─────────────────────────────────────────


def test_report_table_has_required_columns(clean_returns_1000):
    """All 15 required columns are present (Horizon column added in Phase 4)."""
    _, comparison = compare_var_models_backtest(
        clean_returns_1000,
        methods=["historical", "gaussian", "cornish_fisher"],
        window=252,
    )
    report = create_backtesting_report_table(comparison)
    expected_columns = [
        "Method", "Horizon (days)", "Mode", "Observations",
        "Actual Breaches", "Expected Breaches",
        "Actual Breach Rate", "Expected Breach Rate",
        "Kupiec p-value", "Kupiec Pass",
        "Christoffersen p-value", "Christoffersen Pass",
        "CC p-value", "CC Pass", "Traffic Light", "Interpretation",
    ]
    assert list(report.columns) == expected_columns


def test_report_table_formats_pass_correctly(clean_returns_1000):
    """Pass columns show '✓ Pass' or '✗ Fail'."""
    _, comparison = compare_var_models_backtest(
        clean_returns_1000,
        methods=["historical", "gaussian", "cornish_fisher"],
        window=252,
    )
    report = create_backtesting_report_table(comparison)
    for col in ("Kupiec Pass", "Christoffersen Pass", "CC Pass"):
        for value in report[col]:
            assert value in {"✓ Pass", "✗ Fail", "N/A"}


# ── Phase 4: horizon-aware backtesting ──────────────────────────────────────


def test_calculate_realized_horizon_returns_simple():
    """Simple-return forward compounding matches the spec example."""
    returns = pd.Series(
        [0.01, 0.02, -0.01],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    out = calculate_realized_horizon_returns(returns, horizon_days=2, method="simple")
    assert len(out) == 1
    expected = (1.0 + 0.02) * (1.0 - 0.01) - 1.0
    assert out.iloc[0] == pytest.approx(expected)


def test_calculate_realized_horizon_returns_log():
    """Log-return forward aggregation is a forward sum."""
    returns = pd.Series(
        [0.01, 0.02, -0.01, 0.03],
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    out = calculate_realized_horizon_returns(returns, horizon_days=2, method="log")
    assert len(out) == 2
    assert out.iloc[0] == pytest.approx(0.02 + -0.01)
    assert out.iloc[1] == pytest.approx(-0.01 + 0.03)


def test_calculate_realized_horizon_returns_rejects_bad_inputs():
    s = pd.Series([0.01, 0.02])
    with pytest.raises(ValueError):
        calculate_realized_horizon_returns(s, horizon_days=0)
    with pytest.raises(ValueError):
        calculate_realized_horizon_returns(s, horizon_days=2, method="bogus")
    with pytest.raises(ValueError):
        calculate_realized_horizon_returns(s, horizon_days=5)


def test_rolling_var_forecast_horizon_output(clean_returns_1000):
    """For horizon_days=7 the output exposes the horizon column and is not the daily return."""
    out_h1 = rolling_var_forecast(
        clean_returns_1000, method="historical", window=252, horizon_days=1
    )
    out_h7 = rolling_var_forecast(
        clean_returns_1000, method="historical", window=252, horizon_days=7
    )
    assert "horizon_days" in out_h7.columns
    assert (out_h7["horizon_days"] == 7).all()
    # Length shrinks because we need a full forward window for each forecast.
    assert len(out_h7) == len(clean_returns_1000) - 252 - 7 + 1
    # The realised 7-day return at the first overlapping date must differ
    # from the daily realised return at the same date.
    common = out_h1.index.intersection(out_h7.index)
    assert len(common) > 0
    diffs = (
        out_h1.loc[common, "actual_return"] - out_h7.loc[common, "actual_return"]
    ).abs()
    assert (diffs > 1e-9).any()


def test_backtest_var_model_includes_horizon_days(clean_returns_1000):
    """``backtest_var_model`` propagates horizon_days into the result dict."""
    _, result = backtest_var_model(
        clean_returns_1000,
        method="historical",
        confidence_level=0.95,
        window=252,
        horizon_days=7,
    )
    assert "horizon_days" in result
    assert result["horizon_days"] == 7


def test_compare_var_models_backtest_includes_horizon_days(clean_returns_1000):
    """``compare_var_models_backtest`` carries horizon_days through to each row."""
    _, comparison = compare_var_models_backtest(
        clean_returns_1000,
        methods=["historical", "gaussian"],
        confidence_level=0.95,
        window=252,
        horizon_days=10,
    )
    assert "horizon_days" in comparison.columns
    assert (comparison["horizon_days"] == 10).all()


def test_rolling_var_forecast_horizon_preserves_no_lookahead():
    """For horizon>1, the var_forecast at date t must not depend on values
    at t or later — only on the strict lookback window before t."""
    rng = np.random.default_rng(seed=2024)
    series = pd.Series(
        rng.standard_t(df=4, size=500) * 0.02,
        index=pd.date_range("2023-01-01", periods=500, freq="D"),
    )
    out = rolling_var_forecast(
        series,
        method="historical",
        confidence_level=0.95,
        window=100,
        horizon_days=5,
        return_method="simple",
    )
    # Manually recompute the first row's var_forecast from the strict lookback.
    lookback = series.iloc[:100].to_numpy(dtype=float)
    aggregated = np.array(
        [
            float(np.prod(1.0 + lookback[j : j + 5]) - 1.0)
            for j in range(len(lookback) - 5 + 1)
        ]
    )
    expected_var = -float(np.quantile(aggregated, 0.05))
    assert out["var_forecast"].iloc[0] == pytest.approx(expected_var)


# ── Phase 5.5: backtest mode + breach-rate + worst losses + by-period ────────


def test_non_overlapping_fewer_observations_than_overlapping(clean_returns_1000):
    """For horizon > 1, non-overlapping produces ~1/horizon as many rows."""
    horizon = 7
    overlap = rolling_var_forecast(
        clean_returns_1000, method="historical", window=252,
        horizon_days=horizon, backtest_mode="overlapping",
    )
    non_overlap = rolling_var_forecast(
        clean_returns_1000, method="historical", window=252,
        horizon_days=horizon, backtest_mode="non_overlapping",
    )
    assert len(non_overlap) < len(overlap)
    # roughly one-seventh (allow generous tolerance)
    assert len(non_overlap) <= len(overlap) // (horizon - 1)
    assert (non_overlap["step_size"] == horizon).all()
    assert (overlap["step_size"] == 1).all()
    assert (non_overlap["backtest_mode"] == "non_overlapping").all()


def test_horizon_one_modes_are_equivalent(clean_returns_1000):
    """At horizon_days == 1 overlapping and non-overlapping are identical."""
    overlap = rolling_var_forecast(
        clean_returns_1000, method="gaussian", window=252,
        horizon_days=1, backtest_mode="overlapping",
    )
    non_overlap = rolling_var_forecast(
        clean_returns_1000, method="gaussian", window=252,
        horizon_days=1, backtest_mode="non_overlapping",
    )
    assert len(overlap) == len(non_overlap)
    pd.testing.assert_series_equal(
        overlap["var_forecast"], non_overlap["var_forecast"]
    )
    assert (overlap["step_size"] == 1).all()
    assert (non_overlap["step_size"] == 1).all()


def test_invalid_backtest_mode_raises(clean_returns_300):
    with pytest.raises(ValueError, match="backtest_mode"):
        rolling_var_forecast(clean_returns_300, window=100, backtest_mode="weird")


def test_backtest_mode_propagates_to_result(clean_returns_1000):
    _, result = backtest_var_model(
        clean_returns_1000, method="historical", window=252,
        horizon_days=5, backtest_mode="non_overlapping",
    )
    assert result["backtest_mode"] == "non_overlapping"
    assert result["step_size"] == 5


def test_report_table_has_mode_column(clean_returns_1000):
    _, comparison = compare_var_models_backtest(
        clean_returns_1000, methods=["historical", "gaussian"],
        window=252, horizon_days=3, backtest_mode="non_overlapping",
    )
    table = create_backtesting_report_table(comparison)
    assert "Mode" in table.columns
    assert (table["Mode"] == "Non Overlapping").all()


def test_calculate_rolling_breach_rate_returns_series(clean_returns_1000):
    forecast = rolling_var_forecast(clean_returns_1000, method="historical", window=252)
    rolling = calculate_rolling_breach_rate(forecast, window=100)
    assert isinstance(rolling, pd.Series)
    assert len(rolling) == len(forecast)
    assert (rolling >= 0).all() and (rolling <= 1).all()


def test_get_worst_realized_losses_sorted(clean_returns_1000):
    forecast = rolling_var_forecast(clean_returns_1000, method="historical", window=252)
    worst = get_worst_realized_losses(forecast, n=10)
    assert len(worst) == 10
    assert list(worst.columns) == [
        "Date", "Actual Return", "VaR Forecast", "Breach", "Loss",
    ]
    # most negative actual return first (largest loss first)
    assert worst["Actual Return"].is_monotonic_increasing
    assert worst["Loss"].iloc[0] == pytest.approx(-worst["Actual Return"].iloc[0])


def test_summarize_backtest_by_period_columns(clean_returns_1000):
    forecast = rolling_var_forecast(clean_returns_1000, method="historical", window=252)
    summary = summarize_backtest_by_period(forecast, confidence_level=0.95, freq="Y")
    assert list(summary.columns) == [
        "Period", "Observations", "Actual Breaches", "Expected Breaches",
        "Actual Breach Rate", "Expected Breach Rate",
    ]
    assert len(summary) >= 1
    assert np.allclose(summary["Expected Breach Rate"], 0.05)
    # observations across periods sum to the full backtest length
    assert summary["Observations"].sum() == len(forecast)
