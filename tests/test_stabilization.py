"""Stabilization tests for the Phase 3 cleanup pass.

These tests guard against regressions where importing core analytics
accidentally pulls in optional data-source dependencies (yfinance, the
CoinGecko HTTP client) and where ``export.py`` helpers fail on missing
parent directories.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_in_clean_subprocess(snippet: str) -> subprocess.CompletedProcess:
    """Execute ``snippet`` in a clean subprocess that uses the project ``src``."""
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})
        {snippet}
        """
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )


def test_package_import_is_lightweight() -> None:
    """``import var_cvar_crypto_risk`` must not eagerly load yfinance or
    the coingecko client.
    """
    result = _run_in_clean_subprocess(
        """
        import var_cvar_crypto_risk  # noqa: F401
        assert 'yfinance' not in sys.modules, 'yfinance was imported eagerly'
        assert 'var_cvar_crypto_risk.yfinance_client' not in sys.modules
        assert 'var_cvar_crypto_risk.coingecko_client' not in sys.modules
        assert 'var_cvar_crypto_risk.data_loader' not in sys.modules
        print('OK')
        """
    )
    assert result.returncode == 0, (
        f"Subprocess failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_backtesting_import_without_yfinance() -> None:
    """``import var_cvar_crypto_risk.backtesting`` must not pull in yfinance."""
    result = _run_in_clean_subprocess(
        """
        import var_cvar_crypto_risk.backtesting  # noqa: F401
        from var_cvar_crypto_risk.backtesting import (
            backtest_var_model,
            rolling_var_forecast,
        )
        assert 'yfinance' not in sys.modules
        assert 'var_cvar_crypto_risk.yfinance_client' not in sys.modules
        print('OK')
        """
    )
    assert result.returncode == 0, (
        f"Subprocess failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_demo_script_imports() -> None:
    """The unified demo script must be importable as a module (its
    imports resolve) without executing ``main``."""
    demo_path = PROJECT_ROOT / "run_demo.py"
    assert demo_path.exists(), f"Missing demo script: {demo_path}"

    spec = importlib.util.spec_from_file_location("run_demo", str(demo_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
    assert callable(module.main)


def test_export_creates_parent_dirs(tmp_path: Path) -> None:
    """``save_dataframe`` / ``save_series`` / ``save_json`` MUST create
    missing parent directories without raising."""
    from var_cvar_crypto_risk.export import (
        ensure_output_dirs,
        save_dataframe,
        save_json,
        save_series,
    )

    nested = tmp_path / "deep" / "nested" / "missing"

    ensure_output_dirs(str(nested))
    assert (nested / "tables").is_dir()
    assert (nested / "charts").is_dir()
    assert (nested / "reports").is_dir()

    df = pd.DataFrame(
        {"x": [1.0, 2.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    df_path = tmp_path / "auto" / "child" / "df.csv"
    save_dataframe(df, str(df_path))
    assert df_path.exists()
    loaded = pd.read_csv(df_path, index_col=0)
    assert list(loaded.columns) == ["x"]

    ser = pd.Series(
        [10.0, 20.0],
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
        name="value",
    )
    ser_path = tmp_path / "auto" / "series" / "s.csv"
    save_series(ser, str(ser_path))
    assert ser_path.exists()

    json_path = tmp_path / "auto" / "json" / "blob.json"
    save_json({"a": 1, "b": [1, 2, 3]}, str(json_path))
    assert json_path.exists()
    text = json_path.read_text(encoding="utf-8")
    assert '"a": 1' in text


def test_backtesting_works_without_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even if yfinance is removed from sys.modules, the backtesting
    pipeline still runs on a synthetic return series.
    """
    monkeypatch.setitem(sys.modules, "yfinance", None)

    from var_cvar_crypto_risk.backtesting import backtest_var_model

    rng_seed = 11
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    returns = pd.Series(
        rng.standard_t(df=4, size=400) * 0.02,
        index=pd.date_range("2023-01-01", periods=400, freq="D"),
    )

    forecast_df, result = backtest_var_model(
        returns, method="historical", confidence_level=0.95, window=100
    )
    assert len(forecast_df) == 400 - 100
    assert result["observations"] == 400 - 100
    assert "traffic_light" in result
