"""Streamlit session-state test for the asset editor.

Validates that the asset-table initialization does not overwrite an existing
``st.session_state["assets_df"]`` — i.e. user edits survive reruns. Uses
Streamlit's AppTest harness and needs no network (the data fetch only fires
on the "Run risk analysis" click, which this test does not perform).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

from var_cvar_crypto_risk.monitoring.database import create_monitoring_engine
from var_cvar_crypto_risk.monitoring.models import Base

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402


def test_asset_table_init_does_not_overwrite_session_state() -> None:
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception
    assert "assets_df" in at.session_state

    # Simulate a user edit landing in session_state, then rerun. The init
    # guard must NOT clobber the existing (edited) frame.
    marker = pd.DataFrame(
        {
            "Symbol": ["ZZZ"],
            "CoinGecko ID": ["zzz-id"],
            "yfinance Ticker": ["ZZZ-USD"],
            "Weight": [1.0],
        }
    )
    at.session_state["assets_df"] = marker
    at.run()
    assert not at.exception
    assert list(at.session_state["assets_df"]["Symbol"]) == ["ZZZ"]


def test_return_handling_defaults_to_automatic_and_exposes_advanced_log() -> None:
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception

    return_handling = next(
        widget for widget in at.selectbox if widget.label == "Return handling"
    )
    assert return_handling.value == "automatic"
    assert not any(
        widget.label == "Diagnostic return convention"
        for widget in at.selectbox
    )

    return_handling.set_value("advanced").run()
    assert not at.exception
    diagnostic = next(
        widget
        for widget in at.selectbox
        if widget.label == "Diagnostic return convention"
    )
    diagnostic.set_value("log").run()
    assert not at.exception
    assert diagnostic.value == "log"


def test_legacy_return_state_is_invalidated() -> None:
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception
    at.session_state["risk_results"] = {"returns_method": "log"}

    at.run()

    assert not at.exception
    assert at.session_state["risk_results"] is None


def test_monitoring_workspace_has_a_safe_uninitialized_database_state(
    tmp_path: Path, monkeypatch
) -> None:
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'empty.db').as_posix()}"
    monkeypatch.setenv("QCPRF_MONITORING_DATABASE_URL", database_url)

    at = AppTest.from_file("app.py", default_timeout=60).run()
    workspace = next(widget for widget in at.radio if widget.label == "Workspace")
    workspace.set_value("Portfolio Monitor").run()

    assert not at.exception
    assert any(
        "monitoring database has not been initialized" in item.value.lower()
        for item in at.info
    )
    rendered_text = "\n".join(item.value for item in at.caption)
    assert tmp_path.as_posix() not in rendered_text
    assert "sqlite/empty.db" in rendered_text


def test_monitoring_workspace_handles_a_migrated_empty_registry(
    tmp_path: Path, monkeypatch
) -> None:
    database_path = tmp_path / "migrated.db"
    engine = create_monitoring_engine(
        f"sqlite+pysqlite:///{database_path.as_posix()}"
    )
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        connection.execute(
            text(
                "INSERT INTO alembic_version (version_num) "
                "VALUES ('0004_batch5_run_metadata')"
            )
        )
    engine.dispose()
    monkeypatch.setenv(
        "QCPRF_MONITORING_DATABASE_URL",
        f"sqlite+pysqlite:///{database_path.as_posix()}",
    )

    at = AppTest.from_file("app.py", default_timeout=60).run()
    workspace = next(widget for widget in at.radio if widget.label == "Workspace")
    workspace.set_value("Portfolio Monitor").run()

    assert not at.exception
    assert any("No experiments found" in item.value for item in at.info)

    view = next(widget for widget in at.radio if widget.label == "Monitoring view")
    view.set_value("Create Forward Test").run()
    assert not at.exception
    assert any(
        "Post-launch policy" in item.value
        for item in at.markdown
    )

    view = next(widget for widget in at.radio if widget.label == "Monitoring view")
    view.set_value("Comparison").run()
    assert not at.exception
    assert any("At least two experiments" in item.value for item in at.info)
