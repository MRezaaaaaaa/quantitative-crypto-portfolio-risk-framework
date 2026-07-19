"""Streamlit session-state test for the asset editor (Phase 5.5, G1).

Validates that the asset-table initialization does not overwrite an existing
``st.session_state["assets_df"]`` — i.e. user edits survive reruns. Uses
Streamlit's AppTest harness and needs no network (the data fetch only fires
on the "Run risk analysis" click, which this test does not perform).
"""

from __future__ import annotations

import pandas as pd
import pytest

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
