"""File-export helpers for tables, series, and JSON payloads."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ensure_output_dirs(base_dir: str) -> None:
    """Create ``outputs/{tables,charts,reports}`` if they don't exist."""
    base = Path(base_dir)
    for sub in ("tables", "charts", "reports"):
        (base / sub).mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """Save ``df`` to CSV. Creates parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=True)


def save_series(series: pd.Series, path: str) -> None:
    """Save ``series`` to CSV. Creates parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(out, index=True, header=True)


def save_json(data: dict, path: str) -> None:
    """Save ``data`` as pretty-printed JSON. Creates parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
