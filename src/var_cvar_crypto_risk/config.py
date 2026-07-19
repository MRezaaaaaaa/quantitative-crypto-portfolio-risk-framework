"""Configuration loading and validation for the risk engine.

Reads YAML configuration files and merges them into a single runtime
config dictionary used by every other module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str) -> dict:
    """Load a single YAML file and return its contents as a dict.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the YAML file is empty or does not parse to a mapping.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with file_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)

    if loaded is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping at the top level: {path}"
        )
    return loaded


def load_project_config(config_path: str, assets_path: str) -> dict:
    """Merge ``config.yaml`` and ``assets.yaml`` into a single runtime config.

    The merged config has keys: ``project``, ``data``, ``returns``,
    ``portfolio``, ``risk``, ``outputs``, and ``assets``.

    Parameters
    ----------
    config_path : str
        Path to the main ``config.yaml``.
    assets_path : str
        Path to ``assets.yaml`` describing portfolio assets.

    Returns
    -------
    dict
        Merged and validated configuration dictionary.
    """
    base = load_yaml_config(config_path)
    assets_doc = load_yaml_config(assets_path)

    if "assets" not in assets_doc:
        raise KeyError(
            f"Assets file '{assets_path}' must contain a top-level 'assets' key."
        )

    merged: dict[str, Any] = dict(base)
    merged["assets"] = assets_doc["assets"]

    validate_config(merged)
    return merged


def validate_config(config: dict) -> None:
    """Validate that all required keys are present in the merged config.

    Parameters
    ----------
    config : dict
        Merged config dictionary.

    Raises
    ------
    KeyError
        If any required key is missing, with a clear human-readable message.
    """
    required_paths: list[tuple[str, ...]] = [
        ("data", "source"),
        ("data", "start_date"),
        ("returns", "method"),
        ("portfolio", "initial_capital"),
        ("portfolio", "allow_short_selling"),
        ("portfolio", "normalize_weights"),
        ("risk", "confidence_level"),
        ("risk", "time_horizon_days"),
        ("risk", "var_methods"),
        ("risk", "cvar_methods"),
        ("outputs", "output_dir"),
    ]

    for path in required_paths:
        node: Any = config
        for key in path:
            if not isinstance(node, dict) or key not in node:
                dotted = ".".join(path)
                raise KeyError(
                    f"Missing required config key: '{dotted}'. "
                    f"Please add it to your config.yaml."
                )
            node = node[key]

    if "assets" not in config or not isinstance(config["assets"], dict):
        raise KeyError(
            "Missing required config key: 'assets'. "
            "Please ensure assets.yaml is loaded and contains an 'assets' mapping."
        )
    if len(config["assets"]) == 0:
        raise KeyError("Config 'assets' mapping is empty. Add at least one asset.")
