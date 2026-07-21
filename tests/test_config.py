"""Configuration loading and validation contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from var_cvar_crypto_risk.config import (
    load_project_config,
    load_yaml_config,
    validate_config,
)


def _valid_config() -> dict:
    return {
        "data": {"source": "coingecko", "start_date": "2024-01-01"},
        "returns": {"method": "simple"},
        "portfolio": {
            "initial_capital": 100_000,
            "allow_short_selling": False,
            "normalize_weights": True,
        },
        "risk": {
            "confidence_level": 0.95,
            "time_horizon_days": 1,
            "var_methods": ["historical"],
            "cvar_methods": ["historical"],
        },
        "outputs": {"output_dir": "outputs"},
        "assets": {"BTC": {"coingecko_id": "bitcoin"}},
    }


def test_load_yaml_config_rejects_missing_empty_and_non_mapping(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_yaml_config(str(tmp_path / "missing.yaml"))

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Config file is empty"):
        load_yaml_config(str(empty))

    sequence = tmp_path / "sequence.yaml"
    sequence.write_text("- first\n- second\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_yaml_config(str(sequence))


def test_load_project_config_merges_assets_and_validates(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    assets_path = tmp_path / "assets.yaml"
    config_path.write_text(
        """
data: {source: coingecko, start_date: '2024-01-01'}
returns: {method: simple}
portfolio:
  initial_capital: 100000
  allow_short_selling: false
  normalize_weights: true
risk:
  confidence_level: 0.95
  time_horizon_days: 1
  var_methods: [historical]
  cvar_methods: [historical]
outputs: {output_dir: outputs}
""".strip(),
        encoding="utf-8",
    )
    assets_path.write_text(
        "assets:\n  BTC:\n    coingecko_id: bitcoin\n",
        encoding="utf-8",
    )

    loaded = load_project_config(str(config_path), str(assets_path))

    assert loaded["assets"] == {"BTC": {"coingecko_id": "bitcoin"}}
    assert loaded["risk"]["confidence_level"] == pytest.approx(0.95)


def test_load_project_config_requires_assets_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    assets_path = tmp_path / "assets.yaml"
    config_path.write_text("project: {}\n", encoding="utf-8")
    assets_path.write_text("coins: {}\n", encoding="utf-8")

    with pytest.raises(KeyError, match="top-level 'assets'"):
        load_project_config(str(config_path), str(assets_path))


@pytest.mark.parametrize(
    ("mutation", "missing_key"),
    [
        (lambda config: config.pop("data"), "data.source"),
        (lambda config: config["risk"].pop("var_methods"), "risk.var_methods"),
        (lambda config: config["outputs"].pop("output_dir"), "outputs.output_dir"),
    ],
)
def test_validate_config_reports_required_nested_key(
    mutation,
    missing_key: str,
) -> None:
    config = _valid_config()
    mutation(config)

    with pytest.raises(KeyError, match=missing_key.replace(".", r"\.")):
        validate_config(config)


@pytest.mark.parametrize("assets", [None, [], "BTC"])
def test_validate_config_requires_assets_mapping(assets) -> None:
    config = _valid_config()
    config["assets"] = assets

    with pytest.raises(KeyError, match="assets"):
        validate_config(config)


def test_validate_config_rejects_empty_assets_mapping() -> None:
    config = _valid_config()
    config["assets"] = {}

    with pytest.raises(KeyError, match="mapping is empty"):
        validate_config(config)


def test_validate_config_accepts_complete_configuration() -> None:
    validate_config(_valid_config())
