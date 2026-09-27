"""Publication workflow integrity, determinism, and boundary tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import publication_workflow as workflow


CONFIG_PATH = (
    workflow.PROJECT_ROOT / "publication" / "configs" / "methodology_demo_v1.yaml"
)


@pytest.fixture(scope="module")
def generated_bundles(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("publication-bundles")
    first = root / "first"
    second = root / "second"
    workflow.generate_publication_bundle(CONFIG_PATH, first, allow_dirty=True)
    workflow.generate_publication_bundle(CONFIG_PATH, second, allow_dirty=True)
    return first, second


def test_publication_bundle_is_byte_deterministic(
    generated_bundles: tuple[Path, Path],
) -> None:
    first, second = generated_bundles
    first_files = sorted(path.name for path in first.iterdir())
    second_files = sorted(path.name for path in second.iterdir())
    assert first_files == second_files
    assert first_files == sorted([*workflow._GENERATED_FILENAMES, "manifest.json"])
    for name in first_files:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_manifest_verification_and_publication_boundary(
    generated_bundles: tuple[Path, Path],
) -> None:
    first, _ = generated_bundles
    result = workflow.verify_publication_manifest(first / "manifest.json")
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    all_text = "\n".join(path.read_text(encoding="utf-8") for path in first.iterdir())

    assert result["verified"] is True
    assert result["artifact_count"] == 10
    assert manifest["experiment"]["claims_boundary"] == "synthetic_methodology_only"
    assert manifest["generation"]["offline"] is True
    assert (
        manifest["data"]["used_end_date"] <= manifest["data"]["configured_cutoff_date"]
    )
    assert manifest["data"]["source_end_date"] > manifest["data"]["used_end_date"]
    assert manifest["bias_controls"]["optimization"].startswith(
        "Optimization is in-sample"
    )
    assert "/Users/" not in all_text
    assert ".codex" not in all_text
    assert "COINGECKO_API_KEY" not in all_text
    assert "private_holdings" not in all_text.lower()


def test_publication_uses_authoritative_historical_risk_summary_contract(
    generated_bundles: tuple[Path, Path],
) -> None:
    first, _ = generated_bundles
    risk_summary = pd.read_csv(first / "risk_summary.csv")
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    experiment = json.loads(
        (first / "experiment_summary.json").read_text(encoding="utf-8")
    )

    assert list(risk_summary.columns) == [
        "Record Type",
        "Section",
        "Name",
        "Value",
        "Display Value",
        "Unit",
        "Sample Size",
    ]
    metric_names = set(
        risk_summary.loc[risk_summary["Record Type"] == "Metric", "Name"]
    )
    assert "Annualized Return" not in metric_names
    assert "Annualized Volatility" not in metric_names
    assert "Max Drawdown" not in metric_names
    assert "Maximum Drawdown" in metric_names
    assert not any("Sharpe" in name for name in metric_names)

    path_contract = manifest["assumptions"]["portfolio_path"]
    assert path_contract["policy"] == "buy_and_hold"
    assert path_contract["commission_bps"] == 0.0
    assert path_contract["slippage_bps"] == 0.0
    assert path_contract["methodology_version"] == "portfolio-path-v1"
    assert path_contract["cutoff_date"] == manifest["data"]["used_end_date"]
    assert path_contract["policy_defaulted_from_schema_v1"] is False
    assert experiment["portfolio"]["initial_capital"] == 100_000.0
    assert experiment["portfolio"]["ending_net_nav"] > 0.0


def test_publication_summary_includes_first_period_loss_from_explicit_path() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    wealth = np.array(
        [100.0, 80.0, 90.0, 95.0, 100.0, 105.0, 103.0, 106.0, 108.0, 110.0]
    )
    prices = pd.DataFrame({"BTC": wealth, "ETH": wealth, "SOL": wealth}, index=index)
    metadata = {
        "used_end_date": "2024-01-10",
        "path": "tests/fixtures/synthetic_daily_prices.csv",
    }
    path, summary, defaulted = workflow._build_publication_historical_summary(
        config,
        prices,
        pd.Series(config["portfolio"]["weights"], dtype=float),
        metadata,
    )
    primary = summary.primary.set_index("Metric")["Value"]

    assert defaulted is False
    assert primary["Maximum Drawdown"] == pytest.approx(-0.20)
    assert primary["Ending Net NAV"] == pytest.approx(path.latest_net_nav)
    assert primary["Net Cumulative Return"] == pytest.approx(
        path.latest_net_nav / config["portfolio"]["initial_capital"] - 1.0
    )


def test_schema_v1_missing_policy_defaults_to_buy_and_hold_and_is_recorded() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["portfolio"].pop("evolution_policy")
    config["portfolio"].pop("commission_bps")
    config["portfolio"].pop("slippage_bps")

    path_config, defaulted = workflow._publication_portfolio_path_config(config)

    assert path_config.policy.value == "buy_and_hold"
    assert path_config.commission_bps == 0.0
    assert path_config.slippage_bps == 0.0
    assert defaulted is True


def test_manifest_detects_tampered_artifact(
    generated_bundles: tuple[Path, Path],
) -> None:
    _, second = generated_bundles
    artifact = second / "risk_summary.csv"
    artifact.write_text(artifact.read_text(encoding="utf-8") + "tampered\n")

    with pytest.raises(workflow.PublicationWorkflowError, match="hash mismatch"):
        workflow.verify_publication_manifest(second / "manifest.json")


def test_manifest_rejects_unlisted_file(
    generated_bundles: tuple[Path, Path], tmp_path: Path
) -> None:
    first, _ = generated_bundles
    copied = tmp_path / "bundle"
    copied.mkdir()
    for source in first.iterdir():
        (copied / source.name).write_bytes(source.read_bytes())
    (copied / "unlisted.csv").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(workflow.PublicationWorkflowError, match="unexpected artifact"):
        workflow.verify_publication_manifest(copied / "manifest.json")


def test_generation_refuses_dirty_tree_without_preview_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        workflow,
        "_git_metadata",
        lambda: {
            "commit": "0" * 40,
            "branch": "test",
            "dirty_at_generation_start": True,
        },
    )
    with pytest.raises(workflow.PublicationWorkflowError, match="Repository is dirty"):
        workflow.generate_publication_bundle(CONFIG_PATH, tmp_path / "bundle")


def test_generation_refuses_dataset_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["data"]["expected_sha256"] = "0" * 64
    monkeypatch.setattr(workflow, "load_publication_config", lambda _: config)
    with pytest.raises(workflow.PublicationWorkflowError, match="hash mismatch"):
        workflow.generate_publication_bundle(
            CONFIG_PATH,
            tmp_path / "bundle",
            allow_dirty=True,
        )


def test_generation_refuses_non_simple_returns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["returns"]["method"] = "log"
    monkeypatch.setattr(workflow, "load_publication_config", lambda _: config)
    with pytest.raises(workflow.PublicationWorkflowError, match="simple returns"):
        workflow.generate_publication_bundle(
            CONFIG_PATH,
            tmp_path / "bundle",
            allow_dirty=True,
        )


def test_generation_refuses_unexpected_output_file(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "private-holdings.csv").write_text("not-public\n", encoding="utf-8")

    with pytest.raises(workflow.PublicationWorkflowError, match="unexpected files"):
        workflow.generate_publication_bundle(
            CONFIG_PATH,
            output,
            allow_dirty=True,
            overwrite=True,
        )
