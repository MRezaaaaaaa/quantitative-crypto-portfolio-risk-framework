"""Publication workflow integrity, determinism, and boundary tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts import publication_workflow as workflow


CONFIG_PATH = (
    workflow.PROJECT_ROOT
    / "publication"
    / "configs"
    / "methodology_demo_v1.yaml"
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
    all_text = "\n".join(
        path.read_text(encoding="utf-8") for path in first.iterdir()
    )

    assert result["verified"] is True
    assert result["artifact_count"] == 10
    assert manifest["experiment"]["claims_boundary"] == "synthetic_methodology_only"
    assert manifest["generation"]["offline"] is True
    assert manifest["data"]["used_end_date"] <= manifest["data"]["configured_cutoff_date"]
    assert manifest["data"]["source_end_date"] > manifest["data"]["used_end_date"]
    assert manifest["bias_controls"]["optimization"].startswith(
        "Optimization is in-sample"
    )
    assert "/Users/" not in all_text
    assert ".codex" not in all_text
    assert "COINGECKO_API_KEY" not in all_text
    assert "private_holdings" not in all_text.lower()


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
