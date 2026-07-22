"""Tests for the pre-publication Git-history boundary scanner."""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.check_git_history_boundary import scan_history


def _git(repository: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _initialize_repository(path: Path, email: str) -> None:
    path.mkdir()
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "Test Author")
    _git(path, "config", "user.email", email)


def test_history_scanner_accepts_safe_repository(tmp_path: Path) -> None:
    repository = tmp_path / "safe"
    _initialize_repository(repository, "test-author@example.com")
    (repository / "README.md").write_text("# Safe fixture\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "--quiet", "-m", "safe")

    assert scan_history(repository) == []


def test_history_scanner_detects_deleted_private_path(tmp_path: Path) -> None:
    repository = tmp_path / "historical-private-path"
    _initialize_repository(repository, "test-author@example.com")
    (repository / ".env").write_text("DEMO_VALUE=placeholder\n", encoding="utf-8")
    _git(repository, "add", ".env")
    _git(repository, "commit", "--quiet", "-m", "add private path")
    _git(repository, "rm", "--quiet", ".env")
    _git(repository, "commit", "--quiet", "-m", "remove private path")

    findings = scan_history(repository)
    assert any(finding.rule == "environment_file" for finding in findings)


def test_history_scanner_detects_local_only_identity(tmp_path: Path) -> None:
    repository = tmp_path / "local-identity"
    _initialize_repository(repository, "developer@workstation.local")
    (repository / "README.md").write_text("# Identity fixture\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "--quiet", "-m", "local identity")

    rules = {finding.rule for finding in scan_history(repository)}
    assert "non_public_author_email" in rules
    assert "non_public_committer_email" in rules
