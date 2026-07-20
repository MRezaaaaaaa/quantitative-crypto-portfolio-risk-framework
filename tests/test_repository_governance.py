"""Repository-level security and GitHub-governance contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = PROJECT_ROOT / ".github" / "workflows"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def test_all_external_github_actions_are_pinned_to_full_sha() -> None:
    unpinned: list[str] = []
    for workflow in WORKFLOWS.glob("*.yml"):
        for line_number, line in enumerate(
            workflow.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = re.search(r"\buses:\s*([^\s#]+)", line)
            if not match or match.group(1).startswith("./"):
                continue
            reference = match.group(1)
            _, separator, revision = reference.rpartition("@")
            if not separator or not FULL_SHA.fullmatch(revision):
                unpinned.append(f"{workflow.name}:{line_number}: {reference}")
    assert unpinned == []


def test_workflows_avoid_pull_request_target_and_write_by_default() -> None:
    for workflow in WORKFLOWS.glob("*.yml"):
        text = workflow.read_text(encoding="utf-8")
        assert "pull_request_target" not in text
        assert re.search(r"(?m)^permissions:\n  contents: read$", text)


def test_security_workflow_scopes_security_events_to_codeql_job() -> None:
    text = (WORKFLOWS / "security.yml").read_text(encoding="utf-8")
    assert text.count("security-events: write") == 1
    assert "github/codeql-action/init@" in text
    assert "github/codeql-action/analyze@" in text
    assert "actions/dependency-review-action@" in text


def test_ci_runs_public_boundary_check() -> None:
    text = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert "python -m scripts.check_public_boundary" in text


def test_issue_forms_are_parseable_and_disable_blank_issues() -> None:
    issue_dir = PROJECT_ROOT / ".github" / "ISSUE_TEMPLATE"
    config = yaml.safe_load((issue_dir / "config.yml").read_text(encoding="utf-8"))
    assert config["blank_issues_enabled"] is False

    forms = sorted(issue_dir.glob("*.yml"))
    forms.remove(issue_dir / "config.yml")
    assert forms
    for form in forms:
        payload = yaml.safe_load(form.read_text(encoding="utf-8"))
        assert payload["name"]
        assert payload["description"]
        assert payload["body"]
