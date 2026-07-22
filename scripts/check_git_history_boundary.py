"""Audit every reachable Git commit before the repository's first public push.

The scanner reuses the working-tree public-boundary rules, but reads blobs
directly from Git so deleting a secret from the latest checkout cannot hide it.
It reports only commit, path, and rule metadata; matched values are never
printed. Local-only author or committer email domains are also rejected.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from scripts.check_public_boundary import (
    MAX_TEXT_FILE_BYTES,
    scan_path,
    scan_text,
)


@dataclass(frozen=True)
class HistoryFinding:
    """One path, content, symlink, or identity issue in Git history."""

    commit: str
    path: str
    rule: str
    message: str
    line: int | None = None


def _git(repository: Path, *args: str, text: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=text,
    )
    return result.stdout


def _is_public_email(email: str) -> bool:
    normalized = email.strip().lower()
    if "@" not in normalized:
        return False
    domain = normalized.rsplit("@", 1)[1]
    return bool(domain) and not domain.endswith(".local") and domain != "localhost"


def _commit_identity_findings(repository: Path, commit: str) -> list[HistoryFinding]:
    raw = str(
        _git(repository, "show", "-s", "--format=%ae%x00%ce", commit, text=True)
    ).rstrip("\n")
    author_email, _, committer_email = raw.partition("\x00")
    findings: list[HistoryFinding] = []
    if not _is_public_email(author_email):
        findings.append(
            HistoryFinding(
                commit=commit,
                path="<commit-author>",
                rule="non_public_author_email",
                message="commit author email is missing or uses a local-only domain",
            )
        )
    if not _is_public_email(committer_email):
        findings.append(
            HistoryFinding(
                commit=commit,
                path="<commit-committer>",
                rule="non_public_committer_email",
                message="commit committer email is missing or uses a local-only domain",
            )
        )
    return findings


def _tree_entries(repository: Path, commit: str) -> list[tuple[str, str, str]]:
    raw = bytes(_git(repository, "ls-tree", "-r", "-z", "--full-tree", commit))
    entries: list[tuple[str, str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        if object_type != "blob":
            continue
        entries.append(
            (
                mode,
                object_id,
                encoded_path.decode("utf-8", errors="surrogateescape"),
            )
        )
    return entries


def _external_symlink_target(path: str, target: str) -> bool:
    if target.startswith(("/", "\\")):
        return True
    base_parts = list(PurePosixPath(path).parent.parts)
    for part in PurePosixPath(target.replace("\\", "/")).parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not base_parts:
                return True
            base_parts.pop()
        else:
            base_parts.append(part)
    return False


def scan_history(repository: Path) -> list[HistoryFinding]:
    """Scan paths and contents from every commit reachable through local refs."""
    root = repository.resolve()
    commits_output = str(_git(root, "rev-list", "--reverse", "--all", text=True))
    commits = [line for line in commits_output.splitlines() if line]
    findings: list[HistoryFinding] = []
    scanned_blobs: set[tuple[str, str]] = set()

    for commit in commits:
        short_commit = commit[:12]
        findings.extend(_commit_identity_findings(root, short_commit))
        for mode, object_id, path in _tree_entries(root, short_commit):
            key = (object_id, path)
            if key in scanned_blobs:
                continue
            scanned_blobs.add(key)

            for finding in scan_path(path):
                findings.append(
                    HistoryFinding(
                        commit=short_commit,
                        path=finding.path,
                        line=finding.line,
                        rule=finding.rule,
                        message=finding.message,
                    )
                )

            content = bytes(_git(root, "cat-file", "blob", object_id))
            if len(content) > MAX_TEXT_FILE_BYTES:
                findings.append(
                    HistoryFinding(
                        commit=short_commit,
                        path=path,
                        rule="oversized_historical_file",
                        message=f"historical file exceeds {MAX_TEXT_FILE_BYTES} bytes",
                    )
                )
                continue
            if b"\0" in content:
                continue
            text = content.decode("utf-8", errors="replace")
            if mode == "120000" and _external_symlink_target(path, text):
                findings.append(
                    HistoryFinding(
                        commit=short_commit,
                        path=path,
                        rule="external_historical_symlink",
                        message="historical symlink resolves outside the repository",
                    )
                )
            for finding in scan_text(path, text):
                findings.append(
                    HistoryFinding(
                        commit=short_commit,
                        path=finding.path,
                        line=finding.line,
                        rule=finding.rule,
                        message=finding.message,
                    )
                )
    return findings


def _format_finding(finding: HistoryFinding) -> str:
    location = f"{finding.commit}:{finding.path}"
    if finding.line is not None:
        location += f":{finding.line}"
    return f"{location}: [{finding.rule}] {finding.message}"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="Git repository to scan (default: current directory).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        findings = scan_history(args.repository)
    except subprocess.CalledProcessError:
        print("Git-history boundary check failed: unable to inspect repository.")
        return 2
    if findings:
        print(f"Git-history boundary check failed with {len(findings)} finding(s):")
        for finding in findings:
            print(_format_finding(finding))
        return 1
    print("Git-history boundary check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
