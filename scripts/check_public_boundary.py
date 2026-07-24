"""Fail when files prepared for publication cross the public/private boundary.

This lightweight repository policy check complements, but does not replace,
provider-backed secret scanning and push protection on GitHub. It reports only
file, line, and rule names so a detected credential is never repeated in CI
logs.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


MAX_TEXT_FILE_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class Finding:
    """One publication-boundary policy violation."""

    path: str
    line: int | None
    rule: str
    message: str


_CONTENT_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "private_key",
        re.compile(r"-{5}BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-{5}"),
        "private-key material must not be published",
    ),
    (
        "aws_access_key",
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
        "AWS access-key-shaped value detected",
    ),
    (
        "github_token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
        "GitHub token-shaped value detected",
    ),
    (
        "openai_api_key",
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        "API-key-shaped value detected",
    ),
    (
        "generic_secret_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|"
            r"client[_-]?secret|password)\b\s*[:=]\s*[\"']?"
            r"[A-Za-z0-9_./+=:-]{16,}"
        ),
        "credential-like assignment detected",
    ),
    (
        "local_user_path",
        re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/"),
        "local user path must not appear in public artifacts",
    ),
    (
        "codex_workspace_path",
        re.compile(r"(?:^|[/\\])\.codex[/\\]"),
        "local Codex workspace path must not be published",
    ),
)


def scan_path(relative_path: str) -> list[Finding]:
    """Check whether a repository-relative path is safe to publish."""
    normalized = PurePosixPath(relative_path.replace("\\", "/"))
    path_text = normalized.as_posix()
    name = normalized.name
    findings: list[Finding] = []

    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        findings.append(
            Finding(path_text, None, "environment_file", "private environment file")
        )

    if path_text == ".streamlit/secrets.toml":
        findings.append(
            Finding(path_text, None, "streamlit_secrets", "Streamlit secrets file")
        )

    if name.lower() in {"credentials.json", "service-account.json"}:
        findings.append(
            Finding(path_text, None, "credential_file", "credential file")
        )

    if normalized.suffix.lower() in {
        ".db",
        ".key",
        ".p12",
        ".pem",
        ".pfx",
        ".sqlite",
        ".sqlite3",
    }:
        findings.append(
            Finding(
                path_text,
                None,
                "private_artifact_type",
                "private key, credential, or database artifact",
            )
        )

    private_prefixes = ("data/cache/", "data/processed/", "data/raw/", "outputs/")
    if path_text.endswith("/.gitkeep"):
        return findings
    if path_text.startswith(private_prefixes):
        findings.append(
            Finding(
                path_text,
                None,
                "generated_or_private_data",
                "generated or private data directory",
            )
        )
    return findings


def scan_text(relative_path: str, text: str) -> list[Finding]:
    """Scan text without including matched values in returned messages."""
    findings: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for rule, pattern, message in _CONTENT_RULES:
            if pattern.search(line):
                findings.append(
                    Finding(relative_path, line_number, rule, message)
                )
    return findings


def _candidate_files(repository: Path) -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    return sorted(
        item.decode("utf-8")
        for item in completed.stdout.split(b"\0")
        if item
    )


def scan_repository(repository: Path) -> list[Finding]:
    """Scan tracked and non-ignored untracked files intended for publication."""
    root = repository.resolve()
    findings: list[Finding] = []
    for relative_path in _candidate_files(root):
        findings.extend(scan_path(relative_path))
        full_path = root / relative_path
        if full_path.is_symlink():
            target = os.readlink(full_path)
            resolved_target = full_path.resolve()
            if Path(target).is_absolute() or not resolved_target.is_relative_to(root):
                findings.append(
                    Finding(
                        relative_path,
                        None,
                        "external_symlink",
                        "symbolic link resolves outside the repository",
                    )
                )
            findings.extend(scan_text(relative_path, target))
            continue
        if not full_path.is_file():
            continue
        size = full_path.stat().st_size
        if size > MAX_TEXT_FILE_BYTES:
            findings.append(
                Finding(
                    relative_path,
                    None,
                    "oversized_file",
                    f"file exceeds {MAX_TEXT_FILE_BYTES} bytes",
                )
            )
            continue
        content = full_path.read_bytes()
        if b"\0" in content:
            continue
        findings.extend(
            scan_text(relative_path, content.decode("utf-8", errors="replace"))
        )
    return findings


def _format_finding(finding: Finding) -> str:
    location = finding.path
    if finding.line is not None:
        location += f":{finding.line}"
    return f"{location}: [{finding.rule}] {finding.message}"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: current directory)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    findings = scan_repository(args.repository)
    if findings:
        print("Public-boundary check failed:")
        for finding in findings:
            print(f"- {_format_finding(finding)}")
        return 1
    print("Public-boundary check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
