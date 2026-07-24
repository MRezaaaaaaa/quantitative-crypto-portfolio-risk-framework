# Security Policy

## Supported versions

Security fixes are applied to the latest `1.x` release and the default branch.
Pre-1.0 development versions are not supported.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability or exposed secret.
Use GitHub Private Vulnerability Reporting from the repository's **Security**
tab. Select **Report a vulnerability** and submit the report privately. This
feature is enabled for the public repository.

If private vulnerability reporting is temporarily unavailable, do not disclose
the issue in a public issue, discussion, pull request, or social-media post.
Wait until the private reporting channel is available.

When reporting, include the affected version, reproduction steps, potential
impact, and any suggested mitigation. Do not include real credentials, private
portfolio data, or client data in a report.

## Scope

Reports concerning credential exposure, dependency vulnerabilities, unsafe
data handling, or code execution are in scope. Financial model limitations and
investment performance disagreements are not security vulnerabilities, though
they may be reported through the normal issue process after public release.

## Repository security controls

The repository runs SHA-pinned CodeQL and Dependency Review workflows and
enforces local and Git-history public-boundary policy checks. GitHub Secret
Scanning, Push Protection, Dependabot alerts and security updates, Private
Vulnerability Reporting, and required CI/security checks on the protected
default branch are enabled. See the
[public release checklist](docs/public-release-checklist.md).
