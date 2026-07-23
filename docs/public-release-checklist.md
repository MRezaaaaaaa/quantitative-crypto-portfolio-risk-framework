# Public Release Checklist

This checklist separates controls that can be verified locally from controls
that exist only after the GitHub remote is created. A workflow file in the
repository is not evidence that the workflow has run successfully.

## 1. Local repository gate

- [ ] Working tree is clean and the intended branch is identified.
- [ ] Commit author name and email are suitable for public history.
- [ ] No unrelated experiment or generated artifact is included.
- [ ] `uv lock --check` passes.
- [ ] Full tests and the numerical golden baseline pass.
- [ ] Coverage meets the current floor and does not materially regress.
- [ ] Ruff and `git diff --check` pass.
- [ ] `openspec validate --all --strict --no-interactive` passes with every
      active capability discovered by `openspec list --specs`.
- [ ] Wheel and source distribution build without dependency resolution.
- [ ] The built wheel imports in a clean environment.
- [ ] Streamlit starts and its health endpoint responds.
- [ ] Every relative Markdown link resolves.
- [ ] `python -m scripts.check_public_boundary` passes.
- [ ] `python -m scripts.check_git_history_boundary` passes across every local
      ref intended for the first public push.

The local boundary check scans tracked and non-ignored untracked files for
forbidden private artifact paths, common credential shapes, private-key
headers, and local user paths. It deliberately reports no matched value. It is
a defense-in-depth policy check, not a substitute for GitHub Secret Scanning or
a reviewed history scan.

The history check applies the same path/content rules to blobs reachable from
all local refs and rejects local-only commit email domains. Run it before the
first public push and again after any history rewrite.

## 2. Public/private boundary

Public:

- reusable quantitative source code and tests;
- synthetic or explicitly licensed demonstration data;
- sanitized configuration examples;
- methodology, model-risk, and reproducibility documentation;
- pinned CI and security workflows;
- screenshots generated only from publication-safe inputs.

Private:

- credentials, `.env` values, and deployment tokens;
- real holdings, client or investor data, account identifiers, and transaction
  history;
- private portfolio-monitoring databases and live NAV records;
- unpublished proprietary signals or strategy parameters;
- downloaded vendor datasets without redistribution rights;
- local paths, caches, scratch notebooks, and generated outputs.

Any uncertainty is resolved toward keeping the artifact private until its
ownership, license, and contents are reviewed.

## 3. GitHub remote settings

Complete these only after the remote exists:

- [ ] Confirm repository visibility and default branch before the first push.
- [ ] Enable Private Vulnerability Reporting.
- [ ] Enable Secret Scanning and Push Protection where available.
- [ ] Enable Dependabot alerts and security updates.
- [ ] Restrict GitHub Actions to approved actions and require full-SHA pinning
      where repository settings support it.
- [ ] Keep default workflow-token permissions read-only.
- [ ] Create a ruleset for `main` that blocks force pushes and deletion.
- [ ] Require pull requests, conversation resolution, and passing CI/security
      checks before merge.
- [ ] Run CI, CodeQL, and Dependency Review successfully on GitHub before
      marking them as required checks.
- [ ] Review the repository Security tab and resolve or document every alert.

Dependency Review is available on public repositories and on eligible private
repositories. Do not claim it is active until a pull request run completes.

## 4. Research-publication gate

The repository includes a completed synthetic methodology workflow and
manifest generator. The boxes below remain article-specific: check them again
for the exact dataset, config, commit, and claims used by each publication.

- [ ] Pin a legally redistributable input dataset and record its hash.
- [ ] Record configuration, data cutoff, dependency-lock hash, Git commit,
      random seeds, solver, and covariance/constraint governance diagnostics.
- [ ] Generate every article table and figure from the pinned workflow.
- [ ] Verify there is no look-ahead leakage between estimation and evaluation.
- [ ] Label in-sample optimization and overlapping observations explicitly.
- [ ] Avoid prediction, outperformance, suitability, and regulatory-compliance
      claims not supported by the implementation.
- [ ] Store an artifact manifest with output hashes.

## 5. Release gate

- [ ] Update version metadata and changelog in one focused release commit.
- [ ] Review the complete diff from the previous release tag.
- [ ] Confirm all required GitHub checks are green.
- [ ] Build release artifacts from the reviewed commit.
- [ ] Verify artifact contents and hashes.
- [ ] Create the signed or annotated tag only after all gates pass.
- [ ] Publish release notes that distinguish methodology changes from fixes,
      documentation, governance, and dependencies.
- [ ] Do not deploy, publish a package, or create a release from an unclean or
      locally modified tree.
