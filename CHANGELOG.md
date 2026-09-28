# Changelog

All notable changes to this project will be documented in this file. The format
is based on Keep a Changelog, and the project follows Semantic Versioning.

## [Unreleased]

## [2.0.0] - 2026-09-28

### Breaking Changes

- Public chart builders now return `plotly.graph_objects.Figure` objects.
  Matplotlib is no longer a runtime dependency, so downstream integrations that
  expected Matplotlib figures must migrate to Plotly rendering or export APIs.
- Removed the standalone `plot_tail_zoom_distribution` builder. The primary
  interactive distribution figure now owns the historical tail overlay and
  VaR/CVaR reference lines.

### Added

- Strictly retrospective Risk Summary sourced from the finalized selected
  policy path, including audited net NAV/drawdown/cost cards, historical
  descriptive/tail tables, a long-form raw/display CSV contract, policy and
  target-weight context, and explicit source/calendar data-quality details.
- Shared Streamlit-independent Plotly chart system for all user-facing Risk Lab
  distributions, paths, dependence, backtesting, simulation and optimization
  figures, with provenance metadata, browser PNG and standalone HTML export.
- Interactive Robust Assumptions dumbbell audit comparing Raw Historical Mean
  with Final E[r] or another already-computed estimator, including basis-point
  differences, recipe hover details, null-safe Manual Views and explicit asset
  ordering.
- Default All Estimators audit mode showing the full completed return-estimator
  range per asset with stable color/symbol identities, overlap-safe vertical
  offsets, full-dispersion sorting and complete Plotly metadata; the prior
  Raw-Mean pairwise comparison remains available as a separate view mode.

- Versioned pure Portfolio Path V1 engine for fixed-quantity Buy & Hold and
  explicit daily/weekly/monthly/quarterly UTC-close rebalancing, with deferred
  missing-price events, asset-level holdings/trades/drift, accounting checks,
  gross/one-way turnover, proportional commission/slippage and provenance.
- Risk Lab Plotly comparisons for Hold versus selected net NAV, pre/post-trade
  weights and event markers, turnover/cumulative costs, comparative drawdown,
  and gross versus net NAV.
- Deterministic portfolio-path, chart, missing-data and Streamlit policy tests,
  including zero-cost daily equivalence to the legacy constant-weight series.

- Persistent named portfolio experiments with authoritative UUIDs, validated
  lifecycle transitions, retained archive history, and event audit records.
- Immutable point-in-time allocation snapshots containing target allocations,
  manual validation state, versions, dates, source hashes, and available launch
  risk forecasts. Legacy optimized records remain readable without rewriting
  their original solver and residual provenance.
- SQLAlchemy repository/unit-of-work boundary, Alembic migrations, and a private
  local SQLite monitoring store configurable through a sanitized database URL.
- Historical Out-of-Sample Replay that rebuilds through a frozen cutoff and
  reveals evaluation observations sequentially without reusing Streamlit
  optimizer session results.
- Idempotent one-shot Live Forward and Hybrid update services and the
  `qcprf-monitor` CLI for an operator-controlled external scheduler.
- Strict monitoring-price normalization without silent forward fill, explicit
  complete/incomplete states, actual provider/cutoff provenance, atomic writes,
  and sanitized failure records.
- Fixed-quantity daily NAV, benchmark, cash, current weights, allocation drift,
  drawdown, expanding realized volatility, and origin-safe VaR/CVaR forecast
  evaluation.
- Private-by-default CSV/JSON experiment bundles and manifests with file hashes
  and no database credentials or URLs.
- Streamlit Portfolio Monitor with experiment creation/archive, methodology and
  provenance views, Data Quality/Update Now, downloads, and explicit calendar
  or days-since-launch comparison.
- Plotly NAV, 100% stacked allocation, target/current, drift, drawdown,
  VaR/CVaR, exception, forecast/realized, and experiment-comparison charts.
- Portfolio-monitoring, forward-testing, database-operations, and user-guide
  documentation plus updated architecture, methodology, model-risk,
  reproducibility, data-provenance, and public-release boundaries.
- Offline tests for migrations, persistence, immutability, rollback,
  point-in-time replay, live append, idempotency, data quality, forecasts,
  fixed-holdings valuation, charts, comparison alignment, UI states, and exports.

### Changed

- Removed the redundant standalone Left-tail Zoom expander from the Distribution
  section. The primary Plotly histogram retains its highlighted historical tail
  and VaR/CVaR reference lines, with explicitly shared bins for exact overlay
  alignment.
- Removed annualized return, annualized volatility, Sharpe and duplicate
  drawdown from the visible Risk Summary and its CSV. Historical VaR/CVaR models
  are unchanged and now sit under an explicit Historical Tail Distribution
  interpretation boundary.
- Promoted Plotly to a core dependency and removed Matplotlib from the runtime
  dependency set. Pearson/Spearman selection now also governs rolling
  correlation instead of only the static matrix.

- Risk Lab realized performance, descriptive risk, backtesting and exports now
  use the selected explicit portfolio path. Monte Carlo/current-portfolio
  comparisons use current end-of-path weights; current Money VaR/CVaR and
  simulated value paths use latest net NAV rather than launch notional.
- Risk horizon and portfolio rebalancing are separate controls. Stateful paths
  preserve incomplete closes for audited deferral instead of silently forward-
  filling a scheduled event. Portfolio Monitor remains manual fixed holdings.

- Phase 8 monitoring creation is manual-only: explicit asset/percentage inputs
  replace optimizer, expected-return, covariance and scenario controls. Weights
  must total 100% without hidden normalization; fixed quantities use explicitly
  chosen complete launch prices. No optimization runs in new experiment creation.
- Manual snapshots use `manual_validated` and solver `none`, without fabricated
  return forecasts. Existing optimized records remain intact and compatible;
  Risk Lab optimization, financial formulas and database schema are unchanged.
- Added bounded SQLAlchemy, Alembic, and Plotly dependencies while preserving
  the existing financial methodology and version 1.0.0 numerical golden values.
- Split Streamlit into a default Risk Lab workspace and a persistent Portfolio
  Monitor workspace without changing the default analytical workflow.

### Fixed

- Periodic rebalance schedules now use theoretical UTC calendar boundaries.
  Missing or incomplete weekly/monthly/quarterly boundaries defer forward to
  the first later complete close instead of backdating to an earlier observation;
  multiple pending boundaries are all audited but execute one trade and one cost.
- Asset-level drawdown now includes launch wealth `1.0` as the initial peak, so
  a first-period loss is reported immediately and consistently with NAV paths.
- The publication workflow now uses the same explicit policy-specific historical
  summary contract as Streamlit. Schema-v1 configs retain a recorded Buy & Hold,
  zero-cost compatibility default.
- Source rows dated on or after the current UTC date are retained for audit but
  excluded from finalized historical NAV, drawdown, tail metrics, and counts.

### Security

- Upgraded the transitive GitPython runtime from 3.1.58 to 3.1.59, resolving
  five Dependabot advisories: one critical, two high and two medium.
- Extended ignore and publication-boundary rules to reject monitoring databases,
  SQLite sidecars, secret-bearing database URLs, private portfolio artifacts,
  and local monitoring outputs.

## [1.0.0] - 2026-07-24

### Added

- Pre-publication Git-history scanner covering deleted private paths, secret-
  shaped content, external symlinks, oversized historical files, and local-only
  commit identities without printing matched values.
- Deterministic offline publication workflow with a pinned synthetic dataset,
  cutoff enforcement, experiment config, article-to-app mapping, and
  reproducible CSV/SVG outputs.
- Publication artifact manifest recording source, dependency-lock, config,
  dataset, solver-validation, bias-control, and output hashes, plus a strict
  verification command.
- Publication controls that reject dirty-tree production runs, unreviewed data
  hashes, unexpected output-directory files, and non-simple optimization input.
- Pinned CodeQL and Dependency Review workflows with least-privilege job
  permissions and explicit timeouts.
- Tested public/private repository-boundary scanner integrated into CI.
- Pull-request, bug, feature, and methodology-review templates plus structured
  release-note categories.
- Public release checklist separating local gates from remote-only GitHub
  security settings.
- Covariance symmetry, eigenvalue, conditioning, and positive-definiteness
  diagnostics with deterministic correlation-space repair before parametric
  simulation.
- Independent optimizer residual validation for budget, bounds, target return,
  CVaR caps, and Rockafellar-Uryasev auxiliary constraints.
- Streamlit and command-line diagnostics separating raw solver status from the
  accepted public optimization status.
- Centralized Automatic/Advanced return-policy resolver and explicit
  Simple-return API guards for scenario, path, and optimization boundaries.
- Exact portfolio Log-return aggregation for advanced diagnostics, with tests
  preventing weighted-Log approximations.
- Public Simple/Log return-convention and calculation-boundary documentation.
- Canonical signed-loss conversion helpers and contract tests for VaR, CVaR,
  return thresholds, and monetary values.
- Public documentation for VaR/CVaR sign, unit, horizon, backtesting, and
  optimization conventions.
- Repository governance and release-hygiene files.
- Focused architecture, methodology, model-risk, data-provenance, and
  reproducibility documentation.
- Centralized dependency declarations and compatibility requirement entry
  points.
- GitHub Actions CI for the supported Python matrix, lint, coverage, package
  build, and Streamlit startup checks.
- Dependabot configuration for Python and GitHub Actions dependencies.
- Cross-platform `uv.lock` for Python 3.10 through 3.13.
- Synthetic daily-price fixture and reviewed numerical golden baseline covering
  returns, VaR/CVaR, robust assumptions, covariance, Monte Carlo, and
  backtesting outputs.
- Source-distribution manifest containing the documentation, configuration,
  maintenance scripts, complete test support files, and synthetic fixtures.
- Offline data-boundary tests covering configuration, preprocessing, CSV
  validation, CoinGecko retry/cache behavior, yfinance response shapes, and
  provider fallback without live API calls.
- Contract tests for portfolio weights, return horizons, covariance and
  correlation validation, and insufficient-sample VaR/CVaR failures.

### Changed

- Raised the locked `setuptools` build-tool floor to `83.0.0` to remediate
  `CVE-2026-59890` before producing public source distributions.
- Limited push-triggered CI runs to `main`; pull requests retain their own CI
  run so automation branches are not tested twice.
- Migrated active OpenSpec capabilities from legacy flat Markdown files to the
  validated `openspec/specs/<capability>/spec.md` requirement-and-scenario
  structure and added strict CI validation with OpenSpec `1.3.1`.
- Renamed the public project and distribution to **Quantitative Crypto
  Portfolio Risk Framework** while preserving the stable
  `var_cvar_crypto_risk` Python import namespace.
- Archived the completed legacy OpenSpec change records under the official
  date-prefixed `openspec/changes/archive/` convention without rewriting the
  already-established current specifications.
- Replaced the global Streamlit Simple/Log switch with Automatic handling and
  an Advanced diagnostic-only convention selector. Core portfolio, NAV,
  backtesting, Monte Carlo, and optimization workflows now always use Simple
  returns.
- Standardized analytical, scenario, optimization, reporting, plotting, and UI
  descriptions on signed loss space without changing the underlying risk
  formulas.
- Labeled monetary VaR/CVaR derived from log-return metrics as linearized
  equivalents rather than exact transformed tail P&L.
- Corrected all-breach and no-breach Christoffersen results from a synthetic
  pass to an explicit inconclusive result, propagated through conditional
  coverage, reporting, and the Streamlit presentation.
- Clarified that backtesting traffic lights describe breach frequency rather
  than complete model validity.
- Promoted synchronized package metadata from the 0.5.0 development baseline
  to the 1.0.0 first public release.
- Updated contributor setup and quality gates to use the exact locked
  environment.
- Corrected release-blocking README phase and feature contradictions.
- Replaced the phase-by-phase README log with a current product overview and
  qualified unsupported regulatory and model-performance claims.
- Removed unused imports and one redundant f-string to establish the initial
  lint baseline without changing financial behavior.
- Switched CI installation and execution to the committed uv lockfile.
- Aligned the declared Python support window with the tested 3.10–3.13 matrix.
- Raised the enforced package coverage floor from 68% to 80%.

### Removed

- Legacy root-level CLI demo scripts that duplicated Streamlit orchestration,
  depended on live data, and were superseded by the deterministic publication
  workflow.
- Stale generated HTML guide containing superseded phase/version claims and
  oversimplified statistical-test interpretations.
- Placeholder validation notebook that contained no reproducible analysis and
  was not part of the supported publication workflow.

## [0.5.0] - 2026-07-19

### Added

- Scenario-based CVaR portfolio optimization.
- Robust expected-return, volatility, and covariance assumptions.
- Optimizer input-governance diagnostics.
- Horizon-aware VaR backtesting and Monte Carlo scenarios.

[Unreleased]: https://github.com/MRezaaaaaaa/quantitative-crypto-portfolio-risk-framework/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/MRezaaaaaaa/quantitative-crypto-portfolio-risk-framework/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/MRezaaaaaaa/quantitative-crypto-portfolio-risk-framework/releases/tag/v1.0.0
