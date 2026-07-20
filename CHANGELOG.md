# Changelog

All notable changes to this project will be documented in this file. The format
is based on Keep a Changelog, and the project intends to follow Semantic
Versioning when it reaches a public release.

## [Unreleased]

### Added

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

### Changed

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
- Synchronized package development-version metadata at 0.5.0.
- Updated contributor setup and quality gates to use the exact locked
  environment.
- Corrected release-blocking README phase and feature contradictions.
- Replaced the phase-by-phase README log with a current product overview and
  qualified unsupported regulatory and model-performance claims.
- Removed unused imports and one redundant f-string to establish the initial
  lint baseline without changing financial behavior.
- Switched CI installation and execution to the committed uv lockfile.
- Aligned the declared Python support window with the tested 3.10–3.13 matrix.

## [0.5.0] - 2026-07-19

### Added

- Scenario-based CVaR portfolio optimization.
- Robust expected-return, volatility, and covariance assumptions.
- Optimizer input-governance diagnostics.
- Horizon-aware VaR backtesting and Monte Carlo scenarios.
