# Phase 8 implementation tasks

## Batch 0 — Preflight and read-only audit

- [x] Verify branch history, clean working tree, release baseline, lock, and
      current CI/security gates.
- [x] Confirm 412-test baseline and 80% coverage floor on the current main line.
- [x] Audit optimizer outputs and independent residual validation.
- [x] Audit Simple/Log wealth boundary and signed loss-space convention.
- [x] Audit current data loading, one-day forward-fill behavior, source fallback,
      and UTC completeness risks.
- [x] Audit Streamlit session state and identify orchestration that must not be
      copied into the monitoring UI.
- [x] Audit public/private scanners, Git-history scanner, dependency policy, and
      OpenSpec 1.3.1 conventions.
- [x] Record current architecture, data-leakage, persistence, model-risk, and
      implementation gaps in `design.md`.

## Batch 1 — OpenSpec and architecture

- [x] Add Phase 8 proposal and explicit out-of-scope boundary.
- [x] Define experiment identity, modes, lifecycle, status transitions, and
      archive semantics.
- [x] Define point-in-time dates, launch convention, fixed quantities, cash, and
      target/current weight contracts.
- [x] Define immutable optimization snapshot and validity gate.
- [x] Define SQLAlchemy/Alembic/SQLite architecture, tables, keys, indexes,
      transactions, and future PostgreSQL seam.
- [x] Define Historical OOS, Live Forward, and Hybrid workflows.
- [x] Define idempotency, missing-data, partial-day, and correction boundaries.
- [x] Define risk forecast/evaluation alignment and VaR/CVaR semantics.
- [x] Define Plotly chart inputs and visual semantics.
- [x] Define exports, public/private boundary, dependency direction, and test
      plan.
- [x] Pass strict OpenSpec validation.
- [x] Pass Markdown-link validation and `git diff --check`.
- [x] Commit and push design-only files to `phase-8-spec`.
- [x] Open Draft PR `design: specify Phase 8 portfolio experiment monitor` and
      stop for review.

## Batch 2 — Persistence foundation

- [ ] Add reviewed bounded SQLAlchemy and Alembic dependencies and regenerate
      `uv.lock` without changing existing financial versions unnecessarily.
- [ ] Add database URL configuration, sanitized logging, SQLite foreign-key and
      transaction setup.
- [ ] Add Alembic configuration and deterministic initial migration.
- [ ] Add domain value objects, enums, lifecycle validator, canonical JSON, and
      SHA-256 recipe/source hashing.
- [ ] Add ORM models for experiments, snapshots, allocations, observations,
      daily states, forecasts, runs, and events.
- [ ] Add repository protocols, SQLAlchemy adapters, and unit of work.
- [ ] Add temporary-database tests for migration, CRUD, uniqueness, foreign
      keys, restart persistence, archive, rollback, and immutable snapshots.
- [ ] Update `.gitignore`, `.env.example`, and public scanner for monitoring
      databases, WAL, SHM, journals, and secret-bearing URLs.

## Batch 3 — Valuation and experiment registry

- [ ] Add experiment create/query/transition/archive services and event audit.
- [ ] Add serializable optimizer and risk recipe objects.
- [ ] Add point-in-time optimization adapter that calls existing assumptions,
      scenarios, optimizer, and residual validation functions.
- [ ] Add validated immutable snapshot and target-allocation persistence.
- [ ] Add strict monitoring price normalization without silent forward fill.
- [ ] Add next-complete-observation launch validation and fixed quantities.
- [ ] Add deterministic zero/rate cash accrual from launch, not previous writes.
- [ ] Add daily NAV, benchmark, current weights, drift, total drift, running peak,
      drawdown, and maximum drawdown.
- [ ] Add incomplete-date/data-quality behavior and atomic daily writes.
- [ ] Add CSV/JSON export bundles and secret-free manifests.
- [ ] Add valuation, hashing, registry, snapshot, data-quality, and export tests.

## Batch 4 — Historical Out-of-Sample Replay

- [ ] Add training/cutoff/launch/evaluation boundary validation.
- [ ] Rebuild optimization from the training slice; reject session-state result
      reuse for Historical and Hybrid modes.
- [ ] Record maximum input dates for every optimizer input and assert they do not
      exceed `optimization_as_of`.
- [ ] Reveal evaluation observations sequentially through the shared valuation
      workflow.
- [ ] Add origin-safe risk forecast creation and matured-outcome evaluation.
- [ ] Complete Historical experiments and transition Hybrid experiments to
      Active at the exact boundary.
- [ ] Add tests proving post-cutoff changes cannot alter the snapshot, launch
      return is zero, replay is deterministic, and no future frame leaks.

## Batch 5 — Live and Hybrid update

- [ ] Add refreshable price-provider protocol and persistable source mappings.
- [ ] Reject non-refreshable uploaded files for Live/Hybrid creation.
- [ ] Add one-experiment and all-active one-shot update services.
- [ ] Exclude partial current UTC day by default and record actual cutoff/source.
- [ ] Append only new complete observations and preserve finalized rows.
- [ ] Evaluate each matured VaR forecast once; leave future outcomes pending.
- [ ] Add monitoring-run counts, sanitized failures, transaction rollback, and
      retry events.
- [ ] Add idempotent CLI commands suitable for an external scheduler.
- [ ] Add offline fake-provider tests for append, rerun, missing data, future
      dates, rollback, and forecast evaluation.

## Batch 6 — Streamlit and Plotly monitoring

- [ ] Add bounded Plotly dependency and retain existing Matplotlib support.
- [ ] Add `streamlit_ui` package and minimal navigation integration in `app.py`.
- [ ] Add Experiments list and archive-only lifecycle actions.
- [ ] Add Create Forward Test with methodology preview and strict cutoff checks.
- [ ] Add Portfolio Monitor snapshot, performance, risk, provenance, and
      historical/live boundary views.
- [ ] Add NAV/benchmark, 100% stacked allocation, target/current, drift,
      drawdown, VaR/CVaR, breach, and forecast/realized charts.
- [ ] Add Experiment Comparison with explicit calendar or launch-age alignment.
- [ ] Add Data Quality and manual `Update Now` views.
- [ ] Add CSV/JSON downloads and empty/error-state UI tests.
- [ ] Test allocation sums, stable colors, boundary markers, CVaR non-breach
      semantics, and unsupported-fan suppression.

## Batch 7 — Documentation and final verification

- [ ] Update README and `CHANGELOG.md` Unreleased without bumping package version.
- [ ] Update architecture, model-risk, reproducibility, and public-release docs.
- [ ] Add portfolio-monitoring, forward-testing, database, and user-guide docs.
- [ ] Document Historical OOS versus Live Forward, Hybrid, fixed holdings,
      missing costs, source quality, no rebalancing, and no performance guarantee.
- [ ] Document local database, external scheduler/CLI, archive, exports, and
      private-data boundary.
- [ ] Run full tests with coverage at least 80% on Python 3.10–3.13 in CI.
- [ ] Run Ruff, build, clean wheel import, Streamlit smoke, strict OpenSpec,
      public/history scanners, Markdown links, and `git diff --check`.
- [ ] Confirm no existing numerical value changed without separate methodology
      review.
- [ ] Confirm no database, secret, real holding, transaction, tag, deployment,
      release, or package publication occurred.
- [ ] Prepare a reviewable PR; a future release gate, not Phase 8 implementation,
      decides `v1.1.0`.
