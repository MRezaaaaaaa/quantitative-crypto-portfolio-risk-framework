# Tasks

- [x] Add typed Portfolio Path V1 policies, result and provenance.
- [x] Implement launch, mark-to-market, UTC calendars and deferred events.
- [x] Implement deterministic post-cost allocation and accounting checks.
- [x] Route Risk Lab realized analytics and current exposure through the path.
- [x] Add required Plotly path, drift, cost and drawdown figures.
- [x] Add policy/cost/risk-base provenance to relevant exports.
- [x] Preserve manual fixed-holdings Portfolio Monitor behavior.
- [x] Add deterministic finance, missing-data, chart and UI tests.
- [x] Update methodology, model-risk, return, user and release documentation.
- [x] Run all locked-environment gates and record the results.

## Verification record

- Full locked test suite: `566 passed`; combined coverage `83.27%` with the
  repository's `80%` minimum enforced.
- Focused portfolio-path, risk-base and UI suite: `65 passed`.
- Ruff lint and changed-file formatting checks: passed.
- OpenSpec validation: `10 passed`, `0 failed`.
- Public-boundary and Git-history-boundary checks: passed.
- Source distribution and wheel build: passed; CI-equivalent wheel import
  smoke test passed.
- Streamlit local health endpoint: `ok`.
- Relative Markdown link validation and `git diff --check`: passed.
