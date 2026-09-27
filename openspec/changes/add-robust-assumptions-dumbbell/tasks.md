# Tasks

- [x] Add a pure Plotly dumbbell builder over the completed assumption table.
- [x] Add estimator and explicit asset-order selectors to Robust Assumptions.
- [x] Add default All Estimators and optional Pairwise Comparison view modes.
- [x] Add stable marker identities, min/max range connectors and deterministic
      overlap offsets without changing any estimator x-value.
- [x] Sort optionally by the full available estimator dispersion.
- [x] Store complete values, Final E[r], dispersion, parameters and marker
      contract in Plotly metadata.
- [x] Preserve null Manual Views and visible equal endpoints.
- [x] Add complete hover recipe and basis-point audit metadata.
- [x] Add focused numerical, immutability, presentation and independence tests.
- [x] Document the audit-only interpretation and horizon behaviour.
- [x] Run and record complete release-quality verification gates.

## Verification record

- Full regression suite: 625 passed; total coverage 87.58% (80% floor).
- Ruff lint: passed. All 36 changed/new Python files pass
  `ruff format --check`.
- Git whitespace validation: `git diff --check` passed.
- OpenSpec strict validation: 13 passed, 0 failed.
- Relative Markdown links: 58 checked, 0 broken.
- Public working-tree and Git-history boundary checks: passed.
- Manual Streamlit smoke: default Final E[r], equal-endpoint labels, and the
  interactive Median comparison were inspected successfully.
