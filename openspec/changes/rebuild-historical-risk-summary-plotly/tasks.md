# Tasks

- [x] Add a policy-specific retrospective summary model and data-quality audit.
- [x] Rebuild Risk Summary cards, tables, captions and long-form CSV from that
      single model, preserving raw values separately from display values.
- [x] Remove annualized return/volatility, Sharpe and duplicate drawdown rows.
- [x] Add a shared Plotly theme and pure chart-builder boundary.
- [x] Migrate Risk Lab distributions, paths, dependence, backtest, simulation
      and optimization figures to Plotly.
- [x] Keep the selected Pearson/Spearman method consistent across correlation
      outputs.
- [x] Preserve client-side PNG and add standalone interactive HTML downloads.
- [x] Replace Matplotlib with Plotly in the core dependency declaration and
      refresh the lockfile.
- [x] Add focused numerical, presentation and no-Streamlit chart tests.
- [x] Update user, methodology, architecture, model-risk and release docs.
- [x] Run and record the complete release-quality verification gates.
- [x] Exclude current/future UTC-date source rows from finalized historical
      analytics while retaining them for audit.
- [x] Route publication Risk Summary through an explicit portfolio path and the
      same authoritative historical-summary contract.

## Verification record

- Lock consistency checked with uv 0.12.19; `uv lock --check` passed.
- Ruff lint and changed-scope format checks passed; `git diff --check` passed.
- Focused correctness suite: 75 passed. Full regression suite: 613 passed.
- OpenSpec strict validation: 13 passed, 0 failed.
- Public working-tree and Git-history boundary checks passed.
- Relative Markdown links: 75 files scanned, 58 local links checked, 0 broken.
- Dirty-preview publication bundle generated and all 10 artifacts verified.
- Streamlit startup and local health-endpoint smoke checks passed.
