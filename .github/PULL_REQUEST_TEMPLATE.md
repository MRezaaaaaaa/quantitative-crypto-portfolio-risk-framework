## Scope

Describe the problem, root cause, and the smallest change that addresses it.

Change type:

- [ ] Documentation or repository governance only
- [ ] Refactor with no intended numerical change
- [ ] Data or dependency behavior
- [ ] Financial methodology or numerical behavior
- [ ] Streamlit presentation or state behavior

## Financial and model-risk impact

State whether this changes return conventions, horizons, signs/units,
VaR/CVaR, backtesting, scenarios, covariance, optimization, or solver behavior.
If none change, write `No intended financial-behavior change`.

Address as applicable:

- look-ahead bias or data leakage;
- overlapping observations or dependence assumptions;
- estimation error and overfitting;
- in-sample versus out-of-sample interpretation;
- public/private data boundary.

## Validation

List the exact checks run and their outcomes.

- [ ] Focused tests added or updated
- [ ] Full regression suite passed
- [ ] Numerical golden baseline reviewed
- [ ] Lint and `git diff --check` passed
- [ ] Public-boundary check passed
- [ ] Documentation and local links reviewed

## Numerical changes

Describe expected output changes and why they are correct. If the golden
baseline changes, include a reviewed before/after summary. Never update the
baseline solely to make CI pass.

## Publication checklist

- [ ] No secret, local path, private dataset, real holding, transaction record,
      monitoring database, or proprietary parameter is included
- [ ] Generated outputs and vendor caches are excluded
- [ ] New GitHub Actions are pinned to full commit SHAs
- [ ] User-facing claims remain qualified and reproducible
- [ ] Changelog and methodology documentation are updated when required
