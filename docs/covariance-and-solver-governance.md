# Covariance and Solver Governance

## Purpose

Monte Carlo and portfolio optimization are numerical workflows. A model can
fail before its economic assumptions are considered: an estimated covariance
can be asymmetric or indefinite, and an optimizer can report success while its
returned values violate constraints beyond an acceptable numerical tolerance.
This project now checks both boundaries explicitly.

These checks improve numerical reliability. They do not prove that the data,
distribution, covariance estimator, or optimized portfolio is economically
correct.

## Covariance contract

Every covariance matrix entering a parametric Monte Carlo simulation must:

- be a non-empty square `pandas.DataFrame`;
- have unique, identically ordered row and column asset labels;
- contain only finite values;
- be symmetric within a scale-aware numerical tolerance; and
- be positive definite for direct simulation.

The default `repair` policy leaves a valid positive-definite matrix unchanged.
When repair is required, the implementation:

1. symmetrizes the covariance matrix;
2. converts the positive-variance block to correlation space;
3. clips correlation eigenvalues to a small positive floor;
4. renormalizes the correlation diagonal to one; and
5. reconstructs covariance using the original marginal variances.

The report records before/after eigenvalue and conditioning diagnostics,
repair reasons, and absolute and relative Frobenius adjustments. The original
marginal variances are preserved. A `strict` policy is also available and
rejects a matrix that is not already symmetric positive definite.

Repair is a numerical fallback, not an estimator. A large adjustment is a
model-risk warning and should trigger investigation of the sample, missing-data
handling, asset universe, and covariance method. Zero-variance assets remain a
special case and should normally be removed or modeled explicitly.

## Optimizer residual contract

The public optimizer status is not accepted solely from the solver. After a
candidate solution is returned, the project independently checks:

- finite weights;
- the full-investment budget `sum(w) = 1`;
- effective lower and upper weight bounds;
- the target expected-return constraint, when present;
- the CVaR cap, when present; and
- Rockafellar-Uryasev auxiliary-variable inequalities, when available.

The default absolute acceptance tolerance is `1e-5`. The raw solver result is
preserved as `solver_status`. The independent report is returned as
`constraint_validation`, and its largest violation is exposed as
`max_constraint_violation`.

If the solver reports `optimal` or `optimal_inaccurate` but the independent
checks fail, the public result status becomes `validation_failed`. Such a
portfolio must not be presented as solved or compared as a valid optimized
portfolio.

For a CVaR-constrained program, the gate checks the solver's
Rockafellar-Uryasev CVaR expression because that is the quantity constrained by
the optimization model. The separately reported empirical scenario CVaR can
differ slightly because of finite-sample quantile conventions and solver
tolerance; the two should be interpreted together rather than assumed to be
bit-for-bit identical.

## UI and reproducibility

The Robust Assumptions tab shows whether covariance was used unchanged or
repaired, together with minimum eigenvalues and conditioning. The Optimizer tab
shows raw solver status, independent validation status, maximum violation, and
the per-constraint residuals for every result.

Publication manifests should record the covariance policy, whether repair
occurred, the relative adjustment, solver name and raw status, validation
tolerance, and maximum constraint violation. These diagnostics are part of the
result lineage, not optional presentation metadata.
