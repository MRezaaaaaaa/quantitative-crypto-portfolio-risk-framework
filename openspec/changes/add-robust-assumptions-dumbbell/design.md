# Design: Robust expected-return dumbbell audit

## Data boundary

The builder receives the exact output of `build_assumption_table` together
with its `AssumptionConfig`. The table is copied and read only. Candidate
estimators and Final E[r] are never reconstructed inside the chart module.

## Visual contract

- All Estimators is the default; Pairwise Comparison retains the prior behavior.
- Raw Historical Mean is a neutral outlined circle; Median an orange diamond;
  Trimmed Mean a blue square; Winsorized Mean a purple triangle; Shrinkage a
  green cross; Manual View a yellow hexagon; and Final E[r] a large red star.
- In the all-estimator view, each asset has a subtle connector from its minimum
  to maximum available estimate.
- A visible zero line and symmetric x-range distinguish negative and positive
  assumptions.
- Fixed small y-offsets within each asset band prevent equal estimates from
  hiding one another. Their x-values remain exact.
- Missing Manual Views remain null and are omitted as markers; metadata and
  hover context report `N/A`, never zero.
- Values are converted from decimal returns to percentage points only for
  presentation.

## Audit metadata

Figure metadata records asset order, view mode, horizon, every completed value,
full dispersion, signed basis-point differences, Final E[r], estimator
parameters, manual-view blend and the marker contract. Hover data includes
estimator identity, value, mean-relative differences, active recipe, downstream
status, horizon, relevant configuration, overlap peers and Manual View status.

## Ordering

Portfolio order is the default. Largest-estimator-dispersion order uses the
completed table only:

```text
dispersion(asset) = max(available estimates) - min(available estimates)
```

It is not the absolute Raw Mean-to-Final distance.

## Horizon integrity

The default one-day workflow uses the requested one-day title. When the Robust
Assumptions Engine is deliberately built for another horizon, the figure uses
that actual horizon instead of falsely labelling an h-day assumption as daily.
