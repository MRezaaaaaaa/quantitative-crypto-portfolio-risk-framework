## ADDED Requirements

### Requirement: Robust expected-return dumbbell audit

The Robust Assumptions section MUST show every available already-computed
expected-return estimate by default and MUST retain an optional Raw Historical
Mean-versus-one-estimator comparison without recomputing or mutating assumptions.

#### Scenario: Default all-estimator comparison
- **WHEN** assumptions have been built and the chart first appears
- **THEN** Raw Historical Mean, Median, Trimmed Mean, Winsorized Mean,
  Shrinkage Estimate, each available Manual View and exact downstream Final
  E[r] appear simultaneously with stable marker identities

#### Scenario: Pairwise candidate selection
- **WHEN** the user selects Median, Trimmed Mean, Winsorized Mean, Shrinkage
  Estimate, an available Manual View, or Final E[r] in Pairwise Comparison
- **THEN** only the displayed comparison endpoint changes and Raw Historical
  Mean remains fixed

#### Scenario: Missing manual view
- **WHEN** a Manual View is absent for an asset
- **THEN** its marker is omitted, its audit value remains `N/A`, and zero is not
  substituted

#### Scenario: Equal estimates
- **WHEN** two or more estimates have the same x-value
- **THEN** deterministic visual y-offsets and distinct marker symbols keep every
  estimate inspectable without modifying its x-value

#### Scenario: Ordering
- **WHEN** the user has not selected magnitude sorting
- **THEN** assets remain in portfolio order

#### Scenario: Dispersion ordering
- **WHEN** the user selects Largest estimator dispersion
- **THEN** assets are ordered by maximum minus minimum available estimate, not
  only Raw Historical Mean-to-Final distance

#### Scenario: Complete audit metadata
- **WHEN** the all-estimator figure is created
- **THEN** its metadata contains asset order, view mode, horizon, active
  estimator, every estimator value, Final E[r], per-asset dispersion, estimator
  parameters, manual-view blend and marker contract

#### Scenario: Portable audit figure
- **WHEN** the chart builder is called outside Streamlit
- **THEN** it returns a themed `plotly.graph_objects.Figure` and imports no
  Streamlit dependency
