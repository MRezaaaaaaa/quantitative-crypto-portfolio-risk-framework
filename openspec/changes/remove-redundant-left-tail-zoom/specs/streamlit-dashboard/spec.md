## MODIFIED Requirements

### Requirement: Single authoritative historical distribution

The Distribution section SHALL use the primary interactive Plotly historical
distribution as the only tail-distribution view.

#### Scenario: Inspecting historical tail risk

- **WHEN** historical portfolio returns and VaR/CVaR levels are available
- **THEN** the primary distribution SHALL show the full observed distribution
- **AND** it SHALL show the observed left-tail trace
- **AND** it SHALL show the VaR and CVaR reference lines
- **AND** it SHALL NOT render a standalone Left-tail Zoom expander.

### Requirement: Exact histogram overlay alignment

The full-distribution and observed-tail histogram traces SHALL use explicitly
shared Plotly bin alignment.

#### Scenario: Tail observations are highlighted

- **WHEN** the tail trace overlays the full historical histogram
- **THEN** both traces SHALL use the same bin group and requested bin count
- **AND** the red tail bars SHALL correspond to the full histogram bins.

### Requirement: Page scrolling remains usable

The application SHALL NOT enable Plotly mouse-wheel zoom by default for the
historical distribution.

#### Scenario: User scrolls the Streamlit page

- **WHEN** the pointer is over the distribution chart
- **THEN** ordinary page scrolling SHALL not be captured by an explicitly
  enabled Plotly wheel-zoom setting
- **AND** box zoom, pan, autoscale, and reset controls SHALL remain available.
