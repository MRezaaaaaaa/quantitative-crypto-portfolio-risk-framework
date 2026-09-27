## ADDED Requirements

### Requirement: Portable Plotly Risk Lab presentation

Every user-facing Risk Lab chart MUST be a Streamlit-independent Plotly figure
built from chart-ready domain outputs.

#### Scenario: Interactive application rendering
- **WHEN** Streamlit displays a Risk Lab figure
- **THEN** it uses `st.plotly_chart`, exposes client-side PNG export and offers
  a standalone interactive HTML download

#### Scenario: Reuse outside Streamlit
- **WHEN** a caller invokes a public Risk Lab chart builder directly
- **THEN** it receives a `plotly.graph_objects.Figure` without importing
  Streamlit or recomputing a financial model

### Requirement: Shared visual and provenance contract

Risk Lab figures MUST use common accessible styling, explicit units and
practical provenance metadata.

#### Scenario: Time-series comparison
- **WHEN** comparable time series are plotted
- **THEN** the figure uses aligned dates, explicit trace labels, formatted hover
  values and unified hover behavior

#### Scenario: Correlation display
- **WHEN** Pearson or Spearman is selected
- **THEN** static and rolling method-dependent outputs use that method, the
  heatmap is annotated, and its scale remains fixed at `[-1, 1]`
