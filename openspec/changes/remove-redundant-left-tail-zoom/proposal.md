# Remove Redundant Left-tail Zoom

## Why

The primary interactive historical distribution already exposes the observed
tail, VaR/CVaR reference levels, hover inspection, box zoom, pan, autoscale,
axis reset, and HTML export. A second tail-only expander duplicates that view
without adding a separate analytical result.

## What Changes

- Remove the standalone Left-tail Zoom expander and its exports.
- Remove the unused chart builder and public export.
- Keep the primary distribution tail overlay and VaR/CVaR references.
- Force the full and tail histograms to share Plotly bins.
- Keep mouse-wheel zoom disabled by default.

## Out of Scope

- VaR or CVaR calculation changes.
- Changes to return construction, confidence levels, or horizons.
- Changes to any downstream risk or optimization input.
