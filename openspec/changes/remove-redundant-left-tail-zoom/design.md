# Design

The Distribution section will have one authoritative Plotly histogram. Its
full-sample and observed-tail traces use the same `bingroup` and requested bin
count so the tail highlight maps to the full distribution's bin boundaries.

The standalone tail chart, application import, expander, download identity, and
public export are removed together. Existing Plotly modebar interactions remain
available, while mouse-wheel zoom is not enabled so normal Streamlit scrolling
is preserved.

This is a presentation-only cleanup. The chart consumes the same finalized
historical returns and supplied VaR/CVaR values as before.
