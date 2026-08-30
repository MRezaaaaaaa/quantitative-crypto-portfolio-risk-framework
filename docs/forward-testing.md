# Forward Testing

## Three experiment modes

The labels are deliberately strict because the information available at launch
differs by mode.

| Mode | What happens | What it is not |
|---|---|---|
| Historical Out-of-Sample Replay | Rebuild at a past cutoff, freeze the snapshot, then reveal already-available later observations sequentially | A live forward test |
| Live Forward Test | Freeze the snapshot at creation and append only complete observations that become available afterward | A historical backtest with a recent date label |
| Hybrid Historical OOS + Live Forward | Replay a declared historical evaluation interval, then append future complete observations after the boundary | One homogeneous sample |

Historical replay can provide out-of-sample evidence relative to its frozen
cutoff, but the evaluation data are already known at the time the user runs the
software. Research choices made after seeing those data can still create
selection bias. Only Live Forward mode creates genuinely prospective evidence
after the frozen launch.

## Point-in-time boundary

Historical and Hybrid experiments enforce:

```text
training_start <= training_end <= optimization_as_of
optimization_as_of < launch_date <= historical_evaluation_end
```

Expected-return, covariance, scenario, and optimizer inputs must end at or before
`optimization_as_of`. Historical and Hybrid creation rebuilds from the bounded
training data and serialized recipe. Tests perturb future observations to prove
they cannot change the frozen snapshot or earlier forecasts.

Launch uses the explicitly requested complete close after the information
cutoff. Launch NAV equals initial capital and launch return is zero; performance
begins on the following complete observation. This is a research close-price
convention, not an executable fill assumption.

## Sequential evaluation

Evaluation observations are processed in date order through the same valuation
and forecast services used for live updates. A risk forecast stores its origin
and target, and its estimator receives data only through the origin. A future
outcome remains pending until the target date matures.

If a horizon forecast uses overlapping target windows, adjacent realized losses
share observations. They must not be treated as independent evidence. The
monitor records exceptions but does not convert a short forward sample into a
full Kupiec/Christoffersen validation claim.

## Live and Hybrid updates

Live and Hybrid modes require a refreshable provider mapping. An uploaded CSV
may create a Historical OOS experiment, but it cannot be represented as an
ongoing live feed. Each live update is a bounded, idempotent operation that:

1. requests data through an explicit or safe default cutoff;
2. excludes the partial current UTC day by default;
3. records the actual provider and actual cutoff;
4. appends new explicit observations without overwriting finalized rows;
5. values fixed quantities and creates new origin-safe forecasts;
6. evaluates each matured forecast at most once; and
7. commits atomically or rolls back the related financial writes.

Streamlit's **Update Now** executes one such refresh. Continuous operation must
use the one-shot CLI from an external scheduler; Streamlit does not run a hidden
loop.

In Hybrid mode, the historical and live portions remain visibly separated. The
historical results do not become “live” merely because later observations are
appended.

## Bias and interpretation controls

Forward testing does not, by itself, solve:

- **selection bias:** choosing assets, dates, constraints, or methods after
  inspecting the evaluation period;
- **survivorship bias:** using today's available crypto universe for a past
  experiment;
- **multiple testing:** promoting the best of many experiments without recording
  the full search;
- **vendor revision risk:** later downloads may not equal the originally seen
  observations;
- **regime risk:** a short forward window may represent only one market state;
- **estimation error:** expected returns, covariance, tails, and correlations
  remain uncertain;
- **implementation shortfall:** displayed close-price returns omit fees,
  slippage, liquidity, taxes, custody, and market impact.

Experiment names and UUIDs, recipes, hashes, dates, versions, provider metadata,
and update events make the evidence auditable; they do not make it unbiased or
investable. Do not claim prediction accuracy, outperformance, suitability, or a
performance guarantee from a replay or live monitor.

See [Portfolio monitoring](portfolio-monitoring.md),
[Data provenance](data-provenance.md), and [Reproducibility](reproducibility.md).
