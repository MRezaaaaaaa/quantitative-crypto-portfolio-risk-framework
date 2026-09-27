# Reproducible Publication Experiments

This directory defines publication experiments for GitHub, Medium, and
LinkedIn. An experiment freezes its input hash, cutoff, portfolio, return
convention, risk settings, assumptions, backtesting design, optimization
constraints, and article-to-app mapping.

In this experiment, robust expected returns are estimated from rolling seven-
day historical scenarios, while EWMA covariance is estimated from daily
returns. Both are diagnostic outputs: the historical minimum-CVaR optimizer
consumes the seven-day scenario matrix directly and does not consume either
diagnostic estimate. The manifest records this input routing explicitly.

The first experiment, `methodology-demo-v1`, uses the project's own synthetic
fixture. It demonstrates the workflow only. Its results must not be described
as evidence about actual crypto performance, forecasting accuracy, or an
investable strategy.

Publication Risk Summary is built from an explicit Portfolio Path V1 policy,
not from the legacy constant-weight summary helper. The demo config declares
`buy_and_hold`, zero commission, and zero slippage. A schema-v1 publication
config that predates these fields defaults to the same policy and costs, and the
manifest records that compatibility default. Later schemas must declare the
policy explicitly.

`risk_summary.csv` uses the same long-form summary contract as Streamlit:
`Record Type`, `Section`, `Name`, `Value`, `Display Value`, `Unit`, and
`Sample Size`. It excludes annualized return, annualized volatility, Sharpe,
and duplicate drawdown calculations. The manifest records policy, costs,
methodology version, and data cutoff so the result is not mistaken for a generic
portfolio history.

## Generate a candidate artifact bundle

Start from a clean reviewed commit:

```bash
uv run --locked --no-sync python -m scripts.reproduce_publication \
  --config publication/configs/methodology_demo_v1.yaml \
  --output-dir publication/artifacts/methodology-demo-v1
```

The command refuses a dirty Git tree by default. `--allow-dirty` exists only
for local previews and marks the manifest accordingly. It must not be used for
article or release artifacts.

## Verify an existing bundle

```bash
uv run --locked --no-sync python -m scripts.reproduce_publication \
  --verify publication/artifacts/methodology-demo-v1/manifest.json
```

Verification checks the artifact hashes, dataset and configuration hashes,
dependency lock, source-tree hash, Git commit, and cutoff boundary. It does not
prove that the model is economically correct.

## Publication boundary

CoinGecko raw API data is not committed because its standard terms restrict
redistribution. Coin Metrics community archives are CC BY-NC 4.0 and are not
used as the default because an article or portfolio project may later be
monetized or used commercially. A real-data experiment may be added only after
the exact dataset license and redistribution rights are recorded in its config.

Generated bundles are intentionally not pre-committed by this infrastructure
step. Generate them from the final reviewed article commit, inspect the
manifest, run the public-boundary check, and then decide which small artifacts
belong in the publication branch.
