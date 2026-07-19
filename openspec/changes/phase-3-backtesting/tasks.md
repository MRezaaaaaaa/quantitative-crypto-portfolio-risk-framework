# Phase 3 Tasks

- [x] Add backtesting section to config.yaml
- [x] Implement rolling_var_forecast (no look-ahead bias)
- [x] Implement calculate_var_breaches
- [x] Implement kupiec_pof_test
- [x] Implement christoffersen_independence_test
- [x] Implement christoffersen_cc_test
- [x] Implement assign_traffic_light_status (Basel III + rate-based)
- [x] Implement interpret_traffic_light_status
- [x] Implement backtest_var_model (full pipeline)
- [x] Implement compare_var_models_backtest
- [x] Implement create_backtesting_report_table
- [x] Add plot_var_backtest to plotting.py
- [x] Add plot_breach_timeline to plotting.py
- [x] Add plot_model_comparison_backtest to plotting.py
- [x] Add Backtesting tab to app.py
- [x] Write test_backtesting.py (50 tests)
- [x] Update openspec/specs/backtesting.md
- [x] Write openspec/changes/phase-3-backtesting/proposal.md
- [x] Write openspec/changes/phase-3-backtesting/design.md
- [x] Update README.md with Phase 3 section

## Phase 3 Cleanup & Stabilization

- [x] Make `var_cvar_crypto_risk/__init__.py` lightweight (no eager data-source imports)
- [x] Lazy-import yfinance inside `yfinance_client.fetch_yfinance_prices`
- [x] Lazy-import the CoinGecko client + yfinance fallback inside `data_loader.load_price_data`
- [x] Add reproducible Phase 3 CLI demo (`run_phase3_backtest_demo.py`)
- [x] Generate CSV/PNG/JSON Phase 3 outputs under `outputs/`
- [x] Add `.gitkeep` markers for `data/raw`, `data/processed`, `data/cache`, `outputs/{charts,tables,reports}`
- [x] Update `.gitignore` (`__MACOSX/`, `outputs/`, `data/cache/`, build artifacts)
- [x] Add stabilization tests (`tests/test_stabilization.py`)
- [x] Verify full pytest suite passes without network or yfinance access
