# Options Module Placeholder

Options support is still intentionally deferred.

The next phase should begin with options data foundations, not full options backtesting.

## Near-Term Direction

Start with:

- option contract metadata
- option chain snapshots
- quote history persistence
- vendor abstraction for future providers

Do not start with:

- full options strategy logic
- option PnL modeling
- broker execution
- live options trading

## Likely Future Tables

### `option_contracts`

- `contract_symbol`
- `underlying_symbol`
- `expiration_date`
- `strike`
- `option_type`
- `multiplier`
- `exercise_style`
- `first_seen_at`
- `last_seen_at`

### `option_chain_snapshots`

- `snapshot_id`
- `source_vendor`
- `underlying_symbol`
- `snapshot_timestamp`
- `underlying_price`
- `expiration_date`
- `retrieved_at`

### `option_quotes`

- `snapshot_id`
- `contract_symbol`
- `bid`
- `ask`
- `mid`
- `last`
- `volume`
- `open_interest`
- `implied_volatility`
- `delta`
- `gamma`
- `theta`
- `vega`
- `rho`
- `quote_timestamp`

### `option_greeks`

- `snapshot_id`
- `contract_symbol`
- `delta`
- `gamma`
- `theta`
- `vega`
- `rho`
- `implied_volatility`
- `model_timestamp`

### `option_backtest_runs`

- `option_backtest_run_id`
- `strategy_name`
- `underlying_universe`
- `benchmark_symbol`
- `start_date`
- `end_date`
- `pricing_assumptions_json`
- `created_at`

### `option_backtest_trades`

- `option_backtest_run_id`
- `contract_symbol`
- `entry_timestamp`
- `exit_timestamp`
- `entry_price`
- `exit_price`
- `quantity`
- `pnl`
- `exit_reason`

## Why This Is Deferred

Before adding options overlays, the stock signal should already show:

- sufficient trade count
- acceptable drawdown
- positive benchmark-relative performance
- tolerable out-of-sample degradation
- reasonable walk-forward consistency
- no obvious dependence on one trade, one ticker, or one year

The current app focuses on getting that stock research foundation credible first.
