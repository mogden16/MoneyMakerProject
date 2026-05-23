# Data Schema Notes

## Implemented in Phase 1

Current DuckDB tables:

- `stock_bars`
- `backtest_runs`
- `backtest_trades`
- `backtest_equity_curve`

## Planned Future Schema

Target schema expansion:

- `instrument_master`
- `stock_bars`
- `corporate_actions`
- `option_contracts`
- `option_chain_snapshots`
- `option_quotes`
- `backtest_runs`
- `backtest_trades`
- `backtest_equity_curve`

## Design Intent

- Keep provider-specific ingestion separate from normalized research tables.
- Preserve enough metadata to replay and audit a backtest run later.
- Leave room for options-specific chain snapshots, quotes, IV, Greeks, and contract metadata without forcing a redesign of the existing stock backtest tables.
