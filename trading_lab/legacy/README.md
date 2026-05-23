# Legacy Review

Legacy repo URL:

- [TradierTDA_Trader](https://github.com/mogden16/TradierTDA_Trader)

Audit summary:

- The legacy repo was an alert-driven options trading bot, not a research-first backtesting platform.
- Its strongest value is conceptual:
  - QQE and Hull Moving Average signal ideas
  - option-liquidity filter ideas
  - alert-ingestion patterns
  - broker/order lifecycle concepts
- Most execution code is too coupled to old brokers, Mongo, and live-trading loops to reuse safely.

What will be reused conceptually:

- QQE and HMA indicators
- future options chain filter requirements
- future broker abstraction design lessons from Tradier
- future alert-ingestion ideas from Gmail and Discord parsing
- queue cancellation, OCO tracking, and other execution-safety concepts

What will not be reused:

- TD Ameritrade execution code
- TD WebSocket code
- live/paper order loops
- Mongo persistence structure
- screen-scanning/OpenCV workflows
- legacy backtest code as a source of truth

Warning:

- The old code is not part of live trading in the new app.
- Do not directly port old modules into `personal-trading-lab`.
- Rebuild only the small, isolated research concepts that still make sense in the new architecture.
