# Legacy Repo Audit: `TradierTDA_Trader`

Legacy repo URL:

- `https://github.com/mogden16/TradierTDA_Trader`

## A. Executive Summary

The legacy repo was built as an alert-driven options trading bot. Its core job was to ingest trade alerts from Gmail or Discord, optionally run intraday technical-analysis filters, choose an option contract, send orders through TD Ameritrade or Tradier, track the order lifecycle in MongoDB, and manage exits through OCO, stop, or trailing-stop logic.

That is materially different from `personal-trading-lab`, which is a research-first daily stock backtesting platform. The new app is designed to make strategy evaluation repeatable and auditable before any paper or live execution is added.

What is worth keeping conceptually:

- QQE and Hull Moving Average filter ideas
- alert-ingestion design patterns
- broker abstraction lessons
- option-liquidity filter ideas
- queue, order-state, and safety concepts
- corporate-action and benchmark caution mindset

What should not be reused directly:

- live-order code
- TD Ameritrade integration
- Mongo-centric execution state model
- Discord/Gmail execution loops
- ad hoc global state lists
- hard-coded config/token patterns
- backtest logic that assumes option fills from minute bars without spread modeling

## B. Module Inventory

| Legacy Path | Purpose | Dependencies | Risk Level | Reuse Classification | Notes |
| --- | --- | --- | --- | --- | --- |
| `main.py` | Main live bot loop orchestrating scanners, traders, tasks, websocket, and backtest scheduling | TD API, Tradier, Gmail, Discord, Mongo, OpenCV | High | Discard | Operational script, not aligned with research-first architecture |
| `config.py.example` | Monolithic runtime configuration | Secrets, broker accounts, Discord auth, Mongo URI | High | Future reference only | Useful for identifying knobs; unsafe patterns must not be copied |
| `constants.py` | Global mutable in-memory lists | None | High | Discard | Global state should not be carried forward |
| `assets/techanalysis.py` | Intraday HMA + QQE + ADR filters | TD minute bars, `pyti`, `pandas_ta` | Medium | Reuse conceptually | Main source for QQE/HMA research logic |
| `tech_config.py` | Technical-analysis parameter file | None | Low | Reuse conceptually | Good source of default indicator parameters |
| `assets/helper_functions.py` | Time helpers, Gmail formatting, option expiration helper, strategy bootstrap | TD API, market calendar | Medium | Refactor heavily | Some small utilities useful as reference only |
| `assets/tasks.py` | OCO child monitoring, stale queue cancellation, task loop | Mongo, TD/Tradier APIs | High | Future reference only | Good design input for later paper/live safety layer |
| `assets/streamprice.py` | Stream-based exit logic support | Mongo, websocket prices | High | Future reference only | Execution-only |
| `api_trader/__init__.py` | TD order lifecycle and Mongo state management | TD API, Mongo | High | Discard | Tightly coupled to deprecated broker path |
| `api_trader/order_builder.py` | TD order/OCO/trailing-stop payload construction | TD API quotes | High | Future reference only | Useful for future broker abstraction design, not code reuse |
| `tradier/__init__.py` | Tradier order lifecycle and Mongo state management | Tradier API, Mongo | High | Future reference only | Useful later when broker abstraction is built |
| `tradier/tradierOrderBuilder.py` | Tradier order payload construction | Tradier API, Polygon symbol builder | High | Future reference only | Useful to understand OTOCO semantics later |
| `tradier/tradier_helpers.py` | Tradier OCO child parsing | None | Medium | Reuse conceptually | Small isolated logic worth remembering, not copying blindly |
| `tradier/tradier_constants.py` | Tradier endpoints/constants | Requests | Low | Future reference only | Can help later when Tradier integration is revisited |
| `tdameritrade/__init__.py` | TD token refresh and REST client wrapper | TD API, Mongo | High | Discard | TD should remain legacy reference only |
| `tdameritrade/td_helpers.py` | TD client bootstrap | Selenium, TDA SDK | High | Discard | Broker-specific and operationally brittle |
| `td_websocket/stream.py` | Option quote streaming into Mongo | TDA stream SDK, Mongo | High | Future reference only | Useful only for future execution monitoring ideas |
| `discord/discord_scanner.py` | Parses Discord alert feed into trade objects | Discord HTTP API | Medium | Reuse conceptually | Useful alert-ingestion reference, but format-specific |
| `discord/discord_helpers.py` | Sends Discord webhook notifications | Discord webhook | Medium | Future reference only | Notification helper only |
| `gmail/__init__.py` | Gmail auth, inbox polling, alert parsing, auto-trash workflow | Gmail API | Medium | Reuse conceptually | Parsing ideas useful; execution loop should not be reused |
| `mongo/__init__.py` | Mongo connection and collection handles | MongoDB Atlas | High | Discard | New app is local DuckDB-based |
| `mongo/mongo_helpers.py` | Convenience wrappers for Mongo collections | MongoDB Atlas | High | Discard | Persistence concepts only |
| `backtest/backtest.py` | Option-history backtest driver using Polygon and saved alerts/closed trades | Polygon, Excel files, Discord | High | Future reference only | Backtest design is not trustworthy enough to transplant |
| `backtest/optimizestrats.py` | OCO and trailing-stop option payoff evaluation | Pandas, Excel minute bars | Medium | Reuse conceptually | Useful for future option-exit modeling ideas |
| `open_cv/*` | Screen-scanning workflow for chart signals and option lookup | OpenCV, TD chain lookup | High | Discard | Orthogonal to current product direction |
| `assets/pushsafer.py`, `assets/multifilehandler.py`, `assets/exception_handler.py`, `assets/timeformatter.py` | Utility infrastructure | Logging, Pushsafer | Medium | Future reference only | Mostly operational |

## C. Strategy and Indicator Audit

### QQE Logic

What it appears to do:

- Uses `pandas_ta.qqe` on intraday close data
- Watches a QQE line crossing its smoothed trigger line
- Uses 30-minute QQE level checks as trend confirmation
- Prevents call entries in very overbought zones and put entries in very oversold zones

Required inputs:

- intraday OHLCV
- close series
- QQE parameters from `tech_config.py`

Timeframe assumptions:

- built around 5-minute, 10-minute, and 30-minute bars

Daily adaptation:

- QQE itself can be adapted to daily data
- the exact legacy logic cannot be preserved on daily bars because its confirmation stack is intraday-specific

Recommendation:

- rebuild QQE as a research-only indicator module
- do not port the current intraday trigger logic directly

### Hull Moving Average Logic

What it appears to do:

- calculates fast and slow HMA on intraday close data
- uses fast/slow ordering as directional confirmation
- uses HMA crossover as a sell filter for open positions

Required inputs:

- close series
- HMA fast/slow lengths

Timeframe assumptions:

- primarily 10-minute and 5-minute derived logic

Daily adaptation:

- yes, HMA is portable to daily data
- a daily HMA crossover strategy is feasible in the new app

Recommendation:

- rebuild HMA as a reusable indicator
- use a clean vectorized implementation with next-bar execution enforced by the existing engine

### 10-Minute Filter

What it appears to do:

- checks price against ADR-based support/resistance bands
- confirms HMA direction on 10-minute bars
- uses 10-minute price location to avoid chasing highs/lows

Daily adaptation:

- no direct adaptation
- ADR-style regime or distance filters can be rebuilt for daily data, but not as a one-to-one port

Recommendation:

- future reference only

### 30-Minute Filter

What it appears to do:

- uses 30-minute QQE level thresholds to confirm broader intraday direction
- acts as a higher-timeframe filter on top of 5-minute QQE cross entries

Daily adaptation:

- conceptually yes
- implementation would need a different framing, likely daily plus weekly confirmation or daily plus benchmark regime filter

Recommendation:

- reuse conceptually, not literally

### Call/Put Signal Logic

What it appears to do:

- call: 5-minute QQE cross up, 30-minute QQE above lower-neutral threshold, 10-minute HMA aligned up, not too overbought
- put: mirror logic in the opposite direction

Daily adaptation:

- only partially
- the intraday option-entry timing cannot be trusted on daily stock bars

Recommendation:

- use as inspiration for a future `qqe_hma_strategy.py`
- only after indicators are rebuilt cleanly

### Overbought/Oversold Logic

What it appears to do:

- uses thresholds around QQE/RSI-style levels
- avoids calls in super-overbought conditions and puts in super-oversold conditions

Daily adaptation:

- yes

Recommendation:

- keep conceptually

### Trend Direction Logic

What it appears to do:

- combines QQE direction, HMA fast/slow ordering, and ADR location

Daily adaptation:

- partially

Recommendation:

- rebuild into a simpler daily research strategy first

## D. Risk and Execution Audit

### Position Sizing

Old behavior:

- strategy position size came from Mongo strategy objects
- order quantity for options was `int((position_size / 100) / option_price)`

Recommendation:

- stock research app: keep existing clean sizing model
- future options backtester: reuse conceptual sizing rule only

### Minimum/Maximum Option Price

Old behavior:

- rejects alerts outside configured option price band

Recommendation:

- future options chain filtering design input

### Minimum Option Volume

Old behavior:

- rejects illiquid contracts below `MIN_VOLUME`

Recommendation:

- future options chain filtering design input

### Minimum Delta

Old behavior:

- rejects contracts with low absolute delta

Recommendation:

- future options chain filtering design input

### Buy Price / Sell Price Assumption

Old behavior:

- configurable quote field, often buy at bid and sell at ask in config comments
- some OCO logic explicitly forces ask-side pricing due to order-placement issues

Recommendation:

- future options backtester and broker abstraction only
- do not reuse in stock research engine

### Take Profit / Stop Loss / Trailing Stop

Old behavior:

- OCO and trail order builders for live trading
- simple percentage-based backtest evaluation in `optimizestrats.py`

Recommendation:

- stock research app: keep the current generic stop/take-profit logic
- future options backtester: reuse the concepts only
- live-trading safety layer: future reference

### Queue Cancellation

Old behavior:

- queued entries older than `MAX_QUEUE_LENGTH` minutes are canceled

Recommendation:

- future paper/live trading safety layer

### Runner Logic

Old behavior:

- partial-position "runner" concept exists but is only partly used

Recommendation:

- future options backtester or paper trader only

### Multi-Strike Handling

Old behavior:

- can allow or disallow multiple strikes per underlying

Recommendation:

- future options paper/live workflows

### Sell-All-Positions Logic

Old behavior:

- forced day-trade close near end of day

Recommendation:

- future live-trading safety layer

### Live vs Paper Safety Logic

Old behavior:

- paper/live mode switch was global and fragile
- paper state often inferred by fake order IDs and Mongo updates

Recommendation:

- future broker layer should isolate paper vs live completely

## E. Broker and Data Audit

### Tradier

- Useful later as a future options broker abstraction reference
- Strongest value is endpoint shape, order-status lifecycle, and OTOCO/OCO conventions
- Do not integrate now

### TD Ameritrade

- Legacy reference only
- Token refresh, order flow, and websocket logic are tied to a retired or undesirable path
- Do not revive as the primary broker integration

### TD WebSocket

- Shows how open positions were monitored and repriced
- Useful only as future monitoring design input

### Polygon Backtest Logic

- Used for option minute-history pulls and crude OCO optimization
- Interesting as a historical experiment, not as production research infrastructure
- Fill assumptions are too optimistic and spread-unaware

### Quote and Delayed Data Assumptions

- Repo explicitly mixes TD quotes with Tradier execution because Tradier paper data was delayed
- This is a warning sign: data source inconsistency was baked into execution behavior

Recommendation:

- keep only the design lesson that execution and quote provenance must be explicit

## F. Alert Ingestion Audit

### Gmail Scanner

Classification:

- useful later for alert ingestion

Notes:

- parses Thinkorswim scanner emails into trade objects
- contains option-symbol parsing logic worth studying
- auto-trashes emails, which should not be the default in a future research ingestion service

### Discord Scanner

Classification:

- useful only as design reference

Notes:

- highly specific to one Discord bot’s embed format
- parsing is brittle and date-hardcoded to 2022-style option-symbol formatting

### Alert Parser / Duplicate Handling

Classification:

- useful later for alert ingestion

Notes:

- duplicate suppression through Mongo `analysis` records is conceptually useful
- should be rebuilt as idempotent local ingestion metadata later

## G. Persistence Audit

Old Mongo collections:

- `analysis`
- `users`
- `queue`
- `open_positions`
- `closed_positions`
- `rejected`
- `canceled`
- `strategies`
- `forbidden`
- `alert_history`

Mapping to the new app:

- `analysis` -> future `ingested_alerts` table if alert ingestion is added
- `strategies` -> current backtest parameter persistence already covers most research needs
- `open_positions`, `closed_positions`, `queue`, `rejected`, `canceled` -> future paper/live trading database, not current DuckDB research tables
- `users` -> avoid in current local research app unless broker accounts are added later
- `forbidden` -> future execution safety control table if needed

Recommendation:

- do not recreate the Mongo schema inside the stock research app

## H. Backtesting Audit

What the old backtest attempted:

- backtest option alert entries using Polygon option minute history
- optimize OCO-style take-profit and stop-loss settings
- optionally use Discord alerts or Mongo closed positions as trade sources

Limitations:

- relies on alert history instead of strategy-native signal generation
- likely look-ahead risk from using same-session high/low checks after entry
- no bid/ask spread modeling
- no slippage model
- no realistic liquidity model beyond simple filters
- no benchmark context
- no corporate-action handling
- no repeatable run metadata comparable to the new app

Conclusion:

- future reference only

## I. Security and Safety Audit

Findings:

- `config.py.example` includes example structures for Mongo URIs, Discord auth, webhooks, Tradier tokens, TD account data, and Polygon keys
- token handling is file-based and operationally fragile
- live-trading and paper-trading paths are mixed behind config switches
- order-safety confirmation gates are weak
- websocket-driven exit handling can fail silently if the process stops

Recommendations:

- do not copy any config, token, or secret handling patterns
- do not expose or commit any extracted credentials
- never let future paper and live modes share the same persistence semantics

## J. Migration Recommendations

### Phase A

Rebuild QQE and Hull Moving Average as research-only indicators in the new app.

### Phase B

Add `qqe_hma_strategy.py` only after the indicators are implemented and tested cleanly. Start with a simplified daily-data formulation, not a literal 5m/10m/30m port.

### Phase C

Document the intraday-dependent pieces that must wait:

- 5-minute QQE cross entries
- 10-minute HMA confirmation
- 30-minute QQE confirmation
- ADR intraday support/resistance logic

### Phase D

Use legacy options filters as design input for future option-chain screening:

- minimum option price
- maximum option price
- minimum volume
- minimum delta
- quote-field choice
- liquidity thresholds

### Phase E

Use old Tradier workflows only as future broker-abstraction design input:

- order payload shapes
- OTOCO/OCO lifecycle
- queue reconciliation
- stop-modification flows

Do not implement execution now.

### Phase F

Use Gmail/Discord parsing only as future alert-ingestion design input:

- webhook or inbox parsing
- symbol extraction
- option-contract extraction
- duplicate alert suppression

Do not implement alert trading now.
