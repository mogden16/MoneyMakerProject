# Legacy Migration Plan

This plan is intentionally selective. The legacy repo is reference material, not a donor project.

## Ordered Reusable Concepts

1. Rebuild QQE as a research-only indicator module.
2. Rebuild Hull Moving Average as a research-only indicator module.
3. Design a simplified daily `qqe_hma_strategy.py` only after the indicators are tested.
4. Preserve option-liquidity filter concepts for future options chain screening.
5. Preserve broker-lifecycle ideas from Tradier for a future broker abstraction.
6. Preserve duplicate-alert handling and parsing ideas for future alert ingestion.
7. Preserve queue-expiry, forced-liquidation, and paper-vs-live safety concepts for a future execution layer.

## What Should Be Rebuilt First

### First

- `trading_lab/indicators/qqe.py`
- `trading_lab/indicators/hma.py`
- unit tests for both
- documentation describing how the legacy intraday logic differs from the new daily research context

### Second

- `trading_lab/strategies/qqe_hma_strategy.py`
- a daily-data version only
- no alert dependency
- no broker dependency
- no options dependency

### Third

- documentation for future intraday prerequisites:
  - intraday data provider
  - multi-timeframe bar alignment
  - realistic fill assumptions
  - option quote modeling

## What Should Wait

- Tradier integration
- any broker abstraction implementation
- option chain ingestion
- option contract selection
- Gmail ingestion
- Discord ingestion
- webhook ingestion
- intraday strategies
- paper trading
- live trading
- OpenCV/screen-scanning workflows

## What Should Be Discarded

- TD Ameritrade execution code
- TD WebSocket implementation
- Mongo-backed queue/open/closed/rejected/canceled persistence model
- global in-memory lists in `constants.py`
- the monolithic `main.py` loop
- config patterns that mix secrets, execution mode, and strategy settings in one file
- legacy backtest code as a source of truth for research quality

## Risk Notes

- The legacy strategy logic is intraday and option-oriented. A literal port into daily stock backtesting would be misleading.
- The legacy backtest assumes fills that are too optimistic for options research.
- The legacy repo contains risky operational patterns around secrets and live/paper toggles.
- Tradier ideas are more useful than TD Ameritrade ideas because Tradier still fits the future options direction better.
- Gmail and Discord scanners are format-specific and should be treated as parser references, not reusable components.

## Recommended Sprint Sequence After This Audit

1. Rebuild `hma.py` with tests.
2. Rebuild `qqe.py` with tests.
3. Add a research-only `qqe_hma_strategy.py` if the daily formulation is defensible.
4. Document intraday-only legacy behavior that should remain deferred.
5. Add an `indicators/` package to the new app if not already present.
6. Expand strategy comparison reporting to include the new indicator-based strategy once validated.
7. Only after the stock signal layer is credible, start the options data foundation with schema and provider interfaces.

## Sprint 3 Update

What was rebuilt:

- `trading_lab/indicators/moving_average.py`
- `trading_lab/indicators/hma.py`
- `trading_lab/indicators/rsi.py`
- `trading_lab/indicators/qqe.py`
- indicator preview support in the Streamlit app

What was not rebuilt:

- legacy intraday 5m/10m/30m confirmation logic
- ADR-based intraday support/resistance filters
- Discord or Gmail alert flows
- broker execution workflows

Did QQE/HMA become a daily strategy:

- yes, as a research-only daily stock strategy
- no, as an exact legacy-equivalent intraday options workflow

What still requires intraday data:

- multi-timeframe QQE confirmation
- 10-minute and 30-minute HMA/QQE agreement
- intraday resistance-zone logic
- realistic same-session timing behavior

## Sprint 4 Strategy Qualification Update

What was added:

- a `Strategy Qualification` workflow for comparing QQE/HMA Daily against the simpler baseline strategies
- predefined ticker universes with editable custom lists
- persisted `strategy_qualification_runs` and `strategy_qualification_results`
- an `Options Overlay Candidate` heuristic for research triage
- slippage sensitivity checks
- saved sweep stability comparison across strategies

What the qualification tools are meant to answer:

- whether QQE/HMA is actually stronger than the simpler strategies
- whether QQE/HMA survives train/test, walk-forward, and parameter-stability review
- whether a stock signal is strong enough to justify later options-overlay research

Whether QQE/HMA should move forward:

- only if it beats simpler strategies on more than headline return
- only if it shows acceptable robustness, benchmark-relative behavior, and enough trades
- only if parameter stability and slippage sensitivity do not look fragile

What still waits for intraday data:

- legacy 5m/10m/30m confirmation logic
- intraday-only regime timing
- same-session alert timing and execution nuance
- any attempt to reproduce the legacy options-entry workflow faithfully
