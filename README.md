# Personal Trading Lab

Personal Trading Lab is a local Python-based stock backtesting and research application focused on repeatable daily-bar strategy evaluation. The project is intentionally stock-only for now so the research process, benchmark context, and robustness checks are defensible before adding options or live execution.

## Project Purpose

This project is built for:

- Daily stock strategy research
- Local data storage with incremental caching
- Repeatable backtests with persisted benchmark-relative results
- Robustness checks that reduce the odds of fooling yourself with overfit parameters, narrow market regimes, or one-trade outcomes

## Installation

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the environment template if needed:

```bash
cp .env.example .env
```

No API keys are required for the current version.

## How to Run the App

```bash
python -m streamlit run app.py
```

## How to Run Tests

```bash
python -m pytest
```

## Current Features

- `yfinance` daily OHLCV ingestion with incremental DuckDB-backed caching
- NYSE session-aware freshness and missing-session validation
- Research-only indicators package with WMA, HMA, RSI, and a transparent QQE-style implementation
- Saved backtests with persisted benchmark curves and research outputs
- SPY Workbench for a simplified SPY-only workflow
- Simplified four-tab navigation for the core SPY workflow
- Strategy Qualification workflow for apples-to-apples strategy comparison
- Stock Signal Scanner for daily signal triage across predefined universes
- Saved scanner snapshots for historical signal review
- Benchmark selection from the UI, defaulting to `SPY`
- Benchmark-relative metrics including excess CAGR, beta, and correlation
- Strategy audit findings with severity levels
- Profit concentration analysis by trade, ticker, and year
- Parameter sweep with a stability-oriented summary
- Predefined ticker universes plus editable custom universes
- Options Overlay Candidate heuristic for research triage only
- Slippage sensitivity checks across comparable strategy runs
- Manual trade-plan generation from live scanner results
- Local paper-trade journal with planned, open, closed, and canceled states
- Forward paper trading with active paper strategies, engine-managed orders, positions, trades, and equity curves
- Scanner-to-paper-trade linking so signals can be reviewed later against manual outcomes
- Paper-trade analytics with planned-vs-actual comparisons and R multiples
- Structured post-trade reviews and mistake-tag tracking
- Daily Trading Dashboard for signal counts and paper-trade review
- Watchlist with notes, tags, and latest signal status
- Train/test split mode
- Rolling walk-forward evaluation
- Regime analysis using the selected benchmark
- Robustness Score heuristic from 0 to 100
- Raw-price and adjusted-price modes
- Optional dividend-aware reporting in raw-price mode

## UI Layout

The app is organized into these tabs:

- Run Backtest
- SPY Workbench
- Saved Backtests
- Compare Backtests
- Signal Scanner
- Daily Trading Dashboard
- Forward Paper Trading
- Paper Journal
- Scanner History
- Strategy Qualification
- Parameter Sweep
- Train/Test
- Walk-Forward
- Regime Analysis
- Indicators
- Data Health
- Research Dashboard

## Indicators Package

The app now includes broker-agnostic, data-source-agnostic indicator modules under `trading_lab/indicators`.

Included indicators:

- Weighted Moving Average (`WMA`)
- Hull Moving Average (`HMA`)
- Wilder-style Relative Strength Index (`RSI`)
- QQE-style research indicator

These modules are research-only. They do not depend on Streamlit, broker code, or live trading paths.

## Indicator Preview

- The `Indicators` tab lets you preview HMA, RSI, or QQE on a selected ticker and date range.
- You can choose raw or adjusted price mode for the preview.
- The preview shows:
  - a price chart
  - an indicator chart
  - a recent-values table
- If the selected range is too short, the UI warns instead of forcing a misleading chart.

## HMA Explanation

- HMA is built from weighted moving averages.
- The implementation follows the standard formula:
  - `WMA(length / 2)`
  - `WMA(length)`
  - raw HMA = `2 * WMA(half) - WMA(full)`
  - final HMA = `WMA(raw HMA, sqrt(length))`
- In this project, HMA is used as a smoother trend filter rather than a direct execution model.

## RSI Method

- RSI uses Wilder-style smoothing.
- The implementation preserves index alignment and handles flat series without crashing.
- A flat price series converges to a neutral RSI around `50`.

## QQE-Style Indicator

- The QQE implementation here is inspired by the legacy repo, but it is not a literal port.
- It uses:
  - Wilder-style RSI as the base
  - smoothed RSI
  - an ATR-like measure of RSI changes
  - smoothed band distance
  - trailing-band-based trend inference
- This is intentionally transparent and documented so its behavior can be inspected and tested.
- It should be treated as a research indicator, not as a production trading signal by itself.

## Saved Sweeps

- Parameter sweeps now persist into dedicated `sweep_runs`, `sweep_results`, and `sweep_parameters` tables.
- Each saved sweep stores the strategy, benchmark, date range, price mode, risk settings, notes, tags, and parameter grid.
- Each sweep result links back to its saved backtest run when a normal run was persisted.
- Use the Saved Sweeps section to move from an experiment to the exact saved run that produced a result.

## Strategy Qualification Workflow

- The `Strategy Qualification` tab compares:
  - Moving Average Crossover
  - RSI Mean Reversion
  - Daily Breakout
  - QQE/HMA Daily
- Qualification runs reuse the same benchmark, date range, capital, slippage, commission, position sizing, and risk settings so the comparison stays fair.
- Results are persisted into dedicated `strategy_qualification_runs` and `strategy_qualification_results` tables.
- Qualification output focuses on:
  - headline performance
  - benchmark-relative performance
  - robustness
  - red-flag count
  - options-overlay readiness

## SPY Workbench

- `SPY Workbench` is the simplified recommended workflow in the app.
- It is fixed to `SPY`.
- It always compares the selected strategy against buy-and-hold `SPY`.
- It is designed to reduce complexity and reduce overfitting pressure from too many symbols and universes.
- It now includes an `Automated SPY Search` workflow so you do not have to manually test every approved strategy-and-exit combination one at a time.
- It can promote one selected SPY strategy into forward paper trading.
- It does not place real trades and it does not connect to a broker.
- It should be used for forward paper validation before real money is ever considered.

## Simplified Navigation

- The app now uses four main tabs:
  - `SPY Workbench`
  - `Forward Paper`
  - `Research History`
  - `Data & Settings`
- `SPY Workbench` is the recommended starting point.
- `Forward Paper` is for the promoted SPY strategy only.
- `Research History` is for reviewing prior searches, backtests, sweeps, and qualification results.
- `Data & Settings` is for refresh, cache status, diagnostics, and database export.
- Advanced and legacy workflows are still available, but they now live inside expanders instead of the top-level navigation.

## Advanced Mode

- The sidebar includes `Show Advanced Tools`.
- Default: `off`
- When advanced mode is off:
  - the app stays focused on the SPY workflow
  - broad legacy research controls stay hidden
  - the main tabs emphasize search, promotion, and forward review
- When advanced mode is on:
  - additional research tools appear inside expanders
  - legacy multi-ticker research, scanner history, parameter sweep, train/test, walk-forward, and manual paper journal remain accessible

## Automated SPY Strategy Search

- The automated search runs a controlled grid of approved `SPY` entry presets against approved implemented exit presets.
- The default grid is intentionally limited so it stays practical and does not become an uncontrolled optimization exercise.
- Current entry presets include:
  - 3 trend-filter variants
  - 3 moving-average crossover variants
  - 3 RSI pullback variants
  - 3 breakout variants
  - 2 conservative `QQE/HMA` variants marked experimental
- Current implemented exit presets include:
  - signal exit only
  - fixed stop loss
  - fixed take profit
  - OCO bracket
  - trailing stop
  - stop loss plus trailing stop
  - time stop
- The default search currently tests `308` SPY-only combinations.
- Ranking is based on robustness-oriented scoring rather than highest return alone.
- Each result receives:
  - candidate label
  - red-flag count
  - plain-English summary
  - ranking-category highlights such as `Best Overall` and `Most Suspicious High Return`
- You can promote one selected search result directly into SPY-only forward paper trading.
- Do not blindly select the highest CAGR. The workflow is designed to surface suspicious high-return results so they can be rejected on purpose.

The SPY-only presets currently include:

- `SPY 200-Day Trend Filter`
- `SPY Moving Average Crossover`
- `SPY RSI Pullback in Uptrend`
- `SPY Breakout`
- `SPY QQE/HMA Daily`

Each preset starts with conservative defaults. Parameters stay frozen unless you explicitly choose to customize them.

## Entry Strategy Selection

- The workbench lets you select one SPY entry strategy at a time.
- Current choices are:
  - `SPY 200-Day Trend Filter`
  - `SPY Moving Average Crossover`
  - `SPY RSI Pullback in Uptrend`
  - `SPY Breakout`
  - `SPY QQE/HMA Daily`
- The focus is to compare a small number of sensible SPY ideas rather than scanning many symbols.

## Exit Structure Selection

- The workbench separates the SPY entry idea from the exit structure.
- Implemented exit structures currently include:
  - `Signal exit only`
  - `Fixed stop loss`
  - `Fixed take profit`
  - `OCO bracket`
  - `Trailing stop`
  - `Stop loss plus trailing stop`
  - `Time stop`
- Planned but disabled structures are shown in the UI so the workflow is honest about what is and is not implemented yet.

## SPY Vs Buy-And-Hold Comparison

- The SPY workbench compares strategy performance directly to buy-and-hold `SPY`.
- It highlights:
  - total return
  - CAGR
  - max drawdown
  - Sharpe
  - Calmar
  - trade count
  - win rate
  - profit factor
  - exposure / time in market
  - excess CAGR vs SPY
  - drawdown improvement vs SPY
- The summary text is intentionally plain:
  - beat SPY with lower drawdown
  - reduced drawdown but underperformed
  - did not improve risk-adjusted results
  - too few trades to trust

## SPY Robustness Checklist

- The SPY lab uses a simpler checklist than the broader research tabs.
- It asks whether the strategy:
  - beats SPY on CAGR
  - improves max drawdown
  - improves Calmar
  - has enough trades
  - survives train/test
  - survives walk-forward
  - has acceptable parameter stability
  - avoids extreme profit concentration
  - holds up under modest slippage
- Each line is labeled `Pass`, `Caution`, or `Fail`.
- The final label is:
  - `Strong SPY candidate`
  - `Possible SPY candidate`
  - `Not ready`

## Candidate Labels And Ranking Categories

- Automated SPY search grades each tested combination as:
  - `Strong candidate`
  - `Possible candidate`
  - `Not ready`
  - `Reject`
- The grading emphasizes:
  - positive CAGR
  - positive excess CAGR vs SPY
  - acceptable drawdown
  - usable Calmar
  - trade count
  - profit factor
  - average R multiple
  - red-flag count
  - strategy complexity
- Search highlights are grouped into:
  - `Best Overall`
  - `Best Low Drawdown`
  - `Best Risk Adjusted`
  - `Best Simple Strategy`
  - `Most Suspicious High Return`
- `Most Suspicious High Return` exists to catch results that look exciting but are fragile, overfit, under-traded, or too complex to trust.

## SPY Parameter Stability

- Each SPY preset supports a limited sweep rather than a wide optimization grid.
- The goal is to see whether nearby parameter choices still work against buy-and-hold SPY.
- The lab reports:
  - best performance
  - median performance
  - worst performance
  - percent of tested parameter sets beating SPY
- If only one narrow parameter setting works well, the lab warns that the result may be overfit.

## Exit Comparison

- The workbench can hold the SPY entry strategy constant and compare several exit structures side by side.
- The comparison emphasizes:
  - CAGR
  - excess CAGR vs SPY
  - max drawdown
  - drawdown improvement
  - Sharpe
  - Calmar
  - trade count
  - win rate
  - profit factor
  - average R multiple
- A plain-English summary points out which exit performed best, which reduced drawdown most, which looked strongest on a risk-adjusted basis, and which had too few trades to trust.

## SPY Forward Paper Promotion

- A selected SPY strategy can be promoted directly from the SPY workbench into forward paper trading.
- Promotion can come from either the manual workbench run or a selected automated SPY search result.
- Promotion freezes:
  - strategy name
  - parameters
  - activation date
  - price mode
  - fill rule
  - slippage
  - commission
  - risk settings
- The promoted strategy is stored as SPY-only and forward paper trading will only track `SPY` for that strategy.

## Forward Paper Review

- The SPY workbench also surfaces the current forward-paper status for SPY-only strategies.
- It shows:
  - active strategy
  - status
  - activation date
  - pending orders
  - open positions
  - closed trades
  - current paper equity
  - realized P&L
  - latest signal
  - next expected action
  - recent event log
- Closed forward-paper trades are summarized inside the workbench so the normal SPY review loop stays in one place.

## Signal Scanner

- The `Signal Scanner` tab checks the latest available daily bar across a selected universe.
- It supports:
  - `SPY`
  - `Large-cap tech`
  - `Sector ETFs`
  - `Broad ETFs`
  - `Defensive ETFs`
  - `Cyclical ETFs`
  - `Custom`
- It scans:
  - Moving Average Crossover
  - RSI Mean Reversion
  - Daily Breakout
  - QQE/HMA Daily
- Each scanner row includes:
  - signal type
  - signal date
  - latest close
  - entry, stop, and target references
  - reward/risk ratio
  - saved robustness context
  - qualification status when available
  - notes and warnings

## Saved Scanner Snapshots

- A scanner run can be saved as a snapshot.
- Each snapshot stores:
  - universe
  - tickers
  - strategies
  - benchmark
  - price mode
  - scanner result rows
  - notes and tags
- Saved snapshots let you answer:
  - what the scanner showed on a given day
  - which signals later became paper trades
  - which signals led to no action

## Scanner To Paper-Trade Linking

- When you create a paper trade from a scanner result, the app can store:
  - scanner snapshot id
  - scanner result id
  - signal quality score
  - qualification status
  - explanation
  - warnings at scan time
- Saved scanner snapshots show whether each signal became:
  - `no_action`
  - `planned`
  - `open`
  - `closed`
  - `canceled`

## Signal Explanations

- Each scanner result includes a plain-English explanation tied to the actual strategy conditions on the latest bar.
- Example outputs explain:
  - why a breakout triggered
  - why a moving-average state remains active
  - why a filtered signal did not qualify
  - why a QQE/HMA setup still needs caution

## Manual Trade Plans

- Scanner results can be converted into manual trade plans.
- Supported sizing modes:
  - fixed dollar allocation
  - percent of portfolio
  - fixed dollar risk per trade
- Trade plans include:
  - planned entry
  - stop loss
  - take profit
  - risk per share
  - position size
  - estimated capital required
  - max dollar risk
  - reward/risk ratio
  - notes and tags

## Paper Trade Journal

- Paper trades are stored locally in `paper_trades` and `paper_trade_events`.
- Statuses:
  - `planned`
  - `open`
  - `closed`
  - `canceled`
- Trades are manually advanced by the user. The app does not connect to a broker and does not place trades.
- Realized paper P&L is calculated when a trade is closed.

## Forward Paper Trading

- The `Forward Paper Trading` tab is a separate engine-managed workflow for qualified stock strategies.
- Active paper strategies are stored in `active_paper_strategies` with frozen parameters, risk rules, benchmark, sizing, notes, and tags.
- Engine-managed forward state is stored in:
  - `forward_paper_orders`
  - `forward_paper_positions`
  - `forward_paper_trades`
  - `forward_paper_equity_curve`
- This is not broker paper trading. It is a local daily-bar simulation that advances only when you refresh data and run the forward update.

## Recommended First Use

Suggested first test:

1. Use `SPY` only.
2. Test `SPY 200-Day Trend Filter`.
3. Test `SPY Moving Average Crossover`.
4. Test `SPY RSI Pullback in Uptrend`.
5. Compare each result to buy-and-hold `SPY`.
6. Promote only one strategy to forward paper trading.

## Recommended SPY Workflow

1. Choose one SPY entry strategy.
2. Choose one exit structure.
3. Run the backtest or click `Run Automated SPY Search`.
4. Compare the result set to buy-and-hold `SPY`.
5. Ignore suspicious high-return results.
6. Compare exit structures.
7. Review robustness.
8. Promote one configuration to forward paper trading.
9. Run forward paper updates as new daily data arrives.
10. Review forward results before considering real money.

## Recommended Daily Workflow

1. Open `SPY Workbench`.
2. Run `Automated SPY Search`.
3. Review ranked candidates.
4. Promote one candidate to forward paper.
5. Open `Forward Paper`.
6. Run `Forward Paper Update` after new daily data exists.
7. Review forward results.
8. Use `Research History` only when reviewing prior experiments.
9. Use `Data & Settings` for refresh and diagnostics.

## Active Paper Strategies

- A saved qualification result, saved backtest, saved sweep result, or manual strategy configuration can be promoted into an active paper strategy.
- Promotion stores the exact strategy parameters and risk settings at activation time.
- Strategy lifecycle states:
  - `draft`
  - `active`
  - `paused`
  - `retired`
- Promotion uses a checklist so weak historical research is not silently promoted into forward tracking.

## Fill Assumptions

- Entry default: `next_open`
- Optional entry mode: `next_close`
- Exit handling uses daily-bar high/low checks for:
  - stop loss
  - take profit
  - trailing stop
- If stop and target are both hit in the same bar, the default is `conservative_stop_first`.
- Other ambiguity modes are available:
  - `target_first`
  - `skip_ambiguous`
- These are still simplified paper-trading assumptions, not executable market fills.

## Forward Validation Reports

- Each active paper strategy can be reviewed against its linked research expectations.
- The report compares items such as:
  - original backtest CAGR
  - original max drawdown
  - original win rate
  - original profit factor
  - forward-paper CAGR
  - forward-paper drawdown
  - forward-paper win rate
  - forward-paper profit factor
- Warnings are generated if forward-paper results are materially worse, if the live sample is still too short, or if too few trades have closed.

## Paper-Trade Analytics

- The paper journal now summarizes:
  - total realized P&L
  - average realized return
  - win rate
  - average winning trade
  - average losing trade
  - profit factor
  - expectancy per trade
  - average holding period
  - best and worst trades
  - average planned reward/risk
  - realized R multiple
- Grouping views are available by:
  - strategy
  - ticker
  - universe
  - tag
  - signal quality
  - qualification status
  - robustness bucket

## Post-Trade Review Workflow

- Closed paper trades can be reviewed with structured fields such as:
  - thesis review
  - execution review
  - what went well
  - what went wrong
  - lesson learned
  - mistake tags
  - followed-plan flag
  - entry quality
  - exit quality
  - emotional discipline
- The goal is to make recurring workflow mistakes visible before any broker integration is added.

## Planned Vs Actual Comparison

- For each closed paper trade, the app compares:
  - planned entry vs actual entry
  - planned stop vs actual exit
  - planned target vs actual exit
  - planned dollar risk vs actual P&L
  - planned reward/risk vs realized R multiple
- `R multiple` is realized P&L divided by initial planned dollar risk.

## Daily Trading Dashboard

- The daily dashboard summarizes:
  - new buy signals
  - active long signals
  - exit signals
  - open paper trades
  - planned trades
  - closed-trade win rate
  - realized paper P&L
  - average closed-trade return
- It also surfaces:
  - highest-quality current signals
  - signals from qualified strategies
  - lower-quality signals that likely deserve to be ignored
  - open paper trades approaching stops or targets

## Scanner History

- The `Scanner History` tab summarizes:
  - number of saved snapshots over time
  - new buy signals by snapshot date
  - exit signals by snapshot date
  - recurring tickers by signal count
  - strategies producing more high-quality or low-quality signals
- More signals are not automatically better. The history is meant for review, not signal-count chasing.

## Signal Quality Score

- Each scanner result receives a `Signal Quality Score` from 0 to 100.
- Labels:
  - `High quality`
  - `Watch`
  - `Low quality`
  - `Ignore`
- Inputs may include:
  - robustness score
  - qualification status
  - trade count adequacy
  - reward/risk ratio
  - data warnings
  - corporate-action warnings
  - parameter-stability warnings
  - regime context
- This is a research triage tool, not a recommendation.

## Watchlist

- The watchlist stores tickers locally with notes and tags.
- Watchlist rows can be enriched with the latest scanner signal state when a fresh scan has been run.
- Categories include:
  - `general watch`
  - `high priority`
  - `waiting for pullback`
  - `waiting for breakout`
  - `avoid`

## Predefined Universes

- The qualification workflow supports reusable universe presets:
  - `Single benchmark`
  - `Large-cap tech`
  - `Sector ETFs`
  - `Broad ETFs`
  - `Defensive ETFs`
  - `Cyclical ETFs`
  - `Custom`
- You can start from a preset and then edit the ticker list before running the qualification.

## Exchange Calendar Handling

- The app uses NYSE trading sessions for cache freshness and missing-session checks.
- Weekends and exchange holidays are excluded from expected-session checks.
- If `pandas_market_calendars` cannot be loaded, the app falls back to a simpler business-day approximation with holiday handling.

## Benchmark Selection

- You can select a benchmark symbol in the sidebar.
- The selected benchmark is used in:
  - benchmark-relative metrics
  - comparison charts
  - train/test reporting
  - walk-forward reporting
  - regime classification
  - robustness scoring
- Benchmark data is cached through the same daily provider flow as strategy tickers.

## Benchmark Diagnostics

- Each saved run now stores benchmark diagnostics alongside the backtest.
- Diagnostics flag:
  - missing benchmark data
  - weaker benchmark date coverage than strategy coverage
  - missing NYSE sessions
  - unusual runs of zero benchmark returns
  - excessive date dropping during alignment
- Treat weak benchmark diagnostics as a sign that benchmark-relative conclusions may be less trustworthy.

## Rolling Walk-Forward Evaluation

- Rolling walk-forward uses configurable train windows, test windows, and step sizes in months.
- Each fold stores train/test dates and metrics.
- Aggregate walk-forward summary metrics include test-fold profitability, degradation, and a consistency score.
- This version does not optimize parameters inside each fold.

## Regime Analysis

- Regimes are defined from the selected benchmark.
- Trend regimes:
  - bull: benchmark close above 200-day SMA
  - bear: benchmark close below 200-day SMA
- Volatility regimes:
  - high volatility: 20-day rolling volatility above its median
  - low volatility: 20-day rolling volatility below its median
- The app reports return, CAGR, drawdown, Sharpe, Sortino, Calmar, trade count, win rate, profit factor, and average trade return by regime.

## Robustness Score

- The Robustness Score is a research heuristic from 0 to 100.
- It is not a prediction and not a trading recommendation.
- The score considers factors such as:
  - benchmark-relative performance
  - drawdown severity
  - trade count
  - train/test degradation
  - walk-forward consistency
  - profit concentration
  - regime dependence
- parameter stability

## Options Overlay Candidate Flag

- The `Options Overlay Candidate` flag is a research heuristic, not a recommendation.
- It asks whether a stock strategy is strong enough to justify future options research.
- Default inputs include:
  - trade count
  - positive CAGR
  - positive excess CAGR versus benchmark
  - benchmark-comparable drawdown
  - robustness score
  - profit concentration
  - train/test behavior when available
  - walk-forward behavior when available
  - parameter stability when available
- Candidate labels:
  - `Strong candidate`
  - `Possible candidate`
  - `Not ready`
- A strategy should not move toward options just because the stock backtest return looks high.

## Slippage Sensitivity

- Strategy Qualification includes a simple slippage sensitivity run across several default slippage levels.
- Use it to see whether CAGR, profit factor, or trade economics collapse under modest friction.
- If the edge falls apart quickly, treat the stock signal as weaker than the raw backtest suggests.

## Research Dashboard

- The Research Dashboard surfaces saved runs ranked by robustness, CAGR, Sharpe, Calmar, drawdown profile, and benchmark-relative performance.
- It also highlights red flags such as:
  - high profit concentration
  - weak train/test degradation
  - poor walk-forward consistency
  - regime dependence
  - too few trades
  - underperformance versus benchmark
- Notes and tags can be used to filter saved runs and saved sweeps into coherent experiments.
- It also highlights strong options-overlay candidates, best benchmark-relative runs, and recent qualification winners by universe.

## Notes and Tags

- Backtest runs and parameter sweeps both support optional notes and comma-separated tags.
- Example tags:
  - `momentum`
  - `mean-reversion`
  - `options-candidate`
  - `promising`
  - `overfit-risk`
- Notes are intentionally lightweight. They are there to preserve research context, not to become a full journal system.

## Adjusted vs Raw Price Modes

- `raw_price_mode`
  - Uses raw prices for signals and marking
  - Works best for shorter studies or when you explicitly want raw-price behavior
  - Can be distorted by splits and dividends over longer histories unless fully modeled

- `adjusted_price_mode`
  - Uses adjusted close for signal context and benchmark tracking where available
  - Helps reduce obvious distortions from splits in long histories
  - Is labeled as adjusted-price based research

## Dividend Handling

- `price_return_only`
  - Reports price-only results

- `total_return_with_dividends`
  - Credits dividend cash to held positions in raw-price mode
  - Avoids double-counting when adjusted-price mode is used

This is still a simplified accounting model. It does not attempt tax modeling or full tax-lot accounting.

## QQE/HMA Daily Strategy

- A research-only daily `QQE/HMA` strategy is now available in the strategy selector.
- It is intentionally simpler than the legacy intraday options workflow.
- The daily version buys when:
  - the QQE-style trend turns bullish
  - price is above HMA
  - HMA slope is positive, if enabled
- It exits when:
  - the QQE-style trend turns bearish, if enabled
  - price falls below HMA, if enabled
- This is a daily stock adaptation of an intraday legacy idea. Treat it cautiously.
- A more complex signal like QQE/HMA should beat simpler strategies on robustness, benchmark-relative performance, and stability before it earns more research attention.

## Corporate-Action Warnings

- The app stores dividends and splits in the `corporate_actions` table.
- During a backtest, it warns when:
  - split events occurred in raw-price mode
  - large dividends occurred in raw-price mode
  - adjusted-price mode was selected but adjusted data may not be reliable enough
- These warnings are practical sanity checks, not a full institutional accounting engine.

## How to Interpret Results

- High return alone is not enough.
- A complex indicator is not automatically better than a simple one.
- Compare the strategy to the selected benchmark.
- Check drawdown, not just CAGR.
- Check trade count before trusting the summary.
- Check train/test degradation.
- Check walk-forward consistency.
- Check profit concentration.
- Check whether benchmark diagnostics are clean.
- Check regime dependence.
- Check parameter stability instead of only the best parameter set.
- Check whether the signal survives slippage sensitivity.
- Check whether the strategy is even ready for a later options overlay.

## Stock Trading Workflow

1. Run Strategy Qualification.
2. Identify strategies with acceptable robustness.
3. Scan a selected universe.
4. Review signal explanations and warnings.
5. Create a manual trade plan.
6. Track the trade in the paper journal.
7. Review paper-trade results before any future broker integration.

## Manual Stock Trading Workflow

1. Run research and qualification.
2. Run scanner.
3. Save scanner snapshot.
4. Review high-quality signals.
5. Create a trade plan.
6. Open paper trade manually.
7. Close paper trade manually.
8. Complete post-trade review.
9. Review analytics.
10. Decide whether the strategy deserves continued attention.

## Forward Paper Trading Workflow

1. Run research.
2. Run strategy qualification.
3. Promote a qualified strategy to an active paper strategy.
4. Refresh daily data.
5. Run Forward Paper Update.
6. Review pending orders and open positions.
7. Review forward paper results weekly.
8. Retire strategies that underperform or violate expectations.

## How to Interpret Sweep Results

- The highest-return parameter set is not automatically the best strategy.
- A parameter set that only wins in one narrow pocket is a warning sign, not proof of edge.
- Prefer regions where nearby parameter combinations remain acceptable.
- Look for sweeps where a healthy share of parameter sets:
  - make money
  - beat the benchmark
  - stay inside your drawdown tolerance
- If only one narrow setting looks good, the sweep may be describing overfit rather than edge.

## Research Warnings

- This is for research only.
- This is not financial advice.
- The app does not place trades.
- Paper trades are manually tracked and depend on your own manual updates.
- Forward paper trading is engine-managed inside the app, but it still does not connect to a broker, sync broker positions, or place real orders.
- Signal quality, qualification context, and paper analytics are heuristics and review tools, not recommendations.
- `yfinance` is useful for prototyping but should be validated before using for serious trading decisions.
- Backtest results can be misleading due to survivorship bias, look-ahead bias, overfitting, and unrealistic fill assumptions.
- A strong-looking backtest can still be dominated by one trade, one ticker, one year, one regime, or one narrow parameter region.
- The QQE/HMA strategy adapts an intraday legacy concept to daily data, so positive results do not prove the original intraday idea survives unchanged.

## Why Options Are Still Excluded

Options remain intentionally excluded because credible options research requires:

- contract metadata
- chain snapshots
- bid/ask-aware fill modeling
- expiration rules
- IV and Greeks
- spread-aware execution assumptions

Those concerns deserve a separate phase instead of being mixed into the stock research core prematurely.

The new qualification workflow exists specifically to delay options work until the stock signal has earned it.

## Pre-Options Checklist

Before expressing a stock signal with options, the underlying stock strategy should show:

- sufficient trade count
- acceptable drawdown
- positive benchmark-relative performance
- tolerable train/test degradation
- reasonable walk-forward consistency
- no extreme profit concentration
- not dependent on one ticker
- not dependent on one year
- stable parameter region
- clear regime behavior
- clean benchmark diagnostics
- no major unresolved data-quality warnings

## Current Limitations

- Options backtesting is not implemented yet.
- Intraday strategies are not implemented yet.
- The engine is still daily-bar and long-only.
- Forward paper trading uses daily-bar simulation and simplified next-bar fill assumptions.
- Walk-forward is rolling but still simple; it is not a full optimizer.
- Raw-price mode can still be distorted around corporate actions.
- Adjusted-price mode improves continuity but should not be confused with executable fills.
- The QQE implementation is a documented QQE-style reconstruction, not an exact clone of the legacy intraday behavior.

## Roadmap

### Current Phase

- Daily stock research platform with benchmark-aware robustness analysis

### Next Likely Phases

- Review the old GitHub repo for reusable ideas without porting old code directly
- Build the options data foundation starting with option chain snapshots, not full options backtesting
- Better corporate-action-aware accounting
- Broader benchmark/reporting support
- Stronger walk-forward and regime persistence workflows
- Eventually, options research once the stock research core is trustworthy enough
