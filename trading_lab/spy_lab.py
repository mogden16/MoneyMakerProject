from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import dataclass
from itertools import product
from typing import Any
from uuid import uuid4

import pandas as pd

from trading_lab.backtest.engine import BacktestConfig, BacktestEngine
from trading_lab.backtest.qualification import summarize_slippage_warnings
from trading_lab.backtest.robustness import compute_robustness_score, parameter_stability_summary, profit_concentration_analysis
from trading_lab.backtest.sweep import run_parameter_sweep
from trading_lab.backtest.train_test import run_train_test_analysis, split_data_by_percentage
from trading_lab.backtest.walk_forward import run_walk_forward_analysis
from trading_lab.data.intraday import INTRADAY_MAX_HISTORY_DAYS, is_intraday_timeframe
from trading_lab.signals.scanner import determine_signal_type
from trading_lab.strategies.breakout import BreakoutStrategy
from trading_lab.strategies.intraday_qqe_hma import IntradayQQEHMAStateStrategy
from trading_lab.strategies.intraday_breakout import IntradayBreakoutStrategy
from trading_lab.strategies.intraday_pullback import IntradayPullbackStrategy
from trading_lab.strategies.moving_average import MovingAverageCrossStrategy
from trading_lab.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from trading_lab.strategies.qqe_hma_strategy import QQEHMAStrategy
from trading_lab.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from trading_lab.strategies.swingarm_trend import SwingArmTrendStrategy
from trading_lab.strategies.trend_filter import TrendFilterStrategy


@dataclass(frozen=True)
class SpyStrategyPreset:
    key: str
    label: str
    strategy_name: str
    description: str
    parameters: dict[str, Any]
    parameter_grid: dict[str, list[int | float]]
    timeframes: tuple[str, ...] = ("1d",)
    experimental: bool = False


@dataclass(frozen=True)
class SpyExitStructure:
    key: str
    label: str
    description: str
    enabled: bool
    default_params: dict[str, Any]


@dataclass(frozen=True)
class SpyWorkbenchConfig:
    preset_key: str
    entry_label: str
    entry_parameters: dict[str, Any]
    timeframe: str
    exit_structure_key: str
    exit_structure_label: str
    exit_parameters: dict[str, Any]
    start_date: str
    end_date: str
    price_mode: str
    initial_capital: float
    position_sizing_method: str
    position_size_value: float
    max_positions: int
    slippage_pct: float
    commission_per_trade: float


@dataclass(frozen=True)
class SpySearchEntryPreset:
    preset_id: str
    preset_key: str
    entry_strategy_name: str
    label: str
    parameters: dict[str, Any]
    description: str
    complexity_score: int
    timeframe: str = "1d"
    experimental: bool = False


@dataclass(frozen=True)
class SpySearchExitPreset:
    exit_preset_id: str
    exit_structure_key: str
    exit_structure_name: str
    label: str
    parameters: dict[str, Any]
    description: str


@dataclass(frozen=True)
class SpySearchCombination:
    combination_id: str
    entry_preset: SpySearchEntryPreset
    exit_preset: SpySearchExitPreset


SPY_PRESETS: dict[str, SpyStrategyPreset] = {
    "trend_filter_200": SpyStrategyPreset(
        key="trend_filter_200",
        label="SPY 200-Day Trend Filter",
        strategy_name="SPY 200-Day Trend Filter",
        description="Long-only trend filter that stays invested while SPY remains above its 200-day SMA.",
        parameters={"sma_length": 200},
        parameter_grid={"sma_length": [150, 175, 200, 225, 250]},
    ),
    "moving_average_50_200": SpyStrategyPreset(
        key="moving_average_50_200",
        label="SPY Moving Average Crossover",
        strategy_name="Moving Average Crossover",
        description="Classic 50/200 SMA crossover to reduce large bear-market drawdowns.",
        parameters={"fast_window": 50, "slow_window": 200},
        parameter_grid={"fast_window": [20, 50, 75], "slow_window": [100, 150, 200]},
    ),
    "rsi_pullback_uptrend": SpyStrategyPreset(
        key="rsi_pullback_uptrend",
        label="SPY RSI Pullback in Uptrend",
        strategy_name="RSI Mean Reversion",
        description="Buys SPY pullbacks only while the long-term trend remains positive.",
        parameters={"rsi_length": 14, "buy_threshold": 35.0, "sell_threshold": 55.0, "max_holding_days": 10, "trend_sma_window": 200},
        parameter_grid={"buy_threshold": [25.0, 30.0, 35.0, 40.0], "sell_threshold": [50.0, 55.0, 60.0], "trend_sma_window": [150, 200]},
    ),
    "breakout_50_20": SpyStrategyPreset(
        key="breakout_50_20",
        label="SPY Breakout",
        strategy_name="Daily Breakout",
        description="Buys strength above a prior range high and exits when SPY loses a shorter breakout floor.",
        parameters={"lookback_window": 50, "exit_lookback_window": 20},
        parameter_grid={"lookback_window": [20, 50, 100], "exit_lookback_window": [10, 20, 50]},
    ),
    "qqe_hma_daily": SpyStrategyPreset(
        key="qqe_hma_daily",
        label="SPY QQE/HMA Daily",
        strategy_name="QQE/HMA Daily",
        description="Experimental daily adaptation of the QQE/HMA idea with conservative defaults.",
        parameters={
            "hma_length": 21,
            "rsi_length": 14,
            "rsi_smoothing": 5,
            "qqe_factor": 4.236,
            "atr_smoothing": 5,
            "require_hma_slope": True,
            "exit_on_hma_break": True,
            "exit_on_qqe_bearish": True,
        },
        parameter_grid={"hma_length": [18, 21, 24], "rsi_length": [12, 14], "qqe_factor": [3.5, 4.236, 5.0]},
        experimental=True,
    ),
    "intraday_pullback": SpyStrategyPreset(
        key="intraday_pullback",
        label="Daily Trend + Intraday Pullback",
        strategy_name="Daily Trend + Intraday Pullback",
        description="Uses the completed daily 200-day trend filter and buys completed intraday pullback recoveries.",
        parameters={"rsi_length": 14, "oversold_threshold": 35.0, "recovery_threshold": 45.0, "moving_average_length": 8, "pullback_lookback_bars": 4, "require_daily_regime": True, "end_of_day_exit": True, "allow_overnight": False},
        parameter_grid={"oversold_threshold": [30.0, 35.0, 40.0], "recovery_threshold": [40.0, 45.0, 50.0], "moving_average_length": [6, 8, 10]},
        timeframes=("15m", "5m"),
    ),
    "intraday_breakout": SpyStrategyPreset(
        key="intraday_breakout",
        label="Daily Trend + Intraday Breakout",
        strategy_name="Daily Trend + Intraday Breakout",
        description="Uses the completed daily 200-day trend filter and buys completed intraday breakouts.",
        parameters={"breakout_lookback_bars": 12, "exit_lookback_bars": 6, "require_daily_regime": True, "end_of_day_exit": True, "allow_overnight": False},
        parameter_grid={"breakout_lookback_bars": [8, 12, 16], "exit_lookback_bars": [4, 6, 8]},
        timeframes=("15m", "5m"),
    ),
    "opening_range_breakout": SpyStrategyPreset(
        key="opening_range_breakout",
        label="Opening Range Breakout",
        strategy_name="Opening Range Breakout",
        description="Long-only SPY opening-range breakout with optional volume and QQE confirmation.",
        parameters={"breakout_buffer_pct": 0.0005, "max_or_width_pct": 0.01, "max_entry_time": "11:30", "require_daily_regime": True, "use_volume_pressure": True, "volume_pressure_threshold": 0.0, "qqe_state_mode": "off", "use_swingarm_exit": False, "swing_lookback": 10, "atr_length": 14, "atr_multiplier": 2.5, "neutral_exit_bars": 3, "exit_on_or_failure": True, "end_of_day_exit": True, "allow_overnight": False},
        parameter_grid={"breakout_buffer_pct": [0.0005], "max_or_width_pct": [0.01], "volume_pressure_threshold": [0.0, 1.0]},
        timeframes=("15m", "5m"),
    ),
    "intraday_qqe_hma": SpyStrategyPreset(
        key="intraday_qqe_hma",
        label="Intraday QQE/HMA State",
        strategy_name="Intraday QQE/HMA State",
        description="Uses QQE long/short/neutral states as a long-only intraday SPY filter with HMA and optional SwingArm exits.",
        parameters={"hma_length": 21, "rsi_length": 14, "rsi_smoothing": 5, "qqe_factor": 4.236, "qqe_atr_smoothing": 14, "neutral_band": 2.5, "require_daily_regime": True, "require_hma_slope": True, "volume_pressure_threshold": 0.0, "use_swingarm_exit": True, "swing_lookback": 10, "atr_length": 14, "atr_multiplier": 2.5, "neutral_exit_bars": 3, "exit_on_hma_break": True, "end_of_day_exit": True, "allow_overnight": False},
        parameter_grid={"hma_length": [18, 21], "qqe_factor": [3.5, 4.236], "volume_pressure_threshold": [0.0, 1.0]},
        timeframes=("15m", "5m"),
        experimental=True,
    ),
    "swingarm_trend": SpyStrategyPreset(
        key="swingarm_trend",
        label="SwingArm Trend",
        strategy_name="SwingArm Trend",
        description="Long-only SPY intraday trend entry using an ATR-adjusted SwingArm line.",
        parameters={"atr_length": 14, "swing_lookback": 10, "atr_multiplier": 2.5, "require_daily_regime": True, "end_of_day_exit": True, "allow_overnight": False},
        parameter_grid={"swing_lookback": [8, 10, 12], "atr_multiplier": [2.0, 2.5, 3.0]},
        timeframes=("15m", "5m"),
    ),
}

SPY_EXIT_STRUCTURES: dict[str, SpyExitStructure] = {
    "signal_exit_only": SpyExitStructure(
        key="signal_exit_only",
        label="Signal exit only",
        description="Use only the strategy’s native exit signal.",
        enabled=True,
        default_params={},
    ),
    "fixed_stop_loss": SpyExitStructure(
        key="fixed_stop_loss",
        label="Fixed stop loss",
        description="Apply a fixed percentage stop and keep the strategy’s native exits.",
        enabled=True,
        default_params={"stop_loss_pct": 0.08},
    ),
    "fixed_take_profit": SpyExitStructure(
        key="fixed_take_profit",
        label="Fixed take profit",
        description="Apply a fixed percentage take-profit level.",
        enabled=True,
        default_params={"take_profit_pct": 0.12},
    ),
    "oco_bracket": SpyExitStructure(
        key="oco_bracket",
        label="OCO bracket",
        description="Apply both a stop loss and a take-profit level.",
        enabled=True,
        default_params={"stop_loss_pct": 0.08, "take_profit_pct": 0.15},
    ),
    "trailing_stop": SpyExitStructure(
        key="trailing_stop",
        label="Trailing stop",
        description="Use a trailing stop without a fixed target.",
        enabled=True,
        default_params={"trailing_stop_pct": 0.10},
    ),
    "stop_loss_plus_trailing_stop": SpyExitStructure(
        key="stop_loss_plus_trailing_stop",
        label="Stop loss plus trailing stop",
        description="Combine an initial stop with a trailing stop overlay.",
        enabled=True,
        default_params={"stop_loss_pct": 0.08, "trailing_stop_pct": 0.10},
    ),
    "partial_take_profit_plus_trailing_stop": SpyExitStructure(
        key="partial_take_profit_plus_trailing_stop",
        label="Partial take profit plus trailing stop",
        description="Planned. Partial exits are not implemented in the daily engine yet.",
        enabled=False,
        default_params={},
    ),
    "time_stop": SpyExitStructure(
        key="time_stop",
        label="Time stop",
        description="Exit after a maximum holding period even if no other exit has fired.",
        enabled=True,
        default_params={"max_holding_days": 20},
    ),
}


def list_spy_strategy_presets(timeframe: str = "1d") -> list[SpyStrategyPreset]:
    """Return the supported SPY-only strategy presets."""
    return [preset for preset in SPY_PRESETS.values() if timeframe in preset.timeframes]


def get_spy_strategy_preset(preset_key: str) -> SpyStrategyPreset:
    """Load one SPY-only preset by key."""
    return SPY_PRESETS[preset_key]


def list_spy_exit_structures() -> list[SpyExitStructure]:
    """Return supported SPY exit structures, including planned-but-disabled ones."""
    return list(SPY_EXIT_STRUCTURES.values())


def get_spy_exit_structure(exit_key: str) -> SpyExitStructure:
    """Load one SPY exit structure by key."""
    return SPY_EXIT_STRUCTURES[exit_key]


def build_spy_workbench_config(
    *,
    preset_key: str,
    entry_parameters: dict[str, Any],
    timeframe: str,
    exit_structure_key: str,
    exit_parameters: dict[str, Any],
    start_date: str,
    end_date: str,
    price_mode: str,
    initial_capital: float,
    position_sizing_method: str,
    position_size_value: float,
    max_positions: int,
    slippage_pct: float,
    commission_per_trade: float,
) -> SpyWorkbenchConfig:
    """Freeze one SPY-only entry-plus-exit research configuration."""
    preset = get_spy_strategy_preset(preset_key)
    exit_structure = get_spy_exit_structure(exit_structure_key)
    return SpyWorkbenchConfig(
        preset_key=preset_key,
        entry_label=preset.label,
        entry_parameters=dict(entry_parameters),
        timeframe=timeframe,
        exit_structure_key=exit_structure_key,
        exit_structure_label=exit_structure.label,
        exit_parameters=dict(exit_parameters),
        start_date=start_date,
        end_date=end_date,
        price_mode=price_mode,
        initial_capital=float(initial_capital),
        position_sizing_method=position_sizing_method,
        position_size_value=float(position_size_value),
        max_positions=int(max_positions),
        slippage_pct=float(slippage_pct),
        commission_per_trade=float(commission_per_trade),
    )


def build_spy_strategy(preset_key: str, custom_parameters: dict[str, Any] | None = None):
    """Instantiate a strategy from a SPY preset and optional parameter overrides."""
    preset = get_spy_strategy_preset(preset_key)
    params = {**preset.parameters, **(custom_parameters or {})}
    if preset_key == "trend_filter_200":
        return TrendFilterStrategy(**params)
    if preset_key == "moving_average_50_200":
        return MovingAverageCrossStrategy(**params)
    if preset_key == "rsi_pullback_uptrend":
        return RSIMeanReversionStrategy(**params)
    if preset_key == "breakout_50_20":
        return BreakoutStrategy(**params)
    if preset_key == "qqe_hma_daily":
        return QQEHMAStrategy(**params)
    if preset_key == "intraday_pullback":
        return IntradayPullbackStrategy(**params)
    if preset_key == "intraday_breakout":
        return IntradayBreakoutStrategy(**params)
    if preset_key == "opening_range_breakout":
        return OpeningRangeBreakoutStrategy(**params)
    if preset_key == "intraday_qqe_hma":
        return IntradayQQEHMAStateStrategy(**params)
    if preset_key == "swingarm_trend":
        return SwingArmTrendStrategy(**params)
    raise ValueError(f"Unsupported SPY preset: {preset_key}")


def build_spy_backtest_config(workbench: SpyWorkbenchConfig) -> BacktestConfig:
    """Translate the frozen workbench config into the engine config."""
    exit_params = workbench.exit_parameters
    return BacktestConfig(
        initial_capital=workbench.initial_capital,
        slippage_pct=workbench.slippage_pct,
        commission_per_trade=workbench.commission_per_trade,
        position_sizing_method=workbench.position_sizing_method,
        position_size_value=workbench.position_size_value,
        max_positions=workbench.max_positions,
        stop_loss_pct=float(exit_params.get("stop_loss_pct") or 0.0) or None,
        take_profit_pct=float(exit_params.get("take_profit_pct") or 0.0) or None,
        trailing_stop_pct=float(exit_params.get("trailing_stop_pct") or 0.0) or None,
        price_mode=workbench.price_mode,
        timeframe=workbench.timeframe,
        end_of_day_exit=bool(workbench.entry_parameters.get("end_of_day_exit", False) if is_intraday_timeframe(workbench.timeframe) else False),
        allow_overnight=bool(workbench.entry_parameters.get("allow_overnight", True) if is_intraday_timeframe(workbench.timeframe) else True),
    )


def apply_spy_exit_structure(strategy, workbench: SpyWorkbenchConfig):
    """Apply exit-structure overrides to a strategy instance before running the engine."""
    max_holding_days = workbench.exit_parameters.get("max_holding_days")
    if max_holding_days is not None:
        setattr(strategy, "max_holding_days", int(max_holding_days))
    return strategy


def prepare_spy_timeframe_bars(*, primary_bars: pd.DataFrame, timeframe: str, daily_bars: pd.DataFrame | None = None, regime_sma_window: int = 200) -> pd.DataFrame:
    """Prepare a timeframe-specific SPY frame, aligning completed daily regime data to intraday bars."""
    frame = primary_bars.copy().sort_values("timestamp").reset_index(drop=True)
    if timeframe == "1d":
        return frame
    if daily_bars is None or daily_bars.empty:
        frame["daily_regime_bull"] = False
        frame["daily_regime_source_date"] = pd.NaT
        return frame
    daily = daily_bars.copy().sort_values("timestamp").reset_index(drop=True)
    daily["daily_trend_sma"] = daily["close"].rolling(regime_sma_window).mean()
    daily["daily_regime_bull_raw"] = daily["close"] > daily["daily_trend_sma"]
    daily["regime_for_next_session"] = daily["daily_regime_bull_raw"].shift(1).fillna(False)
    daily["regime_source_date"] = pd.to_datetime(daily["session_date"]).shift(1)
    merge = daily[["session_date", "regime_for_next_session", "regime_source_date"]].copy()
    merge["session_date"] = pd.to_datetime(merge["session_date"]).dt.date
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date
    frame = frame.merge(merge, on="session_date", how="left")
    frame["daily_regime_bull"] = frame["regime_for_next_session"].fillna(False).astype(bool)
    frame["daily_regime_source_date"] = frame["regime_source_date"]
    return frame.drop(columns=["regime_for_next_session", "regime_source_date"])


def spy_strategy_summary(metrics: dict[str, float | int], *, benchmark_sharpe: float = 0.0) -> dict[str, float | int]:
    """Compute SPY-vs-buy-and-hold summary metrics for the current strategy run."""
    benchmark_cagr = float(metrics.get("Benchmark CAGR", 0.0) or 0.0)
    benchmark_drawdown = float(metrics.get("Benchmark Max Drawdown", 0.0) or 0.0)
    benchmark_total = float(metrics.get("Benchmark Total Return", 0.0) or 0.0)
    benchmark_calmar = benchmark_cagr / abs(benchmark_drawdown) if benchmark_drawdown not in (0.0, -0.0) else 0.0
    return {
        "Strategy Total Return": float(metrics.get("Total Return", 0.0) or 0.0),
        "Buy-and-Hold SPY Total Return": benchmark_total,
        "Strategy CAGR": float(metrics.get("CAGR", 0.0) or 0.0),
        "Buy-and-Hold SPY CAGR": benchmark_cagr,
        "Strategy Max Drawdown": float(metrics.get("Max Drawdown", 0.0) or 0.0),
        "Buy-and-Hold SPY Max Drawdown": benchmark_drawdown,
        "Strategy Sharpe": float(metrics.get("Sharpe Ratio", 0.0) or 0.0),
        "Buy-and-Hold SPY Sharpe": benchmark_sharpe,
        "Strategy Calmar": float(metrics.get("Calmar Ratio", 0.0) or 0.0),
        "Buy-and-Hold SPY Calmar": benchmark_calmar,
        "Number of Trades": int(metrics.get("Number of Trades", 0) or 0),
        "Win Rate": float(metrics.get("Win Rate", 0.0) or 0.0),
        "Profit Factor": float(metrics.get("Profit Factor", 0.0) or 0.0),
        "Average R Multiple": float(metrics.get("Average R Multiple", 0.0) or 0.0),
        "Exposure %": float(metrics.get("Exposure %", 0.0) or 0.0),
        "Time in Market": float(metrics.get("Exposure %", 0.0) or 0.0),
        "Excess CAGR vs SPY": float(metrics.get("Excess CAGR", 0.0) or 0.0),
        "Drawdown Improvement vs SPY": abs(benchmark_drawdown) - abs(float(metrics.get("Max Drawdown", 0.0) or 0.0)),
    }


def spy_summary_commentary(summary: dict[str, float | int]) -> str:
    """Translate the SPY summary into a plain-English interpretation."""
    if int(summary.get("Number of Trades", 0) or 0) < 5:
        return "Trade count is too low to trust."
    if float(summary.get("Excess CAGR vs SPY", 0.0) or 0.0) > 0 and float(summary.get("Drawdown Improvement vs SPY", 0.0) or 0.0) > 0:
        return "This strategy beat buy-and-hold SPY with lower drawdown."
    if float(summary.get("Drawdown Improvement vs SPY", 0.0) or 0.0) > 0 and float(summary.get("Excess CAGR vs SPY", 0.0) or 0.0) <= 0:
        return "This strategy reduced drawdown but underperformed SPY."
    return "This strategy did not improve risk-adjusted results."


def average_r_multiple(trade_log: pd.DataFrame, exit_parameters: dict[str, Any]) -> float:
    """Estimate average R multiple using the configured initial stop-style risk."""
    if trade_log.empty:
        return 0.0
    risk_pct = float(exit_parameters.get("stop_loss_pct") or exit_parameters.get("trailing_stop_pct") or 0.0)
    if risk_pct <= 0:
        return 0.0
    return float((trade_log["return_pct"] / risk_pct).replace([pd.NA, pd.NaT], 0.0).mean())


def run_spy_parameter_stability(
    *,
    engine,
    config,
    data_by_symbol: dict[str, pd.DataFrame],
    preset_key: str,
    benchmark_symbol: str = "SPY",
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    """Run the limited SPY preset sweep and summarize how stable it looks versus buy-and-hold SPY."""
    preset = get_spy_strategy_preset(preset_key)
    sweep_id, results = run_parameter_sweep(
        engine,
        lambda params: build_spy_strategy(preset_key, params),
        data_by_symbol,
        config,
        preset.parameter_grid,
        benchmark_symbol,
        sort_metric="CAGR",
        strategy_name=preset.strategy_name,
        notes=f"SPY Workbench preset: {preset.label}",
        tags="spy-only,spy-strategy-lab",
        sweep_context={"preset_key": preset.key},
    )
    summary = parameter_stability_summary(results, drawdown_threshold=-0.25) if not results.empty else {}
    if not results.empty:
        summary = {
            **summary,
            "best_cagr": float(results["CAGR"].max()),
            "median_cagr": float(results["CAGR"].median()),
            "worst_cagr": float(results["CAGR"].min()),
            "percent_beating_spy": float((results["Excess CAGR"] > 0).mean()),
        }
    return sweep_id, results, summary


def run_spy_exit_comparison(
    *,
    engine,
    data_by_symbol: dict[str, pd.DataFrame],
    workbench: SpyWorkbenchConfig,
    exit_structure_keys: list[str],
    benchmark_symbol: str = "SPY",
) -> pd.DataFrame:
    """Compare multiple exit structures while holding the SPY entry strategy constant."""
    rows: list[dict[str, Any]] = []
    for exit_key in exit_structure_keys:
        exit_structure = get_spy_exit_structure(exit_key)
        if not exit_structure.enabled:
            rows.append(
                {
                    "Exit Structure": exit_structure.label,
                    "Exit Parameters": exit_structure.default_params,
                    "Status": "Planned",
                    "CAGR": 0.0,
                    "Excess CAGR": 0.0,
                    "Max Drawdown": 0.0,
                    "Drawdown Improvement": 0.0,
                    "Sharpe": 0.0,
                    "Calmar": 0.0,
                    "Number of Trades": 0,
                    "Win Rate": 0.0,
                    "Profit Factor": 0.0,
                    "Average R Multiple": 0.0,
                    "Candidate Label": "Planned only",
                }
            )
            continue
        candidate_workbench = SpyWorkbenchConfig(
            **{**workbench.__dict__, "exit_structure_key": exit_key, "exit_structure_label": exit_structure.label, "exit_parameters": dict(exit_structure.default_params)}
        )
        strategy = apply_spy_exit_structure(build_spy_strategy(candidate_workbench.preset_key, candidate_workbench.entry_parameters), candidate_workbench)
        config = build_spy_backtest_config(candidate_workbench)
        result = engine.run(data_by_symbol=data_by_symbol, strategy=strategy, config=config, benchmark_symbol=benchmark_symbol)
        avg_r = average_r_multiple(result.trade_log, candidate_workbench.exit_parameters)
        candidate_label = "Possible SPY candidate"
        if int(result.metrics.get("Number of Trades", 0) or 0) < 5:
            candidate_label = "Too few trades"
        elif float(result.metrics.get("Excess CAGR", 0.0) or 0.0) <= 0 and abs(float(result.metrics.get("Max Drawdown", 0.0) or 0.0)) >= abs(float(result.metrics.get("Benchmark Max Drawdown", 0.0) or 0.0)):
            candidate_label = "Not ready"
        rows.append(
            {
                "Exit Structure": exit_structure.label,
                "Exit Parameters": dict(exit_structure.default_params),
                "Status": "Implemented",
                "CAGR": float(result.metrics.get("CAGR", 0.0) or 0.0),
                "Excess CAGR": float(result.metrics.get("Excess CAGR", 0.0) or 0.0),
                "Max Drawdown": float(result.metrics.get("Max Drawdown", 0.0) or 0.0),
                "Drawdown Improvement": abs(float(result.metrics.get("Benchmark Max Drawdown", 0.0) or 0.0)) - abs(float(result.metrics.get("Max Drawdown", 0.0) or 0.0)),
                "Sharpe": float(result.metrics.get("Sharpe Ratio", 0.0) or 0.0),
                "Calmar": float(result.metrics.get("Calmar Ratio", 0.0) or 0.0),
                "Number of Trades": int(result.metrics.get("Number of Trades", 0) or 0),
                "Win Rate": float(result.metrics.get("Win Rate", 0.0) or 0.0),
                "Profit Factor": float(result.metrics.get("Profit Factor", 0.0) or 0.0),
                "Average R Multiple": avg_r,
                "Candidate Label": candidate_label,
            }
        )
    return pd.DataFrame(rows)


def summarize_exit_comparison_results(results: pd.DataFrame) -> list[str]:
    """Return plain-English takeaways from the exit comparison table."""
    if results.empty:
        return ["No exit comparison results are available."]
    implemented = results[results["Status"] == "Implemented"]
    if implemented.empty:
        return ["No implemented exit structures were selected."]
    summaries = [
        f"Best CAGR: {implemented.sort_values('CAGR', ascending=False).iloc[0]['Exit Structure']}.",
        f"Best drawdown reduction: {implemented.sort_values('Drawdown Improvement', ascending=False).iloc[0]['Exit Structure']}.",
        f"Best risk-adjusted return: {implemented.sort_values('Calmar', ascending=False).iloc[0]['Exit Structure']}.",
    ]
    low_trade = implemented[implemented["Number of Trades"] < 5]
    if not low_trade.empty:
        summaries.append(f"Too few trades to trust: {', '.join(low_trade['Exit Structure'].tolist())}.")
    return summaries


def run_spy_robustness_checks(
    *,
    engine,
    strategy,
    config,
    data_by_symbol: dict[str, pd.DataFrame],
    benchmark_symbol: str = "SPY",
    parameter_stability_summary_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a compact robustness stack for the SPY lab."""
    train_data, test_data = split_data_by_percentage(data_by_symbol, 0.7)
    train_test = run_train_test_analysis(engine, strategy, config, train_data, test_data, benchmark_symbol)
    walk_forward_id, walk_folds, walk_summary = run_walk_forward_analysis(
        engine,
        strategy,
        config,
        data_by_symbol,
        benchmark_symbol,
        24,
        6,
        6,
        0,
        0,
    )
    slippage_results = pd.DataFrame()
    if data_by_symbol:
        from trading_lab.backtest.qualification import run_slippage_sensitivity

        slippage_results = run_slippage_sensitivity(
            engine,
            {"SPY Strategy": lambda: strategy},
            data_by_symbol,
            config,
            benchmark_symbol,
            [0.0, 0.0002, 0.0005, 0.0010, 0.0025],
        )
    return {
        "train_test": train_test,
        "walk_forward_id": walk_forward_id,
        "walk_folds": walk_folds,
        "walk_summary": walk_summary,
        "slippage_results": slippage_results,
        "slippage_warnings": summarize_slippage_warnings(slippage_results),
        "parameter_stability": parameter_stability_summary_payload or {},
    }


def build_spy_robustness_checklist(
    *,
    metrics: dict[str, float | int],
    concentration: dict[str, Any],
    robustness_payload: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Create a simplified SPY robustness checklist and final candidate label."""
    benchmark_calmar = (
        float(metrics.get("Benchmark CAGR", 0.0) or 0.0) / abs(float(metrics.get("Benchmark Max Drawdown", 0.0) or 0.0))
        if float(metrics.get("Benchmark Max Drawdown", 0.0) or 0.0) != 0.0
        else 0.0
    )
    train_test = robustness_payload.get("train_test") or {}
    walk_summary = robustness_payload.get("walk_summary") or {}
    parameter_stability = robustness_payload.get("parameter_stability") or {}
    slippage_warnings = robustness_payload.get("slippage_warnings") or []

    checks = [
        _check_status(float(metrics.get("Excess CAGR", 0.0) or 0.0) > 0, float(metrics.get("Excess CAGR", 0.0) or 0.0) > -0.02, "Beats SPY on CAGR?"),
        _check_status(abs(float(metrics.get("Max Drawdown", 0.0) or 0.0)) < abs(float(metrics.get("Benchmark Max Drawdown", 0.0) or 0.0)), abs(float(metrics.get("Max Drawdown", 0.0) or 0.0)) <= abs(float(metrics.get("Benchmark Max Drawdown", 0.0) or 0.0)) * 1.1, "Beats SPY on max drawdown?"),
        _check_status(float(metrics.get("Calmar Ratio", 0.0) or 0.0) > benchmark_calmar, float(metrics.get("Calmar Ratio", 0.0) or 0.0) >= benchmark_calmar * 0.9, "Beats SPY on Calmar?"),
        _check_status(int(metrics.get("Number of Trades", 0) or 0) >= 20, int(metrics.get("Number of Trades", 0) or 0) >= 10, "Trade count sufficient?"),
        _check_status(float(train_test.get("degradation", {}).get("CAGR", 0.0) or 0.0) >= -0.05, float(train_test.get("degradation", {}).get("CAGR", 0.0) or 0.0) >= -0.10, "Train/test acceptable?"),
        _check_status(float(walk_summary.get("profitable_test_fold_pct", 0.0) or 0.0) >= 0.5 and float(walk_summary.get("consistency_score", 0.0) or 0.0) >= 0.5, float(walk_summary.get("profitable_test_fold_pct", 0.0) or 0.0) >= 0.4, "Walk-forward acceptable?"),
        _check_status(float(parameter_stability.get("positive_return_pct", 0.0) or 0.0) >= 0.5 and float(parameter_stability.get("percent_beating_spy", 0.0) or 0.0) >= 0.4, float(parameter_stability.get("positive_return_pct", 0.0) or 0.0) >= 0.35, "Parameter stability acceptable?"),
        _check_status(float(concentration.get("best_trade_profit_share", 0.0) or 0.0) <= 0.5 and float(concentration.get("top_5_profit_share", 0.0) or 0.0) <= 0.75, float(concentration.get("top_5_profit_share", 0.0) or 0.0) <= 0.85, "No extreme profit concentration?"),
        _check_status(not slippage_warnings, True, "Slippage sensitivity acceptable?"),
    ]
    frame = pd.DataFrame(checks)
    pass_count = int((frame["result"] == "Pass").sum())
    fail_count = int((frame["result"] == "Fail").sum())
    if fail_count == 0 and pass_count >= 7:
        final_label = "Strong SPY candidate"
    elif fail_count <= 2 and pass_count >= 5:
        final_label = "Possible SPY candidate"
    else:
        final_label = "Not ready"
    return frame, final_label


def spy_daily_signal_status(
    *,
    bars: pd.DataFrame,
    strategy,
    latest_close: float,
    data_freshness_status: str,
    pending_orders: pd.DataFrame,
    open_positions: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize the latest SPY-only signal state for the strategy lab."""
    if bars.empty:
        return {
            "current_signal": "no_signal",
            "last_signal_date": None,
            "position_state": "no_data",
            "pending_order": False,
            "open_position": False,
            "latest_close": 0.0,
            "data_freshness_status": data_freshness_status,
            "next_expected_action": "no action",
        }
    signal_frame = strategy.generate_signals(bars.copy().sort_values("timestamp").reset_index(drop=True))
    position_active = _position_active_series(signal_frame)
    signal_frame["position_active"] = position_active
    signal_type = determine_signal_type(signal_frame)
    latest = signal_frame.iloc[-1]
    signal_mask = signal_frame["entry_signal"].astype(bool) | signal_frame["exit_signal"].astype(bool)
    last_signal_date = pd.Timestamp(signal_frame.loc[signal_mask, "timestamp"].iloc[-1]) if signal_mask.any() else None
    has_pending = bool((not pending_orders.empty) and pending_orders["status"].eq("pending").any())
    has_open = bool((not open_positions.empty) and open_positions["status"].eq("open").any())
    if has_pending:
        next_action = "pending entry next open"
    elif has_open and bool(latest.get("exit_signal", False)):
        next_action = "exit next open"
    elif has_open:
        next_action = "hold"
    elif signal_type == "new_buy_signal":
        next_action = "pending entry next open"
    elif signal_type == "exit_signal":
        next_action = "exit next open"
    elif signal_type == "active_long_signal":
        next_action = "hold"
    elif data_freshness_status == "stale":
        next_action = "wait"
    else:
        next_action = "no action"
    return {
        "current_signal": signal_type,
        "last_signal_date": last_signal_date,
        "position_state": "long" if bool(latest.get("position_active", False)) else "flat",
        "pending_order": has_pending,
        "open_position": has_open,
        "latest_close": float(latest_close),
        "data_freshness_status": data_freshness_status,
        "next_expected_action": next_action,
    }


def _position_active_series(frame: pd.DataFrame) -> list[bool]:
    state = False
    states: list[bool] = []
    for _, row in frame.iterrows():
        if bool(row.get("exit_signal", False)):
            state = False
            states.append(False)
            continue
        if bool(row.get("entry_signal", False)):
            state = True
        states.append(state)
    return states


def _check_status(pass_condition: bool, caution_condition: bool, check_name: str) -> dict[str, str]:
    if pass_condition:
        result = "Pass"
    elif caution_condition:
        result = "Caution"
    else:
        result = "Fail"
    return {"check": check_name, "result": result}


def summarize_profit_concentration(trade_log: pd.DataFrame) -> dict[str, Any]:
    """Expose profit concentration through the SPY workflow helper."""
    return profit_concentration_analysis(trade_log)


def generate_approved_spy_entry_presets(timeframe: str = "1d") -> list[SpySearchEntryPreset]:
    """Return the controlled SPY entry presets used by automated search."""
    if timeframe == "1d":
        return [
            SpySearchEntryPreset("trend_150", "trend_filter_200", "SPY 200-Day Trend Filter", "Trend filter 150", {"sma_length": 150}, "Trend filter with a 150-day SMA.", 1, timeframe="1d"),
            SpySearchEntryPreset("trend_200", "trend_filter_200", "SPY 200-Day Trend Filter", "Trend filter 200", {"sma_length": 200}, "Trend filter with a 200-day SMA.", 1, timeframe="1d"),
            SpySearchEntryPreset("trend_250", "trend_filter_200", "SPY 200-Day Trend Filter", "Trend filter 250", {"sma_length": 250}, "Trend filter with a 250-day SMA.", 1, timeframe="1d"),
            SpySearchEntryPreset("ma_20_100", "moving_average_50_200", "Moving Average Crossover", "MA 20/100", {"fast_window": 20, "slow_window": 100}, "Faster crossover for earlier entries.", 2, timeframe="1d"),
            SpySearchEntryPreset("ma_50_150", "moving_average_50_200", "Moving Average Crossover", "MA 50/150", {"fast_window": 50, "slow_window": 150}, "Intermediate trend-following crossover.", 2, timeframe="1d"),
            SpySearchEntryPreset("ma_50_200", "moving_average_50_200", "Moving Average Crossover", "MA 50/200", {"fast_window": 50, "slow_window": 200}, "Classic long-term crossover.", 2, timeframe="1d"),
            SpySearchEntryPreset("rsi_30_50_10", "rsi_pullback_uptrend", "RSI Mean Reversion", "RSI 30/50 hold 10", {"rsi_length": 14, "buy_threshold": 30.0, "sell_threshold": 50.0, "max_holding_days": 10, "trend_sma_window": 200}, "Pullback buy with tight exit and short holding period.", 3, timeframe="1d"),
            SpySearchEntryPreset("rsi_35_55_20", "rsi_pullback_uptrend", "RSI Mean Reversion", "RSI 35/55 hold 20", {"rsi_length": 14, "buy_threshold": 35.0, "sell_threshold": 55.0, "max_holding_days": 20, "trend_sma_window": 200}, "Balanced SPY pullback preset.", 3, timeframe="1d"),
            SpySearchEntryPreset("rsi_40_60_30", "rsi_pullback_uptrend", "RSI Mean Reversion", "RSI 40/60 hold 30", {"rsi_length": 14, "buy_threshold": 40.0, "sell_threshold": 60.0, "max_holding_days": 30, "trend_sma_window": 200}, "Looser pullback entry with longer holding period.", 3, timeframe="1d"),
            SpySearchEntryPreset("breakout_20_10", "breakout_50_20", "Daily Breakout", "Breakout 20 / exit 10", {"lookback_window": 20, "exit_lookback_window": 10}, "Shorter breakout and faster signal exit.", 2, timeframe="1d"),
            SpySearchEntryPreset("breakout_50_20", "breakout_50_20", "Daily Breakout", "Breakout 50 / exit 20", {"lookback_window": 50, "exit_lookback_window": 20}, "Balanced breakout preset.", 2, timeframe="1d"),
            SpySearchEntryPreset("breakout_100_50", "breakout_50_20", "Daily Breakout", "Breakout 100 / exit 50", {"lookback_window": 100, "exit_lookback_window": 50}, "Longer-term breakout filter.", 2, timeframe="1d"),
            SpySearchEntryPreset("qqe_hma_conservative", "qqe_hma_daily", "QQE/HMA Daily", "QQE/HMA conservative", {"hma_length": 21, "rsi_length": 14, "rsi_smoothing": 5, "qqe_factor": 4.236, "atr_smoothing": 5, "require_hma_slope": True, "exit_on_hma_break": True, "exit_on_qqe_bearish": True}, "Conservative daily QQE/HMA preset.", 5, timeframe="1d", experimental=True),
            SpySearchEntryPreset("qqe_hma_tight", "qqe_hma_daily", "QQE/HMA Daily", "QQE/HMA tight", {"hma_length": 18, "rsi_length": 12, "rsi_smoothing": 5, "qqe_factor": 3.5, "atr_smoothing": 5, "require_hma_slope": True, "exit_on_hma_break": True, "exit_on_qqe_bearish": True}, "Tighter experimental QQE/HMA preset.", 5, timeframe="1d", experimental=True),
        ]
    if timeframe == "15m":
        return [
            SpySearchEntryPreset("opening_range_basic_15m", "opening_range_breakout", "Opening Range Breakout", "15m opening range basic", {"breakout_buffer_pct": 0.0005, "max_or_width_pct": 0.01, "max_entry_time": "11:30", "require_daily_regime": True, "use_volume_pressure": False, "volume_pressure_threshold": 0.0, "qqe_state_mode": "off", "use_swingarm_exit": False, "swing_lookback": 10, "atr_length": 14, "atr_multiplier": 2.5, "neutral_exit_bars": 3, "exit_on_or_failure": True, "end_of_day_exit": True, "allow_overnight": False}, "30-minute opening-range breakout without extra confirmation.", 2, timeframe="15m"),
            SpySearchEntryPreset("opening_range_pressure_15m", "opening_range_breakout", "Opening Range Breakout", "15m opening range + pressure", {"breakout_buffer_pct": 0.0005, "max_or_width_pct": 0.01, "max_entry_time": "11:30", "require_daily_regime": True, "use_volume_pressure": True, "volume_pressure_threshold": 0.0, "qqe_state_mode": "off", "use_swingarm_exit": False, "swing_lookback": 10, "atr_length": 14, "atr_multiplier": 2.5, "neutral_exit_bars": 3, "exit_on_or_failure": True, "end_of_day_exit": True, "allow_overnight": False}, "Opening-range breakout confirmed by non-negative volume pressure.", 3, timeframe="15m"),
            SpySearchEntryPreset("opening_range_swingarm_15m", "opening_range_breakout", "Opening Range Breakout", "15m opening range + SwingArm", {"breakout_buffer_pct": 0.0005, "max_or_width_pct": 0.01, "max_entry_time": "11:30", "require_daily_regime": True, "use_volume_pressure": True, "volume_pressure_threshold": 0.0, "qqe_state_mode": "long_or_neutral_positive", "use_swingarm_exit": True, "swing_lookback": 10, "atr_length": 14, "atr_multiplier": 2.5, "neutral_exit_bars": 3, "exit_on_or_failure": True, "end_of_day_exit": True, "allow_overnight": False}, "Opening-range breakout with pressure, QQE state, and SwingArm exit.", 4, timeframe="15m"),
            SpySearchEntryPreset("intraday_qqe_hma_15m", "intraday_qqe_hma", "Intraday QQE/HMA State", "15m QQE/HMA + SwingArm", {"hma_length": 21, "rsi_length": 14, "rsi_smoothing": 5, "qqe_factor": 4.236, "qqe_atr_smoothing": 14, "neutral_band": 2.5, "require_daily_regime": True, "require_hma_slope": True, "volume_pressure_threshold": 0.0, "use_swingarm_exit": True, "swing_lookback": 10, "atr_length": 14, "atr_multiplier": 2.5, "neutral_exit_bars": 3, "exit_on_hma_break": True, "end_of_day_exit": True, "allow_overnight": False}, "QQE long-state plus HMA and volume pressure with SwingArm exit.", 4, timeframe="15m", experimental=True),
            SpySearchEntryPreset("swingarm_trend_15m", "swingarm_trend", "SwingArm Trend", "15m SwingArm standalone", {"atr_length": 14, "swing_lookback": 10, "atr_multiplier": 2.5, "require_daily_regime": True, "end_of_day_exit": True, "allow_overnight": False}, "Standalone SwingArm trend-cross strategy.", 2, timeframe="15m"),
        ]
    return [
        SpySearchEntryPreset("opening_range_pressure_5m", "opening_range_breakout", "Opening Range Breakout", "5m opening range experimental", {"breakout_buffer_pct": 0.0005, "max_or_width_pct": 0.01, "max_entry_time": "11:30", "require_daily_regime": True, "use_volume_pressure": True, "volume_pressure_threshold": 0.0, "qqe_state_mode": "off", "use_swingarm_exit": False, "swing_lookback": 10, "atr_length": 14, "atr_multiplier": 2.5, "neutral_exit_bars": 3, "exit_on_or_failure": True, "end_of_day_exit": True, "allow_overnight": False}, "Experimental 5-minute opening-range breakout with pressure confirmation.", 4, timeframe="5m", experimental=True),
        SpySearchEntryPreset("intraday_qqe_hma_5m", "intraday_qqe_hma", "Intraday QQE/HMA State", "5m QQE/HMA experimental", {"hma_length": 18, "rsi_length": 12, "rsi_smoothing": 5, "qqe_factor": 3.5, "qqe_atr_smoothing": 14, "neutral_band": 2.5, "require_daily_regime": True, "require_hma_slope": True, "volume_pressure_threshold": 0.0, "use_swingarm_exit": True, "swing_lookback": 10, "atr_length": 14, "atr_multiplier": 2.5, "neutral_exit_bars": 3, "exit_on_hma_break": True, "end_of_day_exit": True, "allow_overnight": False}, "Experimental 5-minute QQE/HMA state strategy.", 5, timeframe="5m", experimental=True),
    ]


def generate_approved_spy_exit_presets() -> list[SpySearchExitPreset]:
    """Return the implemented SPY exit presets used by automated search."""
    return [
        SpySearchExitPreset("signal_only", "signal_exit_only", "Signal exit only", "Signal exit only", {}, "Use only the strategy's native exit signal."),
        SpySearchExitPreset("fixed_stop_3", "fixed_stop_loss", "Fixed stop loss", "Fixed stop 3%", {"stop_loss_pct": 0.03}, "Fixed 3% stop loss."),
        SpySearchExitPreset("fixed_stop_5", "fixed_stop_loss", "Fixed stop loss", "Fixed stop 5%", {"stop_loss_pct": 0.05}, "Fixed 5% stop loss."),
        SpySearchExitPreset("fixed_stop_7", "fixed_stop_loss", "Fixed stop 7%", "Fixed stop 7%", {"stop_loss_pct": 0.07}, "Fixed 7% stop loss."),
        SpySearchExitPreset("fixed_stop_10", "fixed_stop_loss", "Fixed stop loss", "Fixed stop 10%", {"stop_loss_pct": 0.10}, "Fixed 10% stop loss."),
        SpySearchExitPreset("take_5", "fixed_take_profit", "Fixed take profit", "Take profit 5%", {"take_profit_pct": 0.05}, "Fixed 5% take profit."),
        SpySearchExitPreset("take_8", "fixed_take_profit", "Fixed take profit", "Take profit 8%", {"take_profit_pct": 0.08}, "Fixed 8% take profit."),
        SpySearchExitPreset("take_10", "fixed_take_profit", "Fixed take profit", "Take profit 10%", {"take_profit_pct": 0.10}, "Fixed 10% take profit."),
        SpySearchExitPreset("take_15", "fixed_take_profit", "Fixed take profit", "Take profit 15%", {"take_profit_pct": 0.15}, "Fixed 15% take profit."),
        SpySearchExitPreset("oco_3_6", "oco_bracket", "OCO bracket", "OCO 3% / 6%", {"stop_loss_pct": 0.03, "take_profit_pct": 0.06}, "3% stop with 6% target."),
        SpySearchExitPreset("oco_5_10", "oco_bracket", "OCO bracket", "OCO 5% / 10%", {"stop_loss_pct": 0.05, "take_profit_pct": 0.10}, "5% stop with 10% target."),
        SpySearchExitPreset("oco_7_14", "oco_bracket", "OCO bracket", "OCO 7% / 14%", {"stop_loss_pct": 0.07, "take_profit_pct": 0.14}, "7% stop with 14% target."),
        SpySearchExitPreset("trail_3", "trailing_stop", "Trailing stop", "Trailing 3%", {"trailing_stop_pct": 0.03}, "3% trailing stop."),
        SpySearchExitPreset("trail_5", "trailing_stop", "Trailing stop", "Trailing 5%", {"trailing_stop_pct": 0.05}, "5% trailing stop."),
        SpySearchExitPreset("trail_8", "trailing_stop", "Trailing stop", "Trailing 8%", {"trailing_stop_pct": 0.08}, "8% trailing stop."),
        SpySearchExitPreset("trail_10", "trailing_stop", "Trailing stop", "Trailing 10%", {"trailing_stop_pct": 0.10}, "10% trailing stop."),
        SpySearchExitPreset("stop_trail_5_5", "stop_loss_plus_trailing_stop", "Stop loss plus trailing stop", "Stop 5% + trail 5%", {"stop_loss_pct": 0.05, "trailing_stop_pct": 0.05}, "5% fixed stop plus 5% trailing stop."),
        SpySearchExitPreset("stop_trail_7_8", "stop_loss_plus_trailing_stop", "Stop loss plus trailing stop", "Stop 7% + trail 8%", {"stop_loss_pct": 0.07, "trailing_stop_pct": 0.08}, "7% fixed stop plus 8% trailing stop."),
        SpySearchExitPreset("time_5", "time_stop", "Time stop", "Time stop 5", {"max_holding_days": 5}, "Exit after 5 trading days."),
        SpySearchExitPreset("time_10", "time_stop", "Time stop", "Time stop 10", {"max_holding_days": 10}, "Exit after 10 trading days."),
        SpySearchExitPreset("time_20", "time_stop", "Time stop", "Time stop 20", {"max_holding_days": 20}, "Exit after 20 trading days."),
        SpySearchExitPreset("time_30", "time_stop", "Time stop", "Time stop 30", {"max_holding_days": 30}, "Exit after 30 trading days."),
    ]


def generate_spy_search_combinations(timeframe: str = "1d") -> list[SpySearchCombination]:
    """Generate the controlled SPY strategy-search grid."""
    combinations: list[SpySearchCombination] = []
    for entry_preset, exit_preset in product(generate_approved_spy_entry_presets(timeframe), generate_approved_spy_exit_presets()):
        combinations.append(
            SpySearchCombination(
                combination_id=f"{entry_preset.preset_id}__{exit_preset.exit_preset_id}",
                entry_preset=entry_preset,
                exit_preset=exit_preset,
            )
        )
    return combinations


def describe_spy_search_archetype(result_row: dict[str, Any]) -> str:
    """Describe the entry archetype in plain English for reporting."""
    strategy_name = str(result_row.get("entry_strategy_name", "") or "")
    preset_label = str(result_row.get("entry_preset_label", "") or "")
    lower_name = strategy_name.lower()
    lower_label = preset_label.lower()
    if "opening range" in lower_name or "opening range" in lower_label:
        if "swingarm" in lower_label:
            return "opening-range breakout with SwingArm exit logic"
        if "pressure" in lower_label:
            return "opening-range breakout with volume-pressure confirmation"
        return "opening-range breakout"
    if "qqe" in lower_name:
        return "QQE/HMA state filter"
    if "swingarm" in lower_name:
        return "SwingArm trend-following"
    if "rsi" in lower_name:
        return "RSI pullback mean reversion"
    if "moving average" in lower_name:
        return "moving-average crossover"
    if "trend filter" in lower_name:
        return "long-term trend filter"
    if "breakout" in lower_name:
        return "breakout trend-following"
    return strategy_name or "strategy setup"


def describe_spy_exit_archetype(result_row: dict[str, Any]) -> str:
    """Describe the exit archetype in plain English for reporting."""
    exit_name = str(result_row.get("exit_structure_name", "") or "")
    exit_label = str(result_row.get("exit_preset_label", "") or "")
    lower = f"{exit_name} {exit_label}".lower()
    if "oco" in lower:
        return "OCO bracket"
    if "trailing" in lower and "stop loss plus trailing stop" in lower:
        return "fixed stop plus trailing stop"
    if "trailing" in lower:
        return "trailing stop"
    if "take profit" in lower:
        return "fixed take profit"
    if "fixed stop" in lower or "stop loss" in lower:
        return "fixed stop loss"
    if "time stop" in lower:
        return "time stop"
    return "signal exit"


def build_spy_search_summary_comment(result_row: dict[str, Any]) -> str:
    """Summarize one automated SPY search result in plain English."""
    trades = int(result_row.get("number_of_trades", 0) or 0)
    excess_cagr = float(result_row.get("excess_cagr", 0.0) or 0.0)
    drawdown_improvement = float(result_row.get("drawdown_improvement", 0.0) or 0.0)
    experimental = bool(result_row.get("experimental", False))
    timeframe = str(result_row.get("timeframe", "1d"))
    archetype = describe_spy_search_archetype(result_row)
    exit_archetype = describe_spy_exit_archetype(result_row)
    if timeframe in {"15m", "5m"} and trades < 10:
        return f"This {archetype} with {exit_archetype} uses short intraday history and still has too few trades to trust."
    if trades < 5:
        return f"This {archetype} with {exit_archetype} has strong-looking performance but too few trades to trust."
    if excess_cagr > 0 and drawdown_improvement > 0:
        return f"This {archetype} with {exit_archetype} beat SPY with lower drawdown and sufficient trade count."
    if drawdown_improvement > 0 and excess_cagr <= 0:
        return f"This {archetype} with {exit_archetype} reduced drawdown but underperformed buy-and-hold SPY."
    if experimental:
        return f"This {archetype} with {exit_archetype} is experimental and needs extra caution."
    return f"This {archetype} with {exit_archetype} is simple, but it did not clearly improve on buy-and-hold SPY."


def grade_spy_search_candidate(result_row: dict[str, Any]) -> tuple[str, int]:
    """Grade one SPY search candidate without treating the top CAGR as automatically best."""
    trades = int(result_row.get("number_of_trades", 0) or 0)
    cagr = float(result_row.get("cagr", 0.0) or 0.0)
    excess_cagr = float(result_row.get("excess_cagr", 0.0) or 0.0)
    max_drawdown = abs(float(result_row.get("max_drawdown", 0.0) or 0.0))
    spy_drawdown = abs(float(result_row.get("spy_max_drawdown", 0.0) or 0.0))
    calmar = float(result_row.get("calmar", 0.0) or 0.0)
    profit_factor = float(result_row.get("profit_factor", 0.0) or 0.0)
    avg_r_multiple = float(result_row.get("avg_r_multiple", 0.0) or 0.0)
    exposure_pct = float(result_row.get("exposure_pct", 0.0) or 0.0)
    timeframe = str(result_row.get("timeframe", "1d"))
    red_flag_count = 0
    if trades < 10:
        red_flag_count += 1
    if excess_cagr <= 0:
        red_flag_count += 1
    if cagr <= 0:
        red_flag_count += 1
    if max_drawdown >= spy_drawdown and spy_drawdown > 0:
        red_flag_count += 1
    if calmar <= 0.25:
        red_flag_count += 1
    if profit_factor <= 1.0:
        red_flag_count += 1
    if avg_r_multiple < 0:
        red_flag_count += 1
    if exposure_pct > 0.98:
        red_flag_count += 1
    if bool(result_row.get("experimental", False)):
        red_flag_count += 1
    if timeframe in {"15m", "5m"}:
        red_flag_count += 1
    if trades == 0 or (cagr <= 0 and excess_cagr <= 0 and profit_factor <= 1.0):
        return "Reject", red_flag_count
    if trades >= 20 and cagr > 0 and excess_cagr > 0 and calmar >= 0.5 and profit_factor >= 1.1 and red_flag_count <= 2:
        return "Strong candidate", red_flag_count
    if trades >= 10 and cagr > 0 and (excess_cagr > 0 or max_drawdown < spy_drawdown) and profit_factor >= 1.0 and red_flag_count <= 4:
        return "Possible candidate", red_flag_count
    return "Not ready", red_flag_count


def rank_spy_search_results(results: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Pick the top highlighted candidates from the full SPY search result table."""
    if results.empty:
        return {}
    frame = results.copy()
    frame["complexity_score"] = frame["complexity_score"].fillna(99)
    frame["red_flag_count"] = frame["red_flag_count"].fillna(99)
    frame["candidate_grade_rank"] = frame["candidate_label"].map(
        {"Strong candidate": 3, "Possible candidate": 2, "Not ready": 1, "Reject": 0}
    ).fillna(0)
    frame["overall_score"] = (
        frame["candidate_grade_rank"] * 100
        + frame["excess_cagr"].fillna(0.0) * 100
        + frame["calmar"].fillna(0.0) * 15
        + frame["profit_factor"].fillna(0.0) * 5
        + frame["drawdown_improvement"].fillna(0.0) * 25
        - frame["red_flag_count"] * 6
    )
    frame["suspicious_score"] = (
        frame["cagr"].fillna(0.0) * 100
        + frame["red_flag_count"] * 8
        + frame["experimental"].astype(int) * 8
        + (frame["max_drawdown"].abs().fillna(0.0) * 50)
        - frame["number_of_trades"].fillna(0) * 0.5
        - frame["robustness_score"].fillna(0) * 0.2
    )
    highlights: dict[str, dict[str, Any]] = {}
    eligible = frame[frame["candidate_label"].isin(["Strong candidate", "Possible candidate"])]
    if eligible.empty:
        eligible = frame
    highlights["Best Overall"] = eligible.sort_values(["overall_score", "calmar", "excess_cagr"], ascending=[False, False, False]).iloc[0].to_dict()
    highlights["Best Low Drawdown"] = eligible.sort_values(["max_drawdown", "cagr", "number_of_trades"], ascending=[False, False, False]).iloc[0].to_dict()
    highlights["Best Risk Adjusted"] = eligible.sort_values(["calmar", "sharpe", "sortino", "drawdown_improvement"], ascending=[False, False, False, False]).iloc[0].to_dict()
    highlights["Best Simple Strategy"] = eligible.sort_values(["complexity_score", "red_flag_count", "excess_cagr", "calmar"], ascending=[True, True, False, False]).iloc[0].to_dict()
    highlights["Most Suspicious High Return"] = frame.sort_values(["suspicious_score", "cagr"], ascending=[False, False]).iloc[0].to_dict()
    return highlights


def run_automated_spy_search(
    *,
    engine,
    data_by_symbol: dict[str, pd.DataFrame],
    timeframe: str,
    start_date: str,
    end_date: str,
    price_mode: str,
    initial_capital: float,
    position_sizing_method: str,
    position_sizing_value: float,
    slippage_pct: float,
    commission_per_trade: float,
    daily_bars: pd.DataFrame | None = None,
    persist_backtest_runs: bool = False,
    progress_callback=None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, dict[str, Any]]]:
    """Run the controlled automated SPY search across approved entry and exit permutations.

    By default this runs ephemerally and persists only the search summary rows.
    That avoids writing hundreds of full backtest payloads into DuckDB for one search run.
    """
    combinations = generate_spy_search_combinations(timeframe)
    search_run_id = str(uuid4())
    rows: list[dict[str, Any]] = []
    total = len(combinations)
    search_engine = engine if persist_backtest_runs else BacktestEngine(database=None)
    primary_bars = prepare_spy_timeframe_bars(primary_bars=data_by_symbol["SPY"], timeframe=timeframe, daily_bars=daily_bars)
    for idx, combination in enumerate(combinations, start=1):
        workbench = build_spy_workbench_config(
            preset_key=combination.entry_preset.preset_key,
            entry_parameters=combination.entry_preset.parameters,
            timeframe=timeframe,
            exit_structure_key=combination.exit_preset.exit_structure_key,
            exit_parameters=combination.exit_preset.parameters,
            start_date=start_date,
            end_date=end_date,
            price_mode=price_mode,
            initial_capital=initial_capital,
            position_sizing_method=position_sizing_method,
            position_size_value=position_sizing_value,
            max_positions=1,
            slippage_pct=slippage_pct,
            commission_per_trade=commission_per_trade,
        )
        strategy = apply_spy_exit_structure(build_spy_strategy(workbench.preset_key, workbench.entry_parameters), workbench)
        config = build_spy_backtest_config(workbench)
        result = search_engine.run(data_by_symbol={"SPY": primary_bars}, strategy=strategy, config=config, benchmark_symbol="SPY")
        concentration = summarize_profit_concentration(result.trade_log)
        robustness = compute_robustness_score(result.metrics, concentration=concentration)
        avg_r_multiple = average_r_multiple(result.trade_log, workbench.exit_parameters)
        benchmark_sharpe = 0.0
        if not result.benchmark_curve.empty:
            benchmark_equity = result.benchmark_curve.rename(columns={"benchmark_equity": "equity"})
            from trading_lab.backtest.metrics import calculate_sharpe_ratio

            benchmark_sharpe = calculate_sharpe_ratio(benchmark_equity)
        summary = spy_strategy_summary(result.metrics, benchmark_sharpe=benchmark_sharpe)
        row = {
            "result_id": str(uuid4()),
            "search_run_id": search_run_id,
            "timeframe": timeframe,
            "entry_strategy_name": combination.entry_preset.entry_strategy_name,
            "entry_parameters_json": combination.entry_preset.parameters,
            "entry_preset_id": combination.entry_preset.preset_id,
            "entry_preset_label": combination.entry_preset.label,
            "strategy_archetype": describe_spy_search_archetype(
                {
                    "entry_strategy_name": combination.entry_preset.entry_strategy_name,
                    "entry_preset_label": combination.entry_preset.label,
                }
            ),
            "exit_structure_key": combination.exit_preset.exit_structure_key,
            "exit_structure_name": combination.exit_preset.exit_structure_name,
            "exit_parameters_json": combination.exit_preset.parameters,
            "exit_preset_id": combination.exit_preset.exit_preset_id,
            "exit_preset_label": combination.exit_preset.label,
            "exit_archetype": describe_spy_exit_archetype(
                {
                    "exit_structure_name": combination.exit_preset.exit_structure_name,
                    "exit_preset_label": combination.exit_preset.label,
                }
            ),
            "backtest_run_id": result.run_id if persist_backtest_runs else None,
            "total_return": float(result.metrics.get("Total Return", 0.0) or 0.0),
            "cagr": float(result.metrics.get("CAGR", 0.0) or 0.0),
            "spy_cagr": float(summary["Buy-and-Hold SPY CAGR"]),
            "excess_cagr": float(result.metrics.get("Excess CAGR", 0.0) or 0.0),
            "max_drawdown": float(result.metrics.get("Max Drawdown", 0.0) or 0.0),
            "spy_max_drawdown": float(result.metrics.get("Benchmark Max Drawdown", 0.0) or 0.0),
            "drawdown_improvement": float(summary["Drawdown Improvement vs SPY"]),
            "sharpe": float(result.metrics.get("Sharpe Ratio", 0.0) or 0.0),
            "sortino": float(result.metrics.get("Sortino Ratio", 0.0) or 0.0),
            "calmar": float(result.metrics.get("Calmar Ratio", 0.0) or 0.0),
            "number_of_trades": int(result.metrics.get("Number of Trades", 0) or 0),
            "win_rate": float(result.metrics.get("Win Rate", 0.0) or 0.0),
            "profit_factor": float(result.metrics.get("Profit Factor", 0.0) or 0.0),
            "avg_trade_return": float(result.metrics.get("Average Trade Return", 0.0) or 0.0),
            "avg_r_multiple": float(avg_r_multiple),
            "exposure_pct": float(result.metrics.get("Exposure %", 0.0) or 0.0),
            "robustness_score": int(robustness.score),
            "experimental": bool(combination.entry_preset.experimental),
            "complexity_score": int(combination.entry_preset.complexity_score),
            "promoted_active_strategy_id": None,
            "created_at": datetime.now(UTC).replace(tzinfo=None),
        }
        candidate_label, red_flag_count = grade_spy_search_candidate(row)
        row["candidate_label"] = candidate_label
        row["red_flag_count"] = red_flag_count + len(robustness.red_flags)
        row["summary_comment"] = build_spy_search_summary_comment(row)
        rows.append(row)
        if progress_callback is not None:
            progress_callback(idx, total, row)
    results = pd.DataFrame(rows)
    highlights = rank_spy_search_results(results)
    if not results.empty:
        ranking_map = {key: value["result_id"] for key, value in highlights.items()}
        results["ranking_category"] = results["result_id"].map({value: key for key, value in ranking_map.items()}).fillna("")
    payload = {
        "search_run_id": search_run_id,
        "created_at": datetime.now(UTC).replace(tzinfo=None),
        "start_date": start_date,
        "end_date": end_date,
        "timeframe": timeframe,
        "price_mode": price_mode,
        "initial_capital": float(initial_capital),
        "slippage_pct": float(slippage_pct),
        "commission_per_trade": float(commission_per_trade),
        "position_sizing_method": position_sizing_method,
        "position_sizing_value": float(position_sizing_value),
        "benchmark_symbol": "SPY",
        "total_combinations_tested": total,
        "notes": "",
        "tags": "spy-only,automated-search",
    }
    return payload, results, highlights
