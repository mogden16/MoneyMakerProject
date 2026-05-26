from __future__ import annotations

from collections import OrderedDict
from typing import Callable

from trading_lab.pybroker_lab.config import PyBrokerLabConfig, PyBrokerStrategyDefinition
from trading_lab.pybroker_lab.fixed_strategy_utils import StrategyTemplate
from trading_lab.pybroker_lab.strategies.blackflag_fts_hma import BLACKFLAG_FTS_HMA_SETTINGS, build_strategy as build_blackflag_fts_hma, generate_signal_frame as generate_blackflag_fts_hma_signals
from trading_lab.pybroker_lab.strategies.blackflag_fts_qqe_momo import BLACKFLAG_FTS_QQE_MOMO_SETTINGS, build_strategy as build_blackflag_fts_qqe_momo, generate_signal_frame as generate_blackflag_fts_qqe_momo_signals
from trading_lab.pybroker_lab.strategies.ema_compression_volume_breakout import (
    EMA_COMPRESSION_VOLUME_BREAKOUT_SETTINGS,
    build_strategy as build_ema_compression_volume_breakout,
    generate_signal_frame as generate_ema_compression_volume_breakout_signals,
)
from trading_lab.pybroker_lab.strategies.legacy_mtf_qqe_rsi_momentum import (
    LEGACY_MTF_QQE_RSI_SETTINGS,
    build_strategy as build_legacy_mtf_qqe_rsi_momentum,
    generate_signal_frame as generate_legacy_mtf_qqe_rsi_momentum_signals,
)


def fixed_strategy_library() -> OrderedDict[str, StrategyTemplate]:
    return OrderedDict(
        [
            (
                "blackflag_fts_hma",
                StrategyTemplate(
                    strategy_id="blackflag_fts_hma",
                    display_name="Blackflag FTS + HMA",
                    description="Blackflag FTS swing-state flips filtered by a non-lookahead 60-minute Hull Moving Average regime.",
                    fixed_settings=BLACKFLAG_FTS_HMA_SETTINGS,
                    supported_timeframes=("15m", "5m"),
                    builder=build_blackflag_fts_hma,
                    signal_frame_builder=generate_blackflag_fts_hma_signals,
                    overlay_columns=("trail", "fib1", "fib2", "fib3", "higher_hma"),
                    indicator_snapshot_columns=(
                        "close",
                        "true_range",
                        "loss",
                        "blackflag_state",
                        "trail",
                        "buy_signal",
                        "sell_signal",
                        "fib1",
                        "fib2",
                        "fib3",
                        "higher_hma",
                        "concavity",
                        "hull_value",
                    ),
                    momentum_columns=("higher_hma",),
                    minimum_required_bars=60,
                    uses_higher_timeframe_data=True,
                ),
            ),
            (
                "ema_compression_volume_breakout",
                StrategyTemplate(
                    strategy_id="ema_compression_volume_breakout",
                    display_name="EMA Compression Volume Breakout",
                    description="EMA compression breakout with Thinkorswim-style choppiness, volume, candle-expansion, and fixed TP/SL brackets.",
                    fixed_settings=EMA_COMPRESSION_VOLUME_BREAKOUT_SETTINGS,
                    supported_timeframes=("15m", "5m"),
                    builder=build_ema_compression_volume_breakout,
                    signal_frame_builder=generate_ema_compression_volume_breakout_signals,
                    overlay_columns=("fast_ema", "slow_ema"),
                    indicator_snapshot_columns=(
                        "fast_ema",
                        "slow_ema",
                        "ema_spread_percent",
                        "ci",
                        "ci_avg",
                        "candle_range",
                        "prior_candle_range_1",
                        "prior_candle_range_2",
                        "volume",
                        "prior_volume_1",
                        "prior_volume_2",
                        "prior_volume_3",
                        "break_up",
                        "break_down",
                        "volume_confirm",
                        "range_expand",
                    ),
                    minimum_required_bars=250,
                ),
            ),
            (
                "legacy_mtf_qqe_rsi_momentum",
                StrategyTemplate(
                    strategy_id="legacy_mtf_qqe_rsi_momentum",
                    display_name="Legacy MTF QQE RSI Momentum",
                    description="Legacy QQE and RSI momentum with a 30-minute confirmation layer and corrected weighted higher-timeframe blend.",
                    fixed_settings=LEGACY_MTF_QQE_RSI_SETTINGS,
                    supported_timeframes=("15m", "5m"),
                    builder=build_legacy_mtf_qqe_rsi_momentum,
                    signal_frame_builder=generate_legacy_mtf_qqe_rsi_momentum_signals,
                    indicator_snapshot_columns=(
                        "source_used",
                        "rsi",
                        "rsi_ma",
                        "qqe_trailing_line",
                        "higher_rsi_ma",
                        "higher_trailing_line",
                        "combined_momentum",
                        "current_bull",
                        "current_bear",
                        "higher_bull",
                        "higher_bear",
                    ),
                    momentum_columns=("combined_momentum", "rsi_ma", "qqe_trailing_line"),
                    minimum_required_bars=60,
                    uses_higher_timeframe_data=True,
                ),
            ),
            (
                "blackflag_fts_qqe_momo",
                StrategyTemplate(
                    strategy_id="blackflag_fts_qqe_momo",
                    display_name="Blackflag FTS + QQEMoTV Confirmation",
                    description="Blackflag FTS direction confirmed by a fixed close-based QQE momentum state.",
                    fixed_settings=BLACKFLAG_FTS_QQE_MOMO_SETTINGS,
                    supported_timeframes=("15m", "5m"),
                    builder=build_blackflag_fts_qqe_momo,
                    signal_frame_builder=generate_blackflag_fts_qqe_momo_signals,
                    overlay_columns=("trail", "fib1", "fib2", "fib3"),
                    indicator_snapshot_columns=(
                        "close",
                        "true_range",
                        "loss",
                        "blackflag_state",
                        "trail",
                        "buy_signal",
                        "sell_signal",
                        "fib1",
                        "fib2",
                        "fib3",
                        "source_used",
                        "rsi",
                        "qqe_rsi_ma",
                        "qqe_trailing_line",
                        "qqe_bullish",
                        "qqe_bearish",
                    ),
                    momentum_columns=("qqe_rsi_ma", "qqe_trailing_line"),
                    minimum_required_bars=60,
                ),
            ),
        ]
    )


def strategy_registry() -> dict[str, Callable[[PyBrokerLabConfig], PyBrokerStrategyDefinition]]:
    return {strategy_id: template.builder for strategy_id, template in fixed_strategy_library().items()}
