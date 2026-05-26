from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pybroker.common import BarData

from trading_lab.indicators.hma import hull_moving_average
from trading_lab.indicators.rsi import relative_strength_index


def bars_to_frame(data: BarData | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(data.date),
                "open": data.open,
                "high": data.high,
                "low": data.low,
                "close": data.close,
                "volume": data.volume,
            }
        )
    if "timestamp" not in frame.columns:
        if "date" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["date"])
        else:
            raise ValueError("Bars must include a timestamp or date column.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["session_date"] = frame["timestamp"].dt.date
    return frame


def wilder_moving_average(series: pd.Series, period: int) -> pd.Series:
    return pd.Series(series, copy=False).astype(float).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return pd.Series(series, copy=False).astype(float).ewm(span=period, adjust=False, min_periods=period).mean()


def standard_true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - prev_close).abs()
    low_close = (frame["low"] - prev_close).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def modified_true_range(frame: pd.DataFrame, atr_period: int) -> pd.DataFrame:
    price_range = frame["high"] - frame["low"]
    avg_range = price_range.rolling(atr_period).mean()
    hi_lo = np.minimum(price_range, 1.5 * avg_range)
    prior_high = frame["high"].shift(1)
    prior_low = frame["low"].shift(1)
    prior_close = frame["close"].shift(1)
    href = np.where(
        frame["low"] <= prior_high,
        frame["high"] - prior_close,
        (frame["high"] - prior_close) - 0.5 * (frame["low"] - prior_high),
    )
    lref = np.where(
        frame["high"] >= prior_low,
        prior_close - frame["low"],
        (prior_close - frame["low"]) - 0.5 * (prior_low - frame["high"]),
    )
    true_range = pd.concat(
        [
            pd.Series(hi_lo, index=frame.index, dtype=float),
            pd.Series(href, index=frame.index, dtype=float),
            pd.Series(lref, index=frame.index, dtype=float),
        ],
        axis=1,
    ).max(axis=1)
    return pd.DataFrame({"HiLo": hi_lo, "HRef": href, "LRef": lref, "true_range": true_range}, index=frame.index)


def resample_ohlcv(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = timeframe.lower()
    indexed = frame.copy().set_index("timestamp")
    aggregated = (
        indexed.resample(rule, label="right", closed="right", origin="start_day", offset="30min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    aggregated["session_date"] = aggregated["timestamp"].dt.date
    return aggregated


def merge_higher_timeframe_series(frame: pd.DataFrame, higher_frame: pd.DataFrame, column: str, output_column: str) -> pd.Series:
    left = frame[["timestamp"]].copy().sort_values("timestamp")
    right = higher_frame[["timestamp", column]].copy().sort_values("timestamp").dropna(subset=[column])
    merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
    return merged[output_column if output_column in merged.columns else column].rename(output_column)


def cross_above(series: pd.Series, level: float) -> pd.Series:
    values = pd.Series(series, copy=False).astype(float)
    return values.gt(level) & values.shift(1).le(level)


def cross_below(series: pd.Series, level: float) -> pd.Series:
    values = pd.Series(series, copy=False).astype(float)
    return values.lt(level) & values.shift(1).ge(level)


def preclose_signal(frame: pd.DataFrame) -> pd.Series:
    session_last_bar = frame["session_date"].ne(frame["session_date"].shift(-1))
    return session_last_bar.shift(-1, fill_value=False)


def clock_time_strings(frame: pd.DataFrame) -> pd.Series:
    return frame["timestamp"].dt.strftime("%H:%M")


def parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.split(":")
    return int(hour), int(minute)


def time_at_or_after(frame: pd.DataFrame, hhmm: str) -> pd.Series:
    hour, minute = parse_hhmm(hhmm)
    minutes = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    return minutes >= hour * 60 + minute


def time_between(frame: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.Series:
    start_hour, start_minute = parse_hhmm(start_hhmm)
    end_hour, end_minute = parse_hhmm(end_hhmm)
    minutes = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    return minutes.between(start_hour * 60 + start_minute, end_hour * 60 + end_minute, inclusive="left")


def combine_weighted_value(primary: pd.Series, secondary: pd.Series, multiplier: float, mode: str) -> pd.Series:
    if mode == "legacy_exact":
        return (primary * multiplier) + 1.0
    if mode == "corrected_weighted":
        return ((primary * multiplier) + secondary) / (multiplier + 1.0)
    raise ValueError(f"Unsupported weighting mode: {mode}")


def timeframe_minutes(timeframe: str) -> int:
    value = timeframe.lower()
    if value.endswith("min"):
        return int(value[:-3])
    if value.endswith("m"):
        return int(value[:-1])
    if value == "1d":
        return 390
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def compute_higher_timeframe_hma(frame: pd.DataFrame, *, timeframe: str, source: str, length: int) -> pd.DataFrame:
    higher = resample_ohlcv(frame, timeframe)
    higher["hma"] = hull_moving_average(higher[source].astype(float), length)
    merged = merge_higher_timeframe_provenance(
        frame,
        higher[["timestamp", "hma"]].dropna().sort_values("timestamp"),
        timeframe=timeframe,
        value_columns=("hma",),
        prefix="higher_hma",
    )
    return merged.rename(columns={"hma": "higher_hma"})


def merge_higher_timeframe_provenance(
    lower_frame: pd.DataFrame,
    higher_frame: pd.DataFrame,
    *,
    timeframe: str,
    value_columns: tuple[str, ...],
    prefix: str,
) -> pd.DataFrame:
    left = lower_frame[["timestamp"]].copy().sort_values("timestamp")
    right = higher_frame.copy().sort_values("timestamp")
    right = right.rename(columns={"timestamp": f"{prefix}_source_close_timestamp"})
    merged = pd.merge_asof(
        left,
        right,
        left_on="timestamp",
        right_on=f"{prefix}_source_close_timestamp",
        direction="backward",
    )
    merged[f"{prefix}_source_timestamp"] = merged[f"{prefix}_source_close_timestamp"]
    merged[f"{prefix}_higher_close_timestamp"] = merged[f"{prefix}_source_close_timestamp"]
    merged[f"{prefix}_bar_complete"] = (
        pd.to_datetime(merged["timestamp"]).ge(pd.to_datetime(merged[f"{prefix}_higher_close_timestamp"]))
        & merged[f"{prefix}_higher_close_timestamp"].notna()
    )
    merged[f"{prefix}_lookahead_flag"] = (
        pd.to_datetime(merged[f"{prefix}_higher_close_timestamp"]).gt(pd.to_datetime(merged["timestamp"]))
    ).fillna(False)
    merged[f"{prefix}_lookahead_result"] = np.where(
        merged[f"{prefix}_lookahead_flag"],
        "FAIL",
        np.where(merged[f"{prefix}_bar_complete"], "PASS", "WARNING"),
    )
    return merged


def compute_qqe_frame(
    source: pd.Series,
    *,
    rsi_period: int,
    slow_factor: int,
    qqe: float,
) -> pd.DataFrame:
    values = pd.Series(source, copy=False).astype(float)
    rsi = relative_strength_index(values, length=rsi_period)
    rsi_ma = ema(rsi, slow_factor)
    wilder_period = rsi_period * 2 - 1
    atr_rsi = (rsi_ma.shift(1) - rsi_ma).abs()
    atr_rsi_ma = ema(atr_rsi, wilder_period)
    dar = ema(atr_rsi_ma, wilder_period) * qqe
    rs_index = rsi_ma
    new_short_band = rs_index + dar
    new_long_band = rs_index - dar

    long_band = pd.Series(np.nan, index=values.index, dtype=float)
    short_band = pd.Series(np.nan, index=values.index, dtype=float)
    trend = pd.Series(0, index=values.index, dtype=int)

    for idx in range(1, len(values)):
        if pd.isna(rs_index.iloc[idx]) or pd.isna(dar.iloc[idx]):
            continue
        prev_long = long_band.iloc[idx - 1]
        prev_short = short_band.iloc[idx - 1]
        prev_trend = trend.iloc[idx - 1]
        prev_rsi = rs_index.iloc[idx - 1]
        curr_rsi = rs_index.iloc[idx]

        if pd.isna(prev_long):
            long_band.iloc[idx] = new_long_band.iloc[idx]
        else:
            long_band.iloc[idx] = max(prev_long, new_long_band.iloc[idx]) if prev_rsi > prev_long and curr_rsi > prev_long else new_long_band.iloc[idx]

        if pd.isna(prev_short):
            short_band.iloc[idx] = new_short_band.iloc[idx]
        else:
            short_band.iloc[idx] = min(prev_short, new_short_band.iloc[idx]) if prev_rsi < prev_short and curr_rsi < prev_short else new_short_band.iloc[idx]

        if not pd.isna(prev_short) and curr_rsi > prev_short:
            trend.iloc[idx] = 1
        elif not pd.isna(prev_long) and curr_rsi < prev_long:
            trend.iloc[idx] = -1
        else:
            trend.iloc[idx] = prev_trend

    trailing = pd.Series(np.where(trend >= 0, long_band, short_band), index=values.index, dtype=float)
    signal = trend.diff().fillna(0).clip(-1, 1).astype(int)
    return pd.DataFrame(
        {
            "rsi": rsi,
            "rsi_ma": rsi_ma,
            "atr_rsi": atr_rsi,
            "atr_rsi_ma": atr_rsi_ma,
            "dar": dar,
            "new_long_band": new_long_band,
            "new_short_band": new_short_band,
            "long_band": long_band,
            "short_band": short_band,
            "trend": trend,
            "trailing_line": trailing,
            "signal": signal,
            "rsi_ma_dot": rsi_ma.diff(),
        }
    )


def strategy_metric_from_trades(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty or "pnl" not in trades.columns:
        return {"win_rate": 0.0, "profit_factor": 0.0, "average_win": 0.0, "average_loss": 0.0}
    pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    return_pct = pd.to_numeric(trades.get("return_pct", pd.Series(0.0, index=trades.index)), errors="coerce").fillna(0.0) / 100.0
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    avg_win = float(return_pct[pnl > 0].mean()) if not wins.empty else 0.0
    avg_loss = float(return_pct[pnl < 0].mean()) if not losses.empty else 0.0
    win_rate = float((pnl > 0).mean()) if len(pnl) else 0.0
    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_win": avg_win,
        "average_loss": avg_loss,
    }


def resolve_position_size_shares(ctx, *, sizing_method: str, sizing_value: float) -> float:
    method = str(sizing_method)
    if method == "percent_equity":
        return float(ctx.calc_target_shares(float(sizing_value)))
    close = float(ctx.close[-1])
    if close <= 0:
        return 0.0
    if method == "fixed_dollar":
        return max(float(sizing_value) / close, 0.0)
    if method == "fixed_shares":
        return max(float(sizing_value), 0.0)
    raise ValueError(f"Unsupported sizing method: {sizing_method}")


def sizing_method_label(method: str, value: float) -> str:
    if method == "percent_equity":
        return f"{value:.0%} equity allocation"
    if method == "fixed_dollar":
        return f"Fixed dollar allocation ({value:,.2f})"
    if method == "fixed_shares":
        return f"Fixed share quantity ({value:,.4f})"
    return method


def safe_json_dumps(payload: dict[str, Any]) -> str:
    normalized = {}
    for key, value in payload.items():
        if isinstance(value, (np.bool_, bool)):
            normalized[key] = bool(value)
        elif isinstance(value, (np.floating, float)):
            normalized[key] = None if pd.isna(value) else float(value)
        elif isinstance(value, (np.integer, int)):
            normalized[key] = int(value)
        elif pd.isna(value):
            normalized[key] = None
        else:
            normalized[key] = value
    return json.dumps(normalized, sort_keys=True)


@dataclass(frozen=True)
class StrategyTemplate:
    strategy_id: str
    display_name: str
    description: str
    fixed_settings: dict[str, Any]
    supported_timeframes: tuple[str, ...]
    builder: Any
    signal_frame_builder: Any
    overlay_columns: tuple[str, ...] = ()
    indicator_snapshot_columns: tuple[str, ...] = ()
    momentum_columns: tuple[str, ...] = ()
    minimum_required_bars: int = 50
    uses_higher_timeframe_data: bool = False
