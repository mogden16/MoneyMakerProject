from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import pandas as pd

from trading_lab.data.intraday import filter_regular_market_hours


TRADINGVIEW_COLUMN_ALIASES = {
    "time": "timestamp",
    "timestamp": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "symbol": "symbol",
    "ticker": "symbol",
}


@dataclass(frozen=True)
class CandleComparisonResult:
    summary: pd.DataFrame
    merged: pd.DataFrame
    worst_mismatches: pd.DataFrame


def _normalize_column_name(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _read_csv_bytes(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(payload))


def _parse_timestamp_column(series: pd.Series, timezone: str) -> pd.Series:
    raw = series.copy()
    if pd.api.types.is_numeric_dtype(raw):
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.dropna().empty:
            return pd.Series(pd.NaT, index=series.index)
        max_abs = numeric.dropna().abs().max()
        if max_abs > 1_000_000_000_000:
            parsed = pd.to_datetime(numeric, unit="ms", utc=True)
        elif max_abs > 1_000_000_000:
            parsed = pd.to_datetime(numeric, unit="s", utc=True)
        else:
            parsed = pd.to_datetime(numeric, errors="coerce")
    else:
        parsed = pd.to_datetime(raw, errors="coerce", utc=False)
    if getattr(parsed.dt, "tz", None) is None:
        return parsed.dt.tz_localize(timezone).dt.tz_localize(None)
    return parsed.dt.tz_convert(timezone).dt.tz_localize(None)


def timeframe_to_offset(timeframe: str) -> pd.Timedelta:
    value = str(timeframe).lower()
    if value.endswith("min"):
        return pd.Timedelta(minutes=int(value[:-3]))
    if value.endswith("m"):
        return pd.Timedelta(minutes=int(value[:-1]))
    if value.endswith("h"):
        return pd.Timedelta(hours=int(value[:-1]))
    if value == "1d":
        return pd.Timedelta(days=1)
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def normalize_tradingview_candles(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    timezone: str = "America/New_York",
    timestamp_basis: str = "bar_start",
    symbol: str | None = None,
) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [_normalize_column_name(column) for column in renamed.columns]
    mapped_columns = {}
    for column in renamed.columns:
        mapped_columns[column] = TRADINGVIEW_COLUMN_ALIASES.get(column, column)
    renamed = renamed.rename(columns=mapped_columns)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(renamed.columns)
    if missing:
        raise ValueError(f"TradingView CSV is missing required columns: {sorted(missing)}")
    normalized = renamed.loc[:, [column for column in ["timestamp", "open", "high", "low", "close", "volume", "symbol"] if column in renamed.columns]].copy()
    normalized["timestamp"] = _parse_timestamp_column(normalized["timestamp"], timezone)
    if str(timestamp_basis) == "bar_end":
        normalized["timestamp"] = normalized["timestamp"] - timeframe_to_offset(timeframe)
    if "symbol" in normalized.columns:
        normalized["symbol"] = normalized["symbol"].astype(str).replace("", symbol or "")
    else:
        normalized["symbol"] = str(symbol or "")
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)
    return normalized


def parse_tradingview_csv(
    payload: bytes,
    *,
    timeframe: str,
    timezone: str = "America/New_York",
    timestamp_basis: str = "bar_start",
    symbol: str | None = None,
) -> pd.DataFrame:
    return normalize_tradingview_candles(
        _read_csv_bytes(payload),
        timeframe=timeframe,
        timezone=timezone,
        timestamp_basis=timestamp_basis,
        symbol=symbol,
    )


def normalize_yfinance_candles(
    frame: pd.DataFrame,
    *,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    normalized = frame.copy()
    timestamp_column = "timestamp" if "timestamp" in normalized.columns else "date"
    normalized["timestamp"] = pd.to_datetime(normalized[timestamp_column], errors="coerce")
    if getattr(normalized["timestamp"].dt, "tz", None) is None:
        normalized["timestamp"] = normalized["timestamp"].dt.tz_localize(timezone).dt.tz_localize(None)
    else:
        normalized["timestamp"] = normalized["timestamp"].dt.tz_convert(timezone).dt.tz_localize(None)
    keep = [column for column in ["symbol", "timestamp", "open", "high", "low", "close", "volume"] if column in normalized.columns]
    return normalized.loc[:, keep].sort_values("timestamp").reset_index(drop=True)


def maybe_filter_regular_hours(frame: pd.DataFrame, *, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return frame.copy()
    return filter_regular_market_hours(frame, timestamp_column="timestamp")


def apply_alignment_shift(frame: pd.DataFrame, *, timeframe: str, bars: int) -> pd.DataFrame:
    if bars == 0:
        return frame.copy()
    shifted = frame.copy()
    shifted["timestamp"] = shifted["timestamp"] + timeframe_to_offset(timeframe) * bars
    return shifted


def compare_candle_frames(
    yfinance_frame: pd.DataFrame,
    tradingview_frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    timezone: str = "America/New_York",
    regular_hours_only: bool = True,
    timestamp_basis: str = "bar_start",
    shift_dataset: str = "none",
    tolerance: float = 1e-6,
) -> CandleComparisonResult:
    yfinance = normalize_yfinance_candles(yfinance_frame, timezone=timezone)
    tradingview = normalize_tradingview_candles(
        tradingview_frame,
        timeframe=timeframe,
        timezone=timezone,
        timestamp_basis=timestamp_basis,
        symbol=symbol,
    )
    yfinance = maybe_filter_regular_hours(yfinance, enabled=regular_hours_only)
    tradingview = maybe_filter_regular_hours(tradingview, enabled=regular_hours_only)
    if shift_dataset == "shift_yfinance_forward_1_bar":
        yfinance = apply_alignment_shift(yfinance, timeframe=timeframe, bars=1)
    elif shift_dataset == "shift_tradingview_forward_1_bar":
        tradingview = apply_alignment_shift(tradingview, timeframe=timeframe, bars=1)

    yfinance = yfinance.rename(columns={column: f"{column}_yfinance" for column in ["open", "high", "low", "close", "volume"]})
    tradingview = tradingview.rename(columns={column: f"{column}_tradingview" for column in ["open", "high", "low", "close", "volume"]})
    merged = pd.merge(yfinance, tradingview, on="timestamp", how="outer", suffixes=("_yfinance", "_tradingview")).sort_values("timestamp").reset_index(drop=True)
    merged["symbol"] = symbol
    present_yfinance = merged["open_yfinance"].notna()
    present_tradingview = merged["open_tradingview"].notna()
    matched = present_yfinance & present_tradingview
    for field in ["open", "high", "low", "close", "volume"]:
        merged[f"abs_diff_{field}"] = (merged[f"{field}_yfinance"] - merged[f"{field}_tradingview"]).abs()
    merged["avg_abs_ohlc_diff"] = merged[[f"abs_diff_{field}" for field in ["open", "high", "low", "close"]]].mean(axis=1)
    merged["max_abs_ohlc_diff"] = merged[[f"abs_diff_{field}" for field in ["open", "high", "low", "close"]]].max(axis=1)
    merged["ohlc_diff_exceeds_tolerance"] = matched & merged["max_abs_ohlc_diff"].fillna(0.0).gt(float(tolerance))
    merged["close_diff"] = merged["close_yfinance"] - merged["close_tradingview"]
    mismatch_mask = (~present_yfinance) | (~present_tradingview) | merged["ohlc_diff_exceeds_tolerance"]
    first_mismatch = merged.loc[mismatch_mask, "timestamp"].iloc[0] if mismatch_mask.any() else pd.NaT
    worst = merged.loc[mismatch_mask].copy()
    worst = worst.sort_values(["max_abs_ohlc_diff", "timestamp"], ascending=[False, True]).head(25).reset_index(drop=True)
    summary = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "timezone": timezone,
                "timestamp_basis": timestamp_basis,
                "regular_hours_only": regular_hours_only,
                "shift_dataset": shift_dataset,
                "matched_bars_count": int(matched.sum()),
                "missing_bars_in_yfinance": int((~present_yfinance & present_tradingview).sum()),
                "missing_bars_in_tradingview": int((present_yfinance & ~present_tradingview).sum()),
                "first_mismatched_timestamp": first_mismatch,
                "max_absolute_open_difference": float(merged["abs_diff_open"].max(skipna=True) or 0.0),
                "max_absolute_high_difference": float(merged["abs_diff_high"].max(skipna=True) or 0.0),
                "max_absolute_low_difference": float(merged["abs_diff_low"].max(skipna=True) or 0.0),
                "max_absolute_close_difference": float(merged["abs_diff_close"].max(skipna=True) or 0.0),
                "max_absolute_volume_difference": float(merged["abs_diff_volume"].max(skipna=True) or 0.0),
                "average_absolute_ohlc_difference": float(merged.loc[matched, "avg_abs_ohlc_diff"].mean() or 0.0),
                "bars_with_ohlc_difference_over_tolerance": int(merged["ohlc_diff_exceeds_tolerance"].sum()),
            }
        ]
    )
    return CandleComparisonResult(summary=summary, merged=merged, worst_mismatches=worst)
