from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from zoneinfo import ZoneInfo

import pandas as pd


EASTERN_TZ = ZoneInfo("America/New_York")
INTRADAY_MAX_HISTORY_DAYS = 60
INTRADAY_BAR_COUNTS = {"15m": 26, "5m": 78}
SUPPORTED_TIMEFRAMES = {"1d", "15m", "5m"}


@dataclass(frozen=True)
class IntradayRangeClamp:
    requested_start: date
    requested_end: date
    effective_start: date
    effective_end: date
    was_clamped: bool
    warning: str | None


def is_intraday_timeframe(timeframe: str) -> bool:
    return timeframe in {"15m", "5m"}


def clamp_intraday_date_range(start_date: str | date, end_date: str | date, max_days: int = INTRADAY_MAX_HISTORY_DAYS) -> IntradayRangeClamp:
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    earliest = (pd.Timestamp(end) - pd.Timedelta(days=max_days)).date()
    effective_start = max(start, earliest)
    was_clamped = effective_start != start
    warning = None
    if was_clamped:
        warning = (
            "yfinance intraday history is limited to roughly the most recent "
            f"{max_days} days. The requested start date was clamped from {start} to {effective_start}."
        )
    return IntradayRangeClamp(
        requested_start=start,
        requested_end=end,
        effective_start=effective_start,
        effective_end=end,
        was_clamped=was_clamped,
        warning=warning,
    )


def to_eastern_naive(timestamps: pd.Series) -> pd.Series:
    series = pd.to_datetime(timestamps)
    if getattr(series.dt, "tz", None) is None:
        return series.dt.tz_localize(EASTERN_TZ).dt.tz_localize(None)
    return series.dt.tz_convert(EASTERN_TZ).dt.tz_localize(None)


def filter_regular_market_hours(frame: pd.DataFrame, *, timestamp_column: str = "timestamp") -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    filtered = frame.copy()
    filtered[timestamp_column] = to_eastern_naive(filtered[timestamp_column])
    minutes = filtered[timestamp_column].dt.hour * 60 + filtered[timestamp_column].dt.minute
    regular = filtered[(minutes >= 570) & (minutes < 960)].copy()
    regular["session_date"] = regular[timestamp_column].dt.date
    return regular


def infer_intraday_gap_warnings(frame: pd.DataFrame, timeframe: str) -> list[str]:
    if frame.empty or timeframe not in INTRADAY_BAR_COUNTS:
        return []
    warnings: list[str] = []
    counts = frame.groupby("session_date").size()
    expected_count = INTRADAY_BAR_COUNTS[timeframe]
    incomplete = counts[counts < expected_count]
    if not incomplete.empty:
        preview = ", ".join(f"{idx}: {value}/{expected_count}" for idx, value in incomplete.head(5).items())
        warnings.append(f"Missing intraday bars detected. Examples: {preview}.")
    out_of_hours = frame[
        (pd.to_datetime(frame["timestamp"]).dt.hour * 60 + pd.to_datetime(frame["timestamp"]).dt.minute < 570)
        | (pd.to_datetime(frame["timestamp"]).dt.hour * 60 + pd.to_datetime(frame["timestamp"]).dt.minute >= 960)
    ]
    if not out_of_hours.empty:
        warnings.append("Unexpected premarket or after-hours bars were detected in the intraday dataset.")
    return warnings
