from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from trading_lab.data.market_calendar import MarketCalendar, get_default_calendar


REQUIRED_BAR_COLUMNS = [
    "source_vendor",
    "symbol",
    "timeframe",
    "timestamp",
    "session_date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
    "adjusted_flag",
    "retrieved_at",
]


@dataclass
class ValidationResult:
    frame: pd.DataFrame
    warnings: list[str]
    severe_issues: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.severe_issues


def validate_stock_bars(
    df: pd.DataFrame,
    *,
    allow_empty: bool = False,
    market_calendar: MarketCalendar | None = None,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
) -> ValidationResult:
    """Validate normalized stock bars and collect non-fatal warnings."""
    missing = [column for column in REQUIRED_BAR_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required stock bar columns: {missing}")

    if df.empty:
        if allow_empty:
            return ValidationResult(frame=df.copy(), warnings=["No rows available for the requested range."], severe_issues=[])
        raise ValueError("Stock bars dataframe is empty")

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date
    frame["retrieved_at"] = pd.to_datetime(frame["retrieved_at"])

    severe_issues: list[str] = []
    warnings: list[str] = []

    if (frame["high"] < frame["low"]).any():
        severe_issues.append("Found rows where high is below low.")
    for column in ["open", "close"]:
        if (frame["high"] < frame[column]).any():
            severe_issues.append(f"Found rows where high is below {column}.")
        if (frame["low"] > frame[column]).any():
            severe_issues.append(f"Found rows where low is above {column}.")
    if (frame["volume"] < 0).any():
        severe_issues.append("Found rows with negative volume.")

    sorted_dates = pd.Series(pd.to_datetime(frame["session_date"]).sort_values().unique())
    if len(sorted_dates) >= 2:
        cal = market_calendar or get_default_calendar()
        range_start = pd.Timestamp(start_date).date() if start_date is not None else pd.Timestamp(sorted_dates.iloc[0]).date()
        range_end = pd.Timestamp(end_date).date() if end_date is not None else pd.Timestamp(sorted_dates.iloc[-1]).date()
        missing_sessions = cal.missing_sessions(sorted_dates.dt.date.tolist(), range_start, range_end)
        if missing_sessions:
            preview = ", ".join(str(item) for item in missing_sessions[:5])
            warnings.append(f"Missing trading sessions detected in cache. Examples: {preview}.")

    if severe_issues:
        raise ValueError("; ".join(severe_issues))
    return ValidationResult(frame=frame, warnings=warnings, severe_issues=severe_issues)
