from __future__ import annotations

from datetime import date

import pandas as pd


def compute_forward_returns(series: pd.Series, horizon: int) -> pd.Series:
    values = pd.Series(series, copy=False).dropna().sort_index()
    return values.shift(-horizon) / values - 1.0


def compute_average_forward_return(series: pd.Series, horizon: int, *, month: int | None = None, iso_week: int | None = None, before_year: int | None = None) -> tuple[float | None, int]:
    forward_returns = compute_forward_returns(series, horizon).dropna()
    if forward_returns.empty:
        return None, 0
    mask = pd.Series(True, index=forward_returns.index)
    if month is not None:
        mask &= forward_returns.index.month == month
    if iso_week is not None:
        mask &= forward_returns.index.isocalendar().week.astype(int) == iso_week
    if before_year is not None:
        mask &= forward_returns.index.year < before_year
    filtered = forward_returns[mask]
    if filtered.empty:
        return None, 0
    return float(filtered.mean()), int(filtered.shape[0])


def compute_historical_period_average(series: pd.Series, *, month: int | None = None, iso_week: int | None = None, before_year: int | None = None) -> tuple[float | None, int]:
    values = pd.Series(series, copy=False).dropna().sort_index()
    if values.empty:
        return None, 0
    frame = values.to_frame("close")
    frame["year"] = frame.index.year
    frame["month"] = frame.index.month
    frame["iso_week"] = frame.index.isocalendar().week.astype(int)
    groups: list[tuple[tuple[int, int], pd.DataFrame]] = []
    if month is not None:
        groups = list(frame.groupby(["year", "month"]))
        target_match = lambda key: key[1] == month
    elif iso_week is not None:
        groups = list(frame.groupby(["year", "iso_week"]))
        target_match = lambda key: key[1] == iso_week
    else:
        return None, 0
    returns: list[float] = []
    for key, bucket in groups:
        year = int(key[0])
        if before_year is not None and year >= before_year:
            continue
        if not target_match(key) or len(bucket) < 2:
            continue
        start_value = float(bucket["close"].iloc[0])
        if start_value == 0.0:
            continue
        returns.append(float(bucket["close"].iloc[-1] / start_value - 1.0))
    if not returns:
        return None, 0
    return float(sum(returns) / len(returns)), len(returns)


def build_seasonality_rows(series: pd.Series, *, as_of_date: date) -> list[dict[str, object]]:
    iso_week = int(as_of_date.isocalendar()[1])
    month = int(as_of_date.month)
    next_5_value, next_5_samples = compute_average_forward_return(series, 5, iso_week=iso_week, before_year=as_of_date.year)
    next_20_value, next_20_samples = compute_average_forward_return(series, 20, month=month, before_year=as_of_date.year)
    week_avg, week_samples = compute_historical_period_average(series, iso_week=iso_week, before_year=as_of_date.year)
    month_avg, month_samples = compute_historical_period_average(series, month=month, before_year=as_of_date.year)
    return [
        {"metric": "Next 5 trading days historical average", "value": next_5_value, "sample_size": next_5_samples, "methodology": "Historical 5-day forward returns for the current ISO week number."},
        {"metric": "Next 20 trading days historical average", "value": next_20_value, "sample_size": next_20_samples, "methodology": "Historical 20-day forward returns for sessions in the current calendar month."},
        {"metric": "Current calendar week historical average", "value": week_avg, "sample_size": week_samples, "methodology": "Full-week returns for the current ISO week number across prior years."},
        {"metric": "Current calendar month historical average", "value": month_avg, "sample_size": month_samples, "methodology": "Full-month returns for the current calendar month across prior years."},
    ]
