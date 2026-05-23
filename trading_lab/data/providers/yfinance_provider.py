from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
import yfinance as yf

from trading_lab.data.database import TradingLabDatabase
from trading_lab.data.intraday import (
    SUPPORTED_TIMEFRAMES,
    clamp_intraday_date_range,
    filter_regular_market_hours,
    infer_intraday_gap_warnings,
    is_intraday_timeframe,
)
from trading_lab.data.market_calendar import MarketCalendar, get_default_calendar
from trading_lab.data.providers.base import MarketDataProvider
from trading_lab.data.validation import validate_stock_bars


@dataclass
class CacheStatus:
    symbol: str
    timeframe: str
    requested_start: str
    requested_end: str
    latest_cached_session: str | None
    expected_latest_session: str
    cache_status: str
    validation_warnings: list[str]
    used_cached_data: bool
    performed_refresh: bool
    calendar_name: str
    using_calendar_fallback: bool


class YFinanceDataProvider(MarketDataProvider):
    """yfinance provider with incremental DuckDB-backed caching for daily and SPY intraday research."""

    def __init__(
        self,
        database: TradingLabDatabase,
        cache_max_age_hours: int = 24,
        force_refresh_default: bool = False,
        allow_stale_cache: bool = False,
        market_calendar: MarketCalendar | None = None,
    ) -> None:
        self.database = database
        self.cache_max_age_hours = cache_max_age_hours
        self.force_refresh_default = force_refresh_default
        self.allow_stale_cache = allow_stale_cache
        self.market_calendar = market_calendar or get_default_calendar()
        self.last_fetch_status: dict[str, CacheStatus] = {}

    def get_stock_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        symbol = symbol.upper()
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        effective_force_refresh = force_refresh or self.force_refresh_default
        requested_start = pd.Timestamp(start_date).date()
        requested_end = pd.Timestamp(end_date).date()
        clamp_warning: str | None = None
        effective_start = requested_start
        if is_intraday_timeframe(timeframe):
            clamp = clamp_intraday_date_range(requested_start, requested_end)
            effective_start = clamp.effective_start
            clamp_warning = clamp.warning

        cached = self.database.read_stock_bars(symbol=symbol, start_date=str(effective_start), end_date=end_date, timeframe=timeframe)
        latest_cached = self.database.get_latest_stock_bar_session(symbol=symbol, timeframe=timeframe)
        expected_latest = self._expected_latest_marker(requested_end, timeframe)
        cache_status = self._evaluate_cache_status(cached, latest_cached, effective_start, requested_end, expected_latest, timeframe)

        performed_refresh = False
        used_cached_data = not cached.empty
        if effective_force_refresh or cache_status != "fresh":
            try:
                self._refresh_bars(symbol, effective_start, requested_end, timeframe, latest_cached, full_refresh=effective_force_refresh or cached.empty)
                performed_refresh = True
                cached = self.database.read_stock_bars(symbol=symbol, start_date=str(effective_start), end_date=end_date, timeframe=timeframe)
                latest_cached = self.database.get_latest_stock_bar_session(symbol=symbol, timeframe=timeframe)
                cache_status = self._evaluate_cache_status(cached, latest_cached, effective_start, requested_end, expected_latest, timeframe)
                used_cached_data = not cached.empty
            except Exception:
                if not self.allow_stale_cache or cached.empty:
                    raise
                cache_status = "stale_allowed"

        validation = validate_stock_bars(
            cached,
            allow_empty=False,
            market_calendar=self.market_calendar,
            start_date=effective_start,
            end_date=requested_end,
        )
        result = validation.frame.sort_values("timestamp").reset_index(drop=True)
        validation_warnings = list(validation.warnings)
        if clamp_warning:
            validation_warnings.insert(0, clamp_warning)
        validation_warnings.extend(infer_intraday_gap_warnings(result, timeframe))
        status = CacheStatus(
            symbol=symbol,
            timeframe=timeframe,
            requested_start=str(requested_start),
            requested_end=str(requested_end),
            latest_cached_session=str(latest_cached) if latest_cached is not None else None,
            expected_latest_session=str(expected_latest),
            cache_status=cache_status,
            validation_warnings=validation_warnings,
            used_cached_data=used_cached_data,
            performed_refresh=performed_refresh,
            calendar_name=self.market_calendar.calendar_name,
            using_calendar_fallback=self.market_calendar.status.using_fallback,
        )
        result.attrs["cache_status"] = status
        self.last_fetch_status[symbol] = status
        self.database.insert_ingestion_log(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "requested_start": effective_start,
                "requested_end": requested_end,
                "latest_cached_session": pd.Timestamp(latest_cached).date() if latest_cached is not None else None,
                "expected_latest_session": expected_latest,
                "cache_status": cache_status,
                "validation_warnings_json": json.dumps(validation_warnings),
                "used_cached_data": used_cached_data,
                "performed_refresh": performed_refresh,
                "retrieved_at": datetime.now(UTC).replace(tzinfo=None),
            }
        )
        return result

    def _refresh_bars(
        self,
        symbol: str,
        requested_start: pd.Timestamp | date,
        requested_end: pd.Timestamp | date,
        timeframe: str,
        latest_cached: pd.Timestamp | None,
        *,
        full_refresh: bool,
    ) -> None:
        start_ts = pd.Timestamp(requested_start)
        end_ts = pd.Timestamp(requested_end)
        download_start = start_ts if full_refresh or latest_cached is None or latest_cached.date() < start_ts.date() else latest_cached.normalize()
        try:
            bars = self._normalize_download(symbol=symbol, timeframe=timeframe, raw=self._download_bars(symbol=symbol, start=download_start, end=end_ts))
            self.database.upsert_stock_bars(bars, symbol=symbol, timeframe=timeframe)
        except Exception:
            if full_refresh:
                raise
            fallback = self._normalize_download(symbol=symbol, timeframe=timeframe, raw=self._download_bars(symbol=symbol, start=start_ts, end=end_ts))
            self.database.replace_stock_bars(fallback, symbol=symbol, timeframe=timeframe)

        actions = self._download_corporate_actions(symbol)
        if not actions.empty:
            self.database.upsert_corporate_actions(actions, symbol=symbol)

    def _download_bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp, interval: str = "1d") -> pd.DataFrame:
        downloaded = yf.download(
            tickers=symbol,
            start=start,
            end=end + pd.Timedelta(days=1),
            interval=interval,
            auto_adjust=False,
            actions=True,
            progress=False,
            prepost=False,
        )
        if downloaded.empty:
            raise ValueError(f"No data returned for symbol {symbol}")
        return downloaded

    def _evaluate_cache_status(self, cached: pd.DataFrame, latest_cached: pd.Timestamp | None, requested_start, requested_end, expected_latest, timeframe: str) -> str:
        if cached.empty or latest_cached is None:
            return "missing"
        first_cached = pd.to_datetime(cached["session_date"]).min().date()
        latest_expected_for_range = expected_latest if is_intraday_timeframe(timeframe) else min(pd.Timestamp(requested_end).date(), expected_latest)
        freshness_age_hours = (datetime.now(UTC).replace(tzinfo=None) - pd.to_datetime(cached["retrieved_at"]).max()).total_seconds() / 3600
        latest_marker = latest_cached if is_intraday_timeframe(timeframe) else latest_cached.date()
        if latest_marker >= latest_expected_for_range and first_cached <= pd.Timestamp(requested_start).date() and freshness_age_hours <= self.cache_max_age_hours:
            return "fresh"
        if latest_marker >= latest_expected_for_range and first_cached <= pd.Timestamp(requested_start).date():
            return "aged"
        return "stale"

    def _normalize_download(self, symbol: str, timeframe: str, raw: pd.DataFrame) -> pd.DataFrame:
        frame = raw.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame = frame.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
                "Dividends": "dividends",
                "Stock Splits": "stock_splits",
            }
        )
        for column in ["dividends", "stock_splits"]:
            if column not in frame.columns:
                frame[column] = 0.0
        frame["adj_close"] = frame.get("adj_close", frame["close"])
        index_name = raw.index.name or "Date"
        frame = frame.reset_index().rename(columns={index_name: "timestamp", "Datetime": "timestamp", "Date": "timestamp"})
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        if is_intraday_timeframe(timeframe):
            frame = filter_regular_market_hours(frame, timestamp_column="timestamp")
        else:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.tz_localize(None)
        frame["session_date"] = frame["timestamp"].dt.date
        frame["source_vendor"] = "yfinance"
        frame["symbol"] = symbol
        frame["timeframe"] = timeframe
        frame["adjusted_flag"] = False
        frame["retrieved_at"] = datetime.now(UTC).replace(tzinfo=None)
        ordered = frame[
            [
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
        ].copy()
        numeric_columns = ["open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"]
        ordered[numeric_columns] = ordered[numeric_columns].astype(float)
        return validate_stock_bars(ordered, market_calendar=self.market_calendar).frame

    def _expected_latest_marker(self, requested_end: date, timeframe: str):
        if timeframe == "1d":
            return self.market_calendar.latest_completed_session(requested_end)
        return self.market_calendar.latest_completed_bar(requested_end, timeframe)

    def _download_corporate_actions(self, symbol: str) -> pd.DataFrame:
        actions = yf.Ticker(symbol).actions.reset_index()
        if actions.empty:
            return pd.DataFrame(columns=["source_vendor", "symbol", "action_type", "effective_date", "cash_amount", "split_ratio", "split_from", "split_to", "retrieved_at"])
        actions["Date"] = pd.to_datetime(actions["Date"]).dt.tz_localize(None)
        retrieved_at = datetime.now(UTC).replace(tzinfo=None)
        rows: list[dict[str, object]] = []
        for _, row in actions.iterrows():
            if row.get("Dividends", 0.0):
                rows.append(
                    {
                        "source_vendor": "yfinance",
                        "symbol": symbol.upper(),
                        "action_type": "dividend",
                        "effective_date": row["Date"].date(),
                        "cash_amount": float(row["Dividends"]),
                        "split_ratio": None,
                        "split_from": None,
                        "split_to": None,
                        "retrieved_at": retrieved_at,
                    }
                )
            if row.get("Stock Splits", 0.0):
                split_ratio = float(row["Stock Splits"])
                rows.append(
                    {
                        "source_vendor": "yfinance",
                        "symbol": symbol.upper(),
                        "action_type": "split",
                        "effective_date": row["Date"].date(),
                        "cash_amount": None,
                        "split_ratio": split_ratio,
                        "split_from": 1.0,
                        "split_to": split_ratio,
                        "retrieved_at": retrieved_at,
                    }
                )
        return pd.DataFrame(rows)

    def get_corporate_actions(self, symbol: str) -> pd.DataFrame:
        actions = self._download_corporate_actions(symbol.upper())
        if not actions.empty:
            self.database.upsert_corporate_actions(actions, symbol.upper())
        return self.database.read_corporate_actions([symbol.upper()], limit_per_symbol=20)

    def get_metadata(self, symbol: str) -> dict:
        info = getattr(yf.Ticker(symbol), "fast_info", {}) or {}
        return {
            "symbol": symbol.upper(),
            "currency": info.get("currency"),
            "exchange": info.get("exchange"),
            "timezone": info.get("timezone"),
            "last_price": info.get("lastPrice"),
        }

    def get_last_fetch_status(self, symbol: str) -> CacheStatus | None:
        return self.last_fetch_status.get(symbol.upper())
