from pathlib import Path

import pandas as pd

from trading_lab.data.market_calendar import MarketCalendar
from trading_lab.data.database import TradingLabDatabase
from trading_lab.data.providers.yfinance_provider import YFinanceDataProvider
from trading_lab.data.validation import validate_stock_bars


def test_yfinance_provider_normalizes_expected_columns(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "test.duckdb"))
    provider = YFinanceDataProvider(database=db)
    raw = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [11.0, 12.0],
            "Low": [9.5, 10.5],
            "Close": [10.5, 11.5],
            "Adj Close": [10.4, 11.4],
            "Volume": [1000, 1200],
            "Dividends": [0.0, 0.0],
            "Stock Splits": [0.0, 0.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    raw.index.name = "Date"

    result = provider._normalize_download("AAPL", "1d", raw)

    expected_columns = {
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
    }
    assert expected_columns.issubset(result.columns)
    assert result["symbol"].unique().tolist() == ["AAPL"]


def test_incremental_update_and_corporate_actions_are_persisted(tmp_path: Path, monkeypatch):
    db = TradingLabDatabase(str(tmp_path / "test.duckdb"))
    provider = YFinanceDataProvider(database=db)
    download_calls: list[pd.Timestamp] = []

    def fake_download(tickers, start, end, interval, auto_adjust, actions, progress, prepost=False):
        download_calls.append(pd.Timestamp(start))
        frame = pd.DataFrame(
            {
                "Open": [10.0, 11.0, 12.0],
                "High": [11.0, 12.0, 13.0],
                "Low": [9.0, 10.0, 11.0],
                "Close": [10.5, 11.5, 12.5],
                "Adj Close": [10.5, 11.5, 12.5],
                "Volume": [1000, 1200, 1300],
                "Dividends": [0.0, 0.5, 0.0],
                "Stock Splits": [0.0, 0.0, 2.0],
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        )
        frame.index.name = "Date"
        return frame[(frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(end))]

    class FakeTicker:
        fast_info = {}

        @property
        def actions(self):
            actions = pd.DataFrame(
                {
                    "Dividends": [0.5, 0.0],
                    "Stock Splits": [0.0, 2.0],
                },
                index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            )
            actions.index.name = "Date"
            return actions

    monkeypatch.setattr("trading_lab.data.providers.yfinance_provider.yf.download", lambda **kwargs: fake_download(**kwargs))
    monkeypatch.setattr("trading_lab.data.providers.yfinance_provider.yf.Ticker", lambda symbol: FakeTicker())

    first = provider.get_stock_bars("AAPL", "2024-01-01", "2024-01-02")
    second = provider.get_stock_bars("AAPL", "2024-01-01", "2024-01-03")

    assert len(first) == 2
    assert len(second) == 3
    assert len(download_calls) == 2
    assert download_calls[1] == pd.Timestamp("2024-01-02")

    actions = db.read_corporate_actions(["AAPL"], limit_per_symbol=10)
    assert set(actions["action_type"]) == {"dividend", "split"}
    assert (actions["symbol"] == "AAPL").all()


def test_market_calendar_expected_sessions_and_holiday_detection():
    calendar = MarketCalendar("NYSE")
    sessions = calendar.expected_sessions("2024-07-03", "2024-07-05")
    assert pd.Timestamp("2024-07-04").date() not in sessions
    assert pd.Timestamp("2024-07-03").date() in sessions
    assert pd.Timestamp("2024-07-05").date() in sessions


def test_validate_stock_bars_distinguishes_holiday_from_missing_session():
    calendar = MarketCalendar("NYSE")
    frame = pd.DataFrame(
        {
            "source_vendor": ["test", "test"],
            "symbol": ["AAA", "AAA"],
            "timeframe": ["1d", "1d"],
            "timestamp": pd.to_datetime(["2024-07-03", "2024-07-05"]),
            "session_date": [pd.Timestamp("2024-07-03").date(), pd.Timestamp("2024-07-05").date()],
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "adj_close": [10.5, 11.5],
            "volume": [1000.0, 1000.0],
            "dividends": [0.0, 0.0],
            "stock_splits": [0.0, 0.0],
            "adjusted_flag": [False, False],
            "retrieved_at": [pd.Timestamp("2024-07-06"), pd.Timestamp("2024-07-06")],
        }
    )
    holiday_result = validate_stock_bars(frame, market_calendar=calendar, start_date="2024-07-03", end_date="2024-07-05")
    assert holiday_result.warnings == []

    missing_frame = frame.copy()
    missing_frame.loc[1, "timestamp"] = pd.Timestamp("2024-07-08")
    missing_frame.loc[1, "session_date"] = pd.Timestamp("2024-07-08").date()
    missing_result = validate_stock_bars(missing_frame, market_calendar=calendar, start_date="2024-07-03", end_date="2024-07-08")
    assert missing_result.warnings


def test_intraday_range_clamp_and_regular_hours_filter(tmp_path: Path, monkeypatch):
    db = TradingLabDatabase(str(tmp_path / "intraday.duckdb"))
    provider = YFinanceDataProvider(database=db)

    def fake_download(**kwargs):
        index = pd.to_datetime(
            [
                "2024-05-20 08:00",
                "2024-05-20 09:30",
                "2024-05-20 09:45",
                "2024-05-20 16:15",
            ]
        ).tz_localize("America/New_York")
        frame = pd.DataFrame(
            {
                "Open": [10.0, 10.1, 10.2, 10.3],
                "High": [10.2, 10.3, 10.4, 10.5],
                "Low": [9.9, 10.0, 10.1, 10.2],
                "Close": [10.1, 10.2, 10.3, 10.4],
                "Adj Close": [10.1, 10.2, 10.3, 10.4],
                "Volume": [100, 200, 300, 400],
            },
            index=index,
        )
        frame.index.name = "Datetime"
        return frame

    class FakeTicker:
        fast_info = {}

        @property
        def actions(self):
            empty = pd.DataFrame(columns=["Dividends", "Stock Splits"])
            empty.index.name = "Date"
            return empty

    monkeypatch.setattr("trading_lab.data.providers.yfinance_provider.yf.download", fake_download)
    monkeypatch.setattr("trading_lab.data.providers.yfinance_provider.yf.Ticker", lambda symbol: FakeTicker())

    bars = provider.get_stock_bars("SPY", "2024-01-01", "2024-05-20", timeframe="15m")
    assert len(bars) == 2
    assert bars["timeframe"].eq("15m").all()
    status = provider.get_last_fetch_status("SPY")
    assert status is not None
    assert any("clamped" in warning.lower() for warning in status.validation_warnings)


def test_intraday_normalization_preserves_eastern_session_dates(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "normalize_intraday.duckdb"))
    provider = YFinanceDataProvider(database=db)
    raw = pd.DataFrame(
        {
            "Open": [10.0, 10.1],
            "High": [10.2, 10.3],
            "Low": [9.9, 10.0],
            "Close": [10.1, 10.2],
            "Adj Close": [10.1, 10.2],
            "Volume": [100, 200],
        },
        index=pd.to_datetime(["2024-05-20 09:30", "2024-05-20 09:45"]).tz_localize("America/New_York"),
    )
    raw.index.name = "Datetime"
    result = provider._normalize_download("SPY", "15m", raw)
    assert result["timestamp"].dt.tz is None
    assert result["session_date"].nunique() == 1
