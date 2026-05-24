from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def build_demo_price_history(symbol: str, *, end_date: str, periods: int = 320) -> pd.DataFrame:
    end_ts = pd.Timestamp(end_date).normalize()
    index = pd.bdate_range(end=end_ts, periods=periods)
    seed = sum(ord(char) for char in symbol.upper())
    drift = ((seed % 9) - 4) * 0.00018
    wave = np.sin(np.arange(periods) / 17 + seed / 13) * 0.004
    pulse = np.cos(np.arange(periods) / 7 + seed / 9) * 0.002
    returns = drift + wave + pulse
    base_price = 75.0 + (seed % 40) * 3.5
    close = base_price * np.cumprod(1 + returns)
    open_price = close * (1 - 0.0015)
    high = close * 1.006
    low = close * 0.994
    frame = pd.DataFrame(
        {
            "source_vendor": "demo",
            "symbol": symbol.upper(),
            "timeframe": "1d",
            "timestamp": index,
            "session_date": index.date,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": np.linspace(900000, 1300000, periods),
            "dividends": 0.0,
            "stock_splits": 0.0,
            "adjusted_flag": False,
            "retrieved_at": pd.Timestamp.utcnow().tz_localize(None),
        }
    )
    return frame


def build_demo_price_histories(symbols: list[str], *, end_date: str, periods: int = 320) -> dict[str, pd.DataFrame]:
    return {symbol: build_demo_price_history(symbol, end_date=end_date, periods=periods) for symbol in symbols}


def build_demo_breadth_rows() -> list[dict[str, object]]:
    return [
        {
            "universe": "S&P 500",
            "pct_above_20dma": 63.0,
            "pct_above_50dma": 58.0,
            "pct_above_200dma": 54.0,
            "new_highs": 74,
            "new_lows": 21,
            "up_down_volume_signal": "Positive",
        },
        {
            "universe": "Nasdaq 100",
            "pct_above_20dma": 61.0,
            "pct_above_50dma": 56.0,
            "pct_above_200dma": 49.0,
            "new_highs": 39,
            "new_lows": 17,
            "up_down_volume_signal": "Positive",
        },
    ]


def build_demo_options_rows() -> list[dict[str, object]]:
    return [
        {
            "underlying": "SPX",
            "net_gamma_exposure_billion": 1.4,
            "gamma_flip_level": 5125.0,
            "largest_positive_gamma_strike": 5200.0,
            "largest_negative_gamma_strike": 5050.0,
            "dealer_regime": "Neutral",
            "historical_percentile": 57.0,
        }
    ]


def build_demo_macro_rows(as_of_date: date) -> list[dict[str, object]]:
    return [
        {
            "event_name": "Core PCE",
            "date": str(as_of_date + timedelta(days=5)),
            "prior": "2.8%",
            "consensus": "2.7%",
            "actual": "Not available",
            "market_importance": "High",
        },
        {
            "event_name": "Initial Jobless Claims",
            "date": str(as_of_date + timedelta(days=4)),
            "prior": "222K",
            "consensus": "225K",
            "actual": "Not available",
            "market_importance": "Medium",
        },
        {
            "event_name": "GDP Second Estimate",
            "date": str(as_of_date + timedelta(days=3)),
            "prior": "1.8%",
            "consensus": "1.9%",
            "actual": "Not available",
            "market_importance": "High",
        },
    ]


def build_demo_earnings_rows(as_of_date: date) -> list[dict[str, object]]:
    return [
        {
            "ticker": "NVDA",
            "company": "NVIDIA",
            "report_date": str(as_of_date + timedelta(days=4)),
            "consensus_eps": "5.61",
            "options_implied_move": "8.2%",
            "prediction_market_odds": "Not available",
            "importance": "Mega Cap",
        },
        {
            "ticker": "CRM",
            "company": "Salesforce",
            "report_date": str(as_of_date + timedelta(days=3)),
            "consensus_eps": "2.44",
            "options_implied_move": "6.1%",
            "prediction_market_odds": "Not available",
            "importance": "Large Cap",
        },
    ]
