from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from pybroker import indicator, model
from pybroker.common import BarData
from pybroker.scope import StaticScope


def ensure_indicator(name: str, fn: Callable[..., np.ndarray], **kwargs):
    scope = StaticScope.instance()
    if scope.has_indicator(name):
        return scope.get_indicator(name)
    return indicator(name, fn, **kwargs)


def ensure_model(name: str, fn: Callable[..., object], **kwargs):
    scope = StaticScope.instance()
    if scope.has_model_source(name):
        return scope.get_model_source(name)
    return model(name, fn, **kwargs)


def _series(values: np.ndarray) -> pd.Series:
    return pd.Series(values, dtype=float)


def rolling_mean(values: np.ndarray, period: int) -> np.ndarray:
    return _series(values).rolling(period).mean().to_numpy()


def rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    return _series(values).rolling(period).std(ddof=0).to_numpy()


def pct_change(values: np.ndarray, period: int) -> np.ndarray:
    return _series(values).pct_change(period).to_numpy()


def realized_volatility(values: np.ndarray, period: int) -> np.ndarray:
    returns = _series(values).pct_change()
    return returns.rolling(period).std(ddof=0).mul(np.sqrt(252)).to_numpy()


def rsi_values(values: np.ndarray, period: int) -> np.ndarray:
    delta = _series(values).diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0).to_numpy()


def drawdown_from_high(values: np.ndarray, period: int) -> np.ndarray:
    series = _series(values)
    rolling_high = series.rolling(period).max()
    return (series / rolling_high - 1.0).to_numpy()


def volume_change(values: np.ndarray, period: int) -> np.ndarray:
    return _series(values).pct_change(period).to_numpy()


def close_sma_distance(values: np.ndarray, period: int) -> np.ndarray:
    sma = rolling_mean(values, period)
    return ((values - sma) / sma).astype(float)


def make_sma_indicator(name: str, period: int):
    return ensure_indicator(name, lambda data: rolling_mean(data.close, period))


def make_high_indicator(name: str, period: int):
    return ensure_indicator(name, lambda data: _series(data.close).rolling(period).max().to_numpy())


def make_bollinger_lower_indicator(name: str, period: int, std_dev: float):
    def _bollinger(data: BarData) -> np.ndarray:
        mean = rolling_mean(data.close, period)
        std = rolling_std(data.close, period)
        return mean - std_dev * std

    return ensure_indicator(name, _bollinger)


def make_rsi_indicator(name: str, period: int):
    return ensure_indicator(name, lambda data: rsi_values(data.close, period))


def make_vol_indicator(name: str, period: int):
    return ensure_indicator(name, lambda data: realized_volatility(data.close, period))


def make_return_indicator(name: str, period: int):
    return ensure_indicator(name, lambda data: pct_change(data.close, period))


def make_sma_distance_indicator(name: str, period: int):
    return ensure_indicator(name, lambda data: close_sma_distance(data.close, period))


def make_volume_change_indicator(name: str, period: int):
    return ensure_indicator(name, lambda data: volume_change(data.volume, period))


def make_drawdown_indicator(name: str, period: int):
    return ensure_indicator(name, lambda data: drawdown_from_high(data.close, period))
