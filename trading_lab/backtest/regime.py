from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_lab.backtest.metrics import compute_summary_metrics


@dataclass
class RegimeClassification:
    frame: pd.DataFrame
    volatility_median: float


def classify_market_regimes(benchmark_bars: pd.DataFrame, price_column: str = "close") -> RegimeClassification:
    """Classify benchmark dates into simple bull/bear and high/low volatility regimes."""
    frame = benchmark_bars.copy().sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        return RegimeClassification(frame=frame, volatility_median=0.0)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["benchmark_price"] = frame[price_column].astype(float)
    frame["sma_200"] = frame["benchmark_price"].rolling(200, min_periods=20).mean()
    frame["daily_return"] = frame["benchmark_price"].pct_change()
    frame["vol_20"] = frame["daily_return"].rolling(20, min_periods=10).std() * (252 ** 0.5)
    vol_median = float(frame["vol_20"].median(skipna=True)) if frame["vol_20"].notna().any() else 0.0
    frame["trend_regime"] = frame.apply(
        lambda row: "bull" if pd.notna(row["sma_200"]) and row["benchmark_price"] >= row["sma_200"] else "bear",
        axis=1,
    )
    frame["vol_regime"] = frame.apply(
        lambda row: "high_vol" if pd.notna(row["vol_20"]) and row["vol_20"] >= vol_median else "low_vol",
        axis=1,
    )
    return RegimeClassification(frame=frame, volatility_median=vol_median)


def compute_regime_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    initial_capital: float,
    benchmark_curve: pd.DataFrame | None = None,
    price_column: str = "close",
) -> pd.DataFrame:
    classification = classify_market_regimes(benchmark_bars, price_column=price_column)
    regime_frame = classification.frame
    if equity_curve.empty or regime_frame.empty:
        return pd.DataFrame()

    equity = equity_curve.copy()
    equity["timestamp"] = pd.to_datetime(equity["timestamp"])
    regime_frame = regime_frame[["timestamp", "trend_regime", "vol_regime"]].copy()
    merged = equity.merge(regime_frame, on="timestamp", how="left").ffill()

    trade_frame = trades.copy()
    if not trade_frame.empty:
        trade_frame["entry_timestamp"] = pd.to_datetime(trade_frame["entry_timestamp"])
        trade_frame = trade_frame.merge(regime_frame, left_on="entry_timestamp", right_on="timestamp", how="left").drop(columns=["timestamp"], errors="ignore")

    rows: list[dict[str, object]] = []
    for regime_type, column in [("trend", "trend_regime"), ("volatility", "vol_regime")]:
        for regime_name, subset in merged.groupby(column):
            if subset.empty or pd.isna(regime_name):
                continue
            trade_subset = trade_frame.loc[trade_frame[column] == regime_name].copy() if not trade_frame.empty and column in trade_frame.columns else pd.DataFrame()
            metrics = compute_summary_metrics(subset, trade_subset, initial_capital, benchmark_curve=benchmark_curve)
            rows.append(
                {
                    "regime_type": regime_type,
                    "regime_name": regime_name,
                    "total_return": metrics.get("Total Return", 0.0),
                    "cagr": metrics.get("CAGR", 0.0),
                    "max_drawdown": metrics.get("Max Drawdown", 0.0),
                    "sharpe_ratio": metrics.get("Sharpe Ratio", 0.0),
                    "sortino_ratio": metrics.get("Sortino Ratio", 0.0),
                    "calmar_ratio": metrics.get("Calmar Ratio", 0.0),
                    "number_of_trades": metrics.get("Number of Trades", 0),
                    "win_rate": metrics.get("Win Rate", 0.0),
                    "profit_factor": metrics.get("Profit Factor", 0.0),
                    "average_trade_return": metrics.get("Average Trade Return", 0.0),
                }
            )
    return pd.DataFrame(rows)


def summarize_regime_comments(regime_metrics: pd.DataFrame) -> list[str]:
    comments: list[str] = []
    if regime_metrics.empty:
        return ["Regime analysis could not be computed because benchmark or equity data was insufficient."]

    trend = regime_metrics[regime_metrics["regime_type"] == "trend"].set_index("regime_name")
    if {"bull", "bear"}.issubset(trend.index):
        bull = float(trend.loc["bull", "cagr"])
        bear = float(trend.loc["bear", "cagr"])
        bull_trades = int(trend.loc["bull", "number_of_trades"])
        bear_trades = int(trend.loc["bear", "number_of_trades"])
        if min(bull_trades, bear_trades) < 5:
            comments.append("Trade count is too low to trust regime conclusions.")
        elif bear < bull - 0.1:
            comments.append("Strategy performs materially worse in bear regimes.")
        elif bull < bear - 0.1:
            comments.append("Strategy performs materially worse in bull regimes.")

    vol = regime_metrics[regime_metrics["regime_type"] == "volatility"].set_index("regime_name")
    if {"high_vol", "low_vol"}.issubset(vol.index):
        high = float(vol.loc["high_vol", "total_return"])
        low = float(vol.loc["low_vol", "total_return"])
        if low > high * 1.5 and low > 0:
            comments.append("Most returns occur during low-volatility periods.")
        elif high > low * 1.5 and high > 0:
            comments.append("Most returns occur during high-volatility periods.")

    if not comments:
        comments.append("No strong regime dependence was detected from the current sample.")
    return comments
