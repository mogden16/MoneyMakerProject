from __future__ import annotations

import pandas as pd

from trading_lab.backtest.engine import BacktestConfig, BacktestEngine
from trading_lab.backtest.metrics import compute_summary_metrics


def split_data_by_date(data_by_symbol: dict[str, pd.DataFrame], split_date: str) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    split_ts = pd.Timestamp(split_date)
    train: dict[str, pd.DataFrame] = {}
    test: dict[str, pd.DataFrame] = {}
    for symbol, frame in data_by_symbol.items():
        ts = pd.to_datetime(frame["timestamp"])
        train[symbol] = frame.loc[ts < split_ts].reset_index(drop=True)
        test[symbol] = frame.loc[ts >= split_ts].reset_index(drop=True)
    return train, test


def split_data_by_percentage(data_by_symbol: dict[str, pd.DataFrame], split_pct: float) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    train: dict[str, pd.DataFrame] = {}
    test: dict[str, pd.DataFrame] = {}
    for symbol, frame in data_by_symbol.items():
        cutoff = max(1, min(len(frame) - 1, int(len(frame) * split_pct)))
        train[symbol] = frame.iloc[:cutoff].reset_index(drop=True)
        test[symbol] = frame.iloc[cutoff:].reset_index(drop=True)
    return train, test


def run_train_test_analysis(
    engine: BacktestEngine,
    strategy,
    config: BacktestConfig,
    train_data: dict[str, pd.DataFrame],
    test_data: dict[str, pd.DataFrame],
    benchmark_symbol: str,
) -> dict[str, object]:
    if any(frame.empty for frame in train_data.values()) or any(frame.empty for frame in test_data.values()):
        raise ValueError("Train/test split produced an empty dataset for at least one symbol.")
    stateless_engine = BacktestEngine(database=None)
    train_result = stateless_engine.run(train_data, strategy, config, benchmark_symbol=benchmark_symbol)
    test_result = stateless_engine.run(test_data, strategy, config, benchmark_symbol=benchmark_symbol)
    train_metrics = compute_summary_metrics(train_result.equity_curve, train_result.trade_log, config.initial_capital, benchmark_curve=train_result.benchmark_curve)
    test_metrics = compute_summary_metrics(test_result.equity_curve, test_result.trade_log, config.initial_capital, benchmark_curve=test_result.benchmark_curve)
    return {
        "train_result": train_result,
        "test_result": test_result,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "degradation": {
            "CAGR": float(test_metrics.get("CAGR", 0.0)) - float(train_metrics.get("CAGR", 0.0)),
            "Sharpe Ratio": float(test_metrics.get("Sharpe Ratio", 0.0)) - float(train_metrics.get("Sharpe Ratio", 0.0)),
            "Max Drawdown": float(test_metrics.get("Max Drawdown", 0.0)) - float(train_metrics.get("Max Drawdown", 0.0)),
        },
    }
