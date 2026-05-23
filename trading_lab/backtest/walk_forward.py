from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pandas as pd

from trading_lab.backtest.engine import BacktestConfig, BacktestEngine


@dataclass
class WalkForwardFold:
    fold_number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_walk_forward_folds(
    start_date: str,
    end_date: str,
    train_window_months: int,
    test_window_months: int,
    step_months: int,
) -> list[WalkForwardFold]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    folds: list[WalkForwardFold] = []
    fold_number = 1
    cursor = start
    while True:
        train_end = cursor + pd.DateOffset(months=train_window_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_window_months) - pd.Timedelta(days=1)
        if test_end > end:
            break
        folds.append(
            WalkForwardFold(
                fold_number=fold_number,
                train_start=cursor,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_number += 1
        cursor = cursor + pd.DateOffset(months=step_months)
    return folds


def _slice_data(data_by_symbol: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    sliced: dict[str, pd.DataFrame] = {}
    for symbol, frame in data_by_symbol.items():
        ts = pd.to_datetime(frame["timestamp"])
        sliced[symbol] = frame.loc[(ts >= start) & (ts <= end)].reset_index(drop=True)
    return sliced


def run_walk_forward_analysis(
    engine: BacktestEngine,
    strategy,
    config: BacktestConfig,
    data_by_symbol: dict[str, pd.DataFrame],
    benchmark_symbol: str,
    train_window_months: int,
    test_window_months: int,
    step_months: int,
    min_train_trades: int = 0,
    min_test_trades: int = 0,
) -> tuple[str, pd.DataFrame, dict[str, float | int]]:
    timestamps = pd.concat([pd.to_datetime(frame["timestamp"]) for frame in data_by_symbol.values()]).sort_values()
    folds = generate_walk_forward_folds(str(timestamps.min().date()), str(timestamps.max().date()), train_window_months, test_window_months, step_months)
    stateless_engine = BacktestEngine(database=None)
    rows: list[dict[str, object]] = []
    walk_forward_id = str(uuid4())
    for fold in folds:
        train_data = _slice_data(data_by_symbol, fold.train_start, fold.train_end)
        test_data = _slice_data(data_by_symbol, fold.test_start, fold.test_end)
        if any(frame.empty for frame in train_data.values()) or any(frame.empty for frame in test_data.values()):
            continue
        train_result = stateless_engine.run(train_data, strategy, config, benchmark_symbol=benchmark_symbol)
        test_result = stateless_engine.run(test_data, strategy, config, benchmark_symbol=benchmark_symbol)
        train_metrics = train_result.metrics
        test_metrics = test_result.metrics
        if int(train_metrics.get("Number of Trades", 0)) < min_train_trades or int(test_metrics.get("Number of Trades", 0)) < min_test_trades:
            continue
        degradation_score = float(test_metrics.get("CAGR", 0.0)) - float(train_metrics.get("CAGR", 0.0))
        rows.append(
            {
                "walk_forward_id": walk_forward_id,
                "fold_number": fold.fold_number,
                "train_start": fold.train_start.date(),
                "train_end": fold.train_end.date(),
                "test_start": fold.test_start.date(),
                "test_end": fold.test_end.date(),
                "train_cagr": train_metrics.get("CAGR", 0.0),
                "test_cagr": test_metrics.get("CAGR", 0.0),
                "train_max_drawdown": train_metrics.get("Max Drawdown", 0.0),
                "test_max_drawdown": test_metrics.get("Max Drawdown", 0.0),
                "train_sharpe": train_metrics.get("Sharpe Ratio", 0.0),
                "test_sharpe": test_metrics.get("Sharpe Ratio", 0.0),
                "train_trades": train_metrics.get("Number of Trades", 0),
                "test_trades": test_metrics.get("Number of Trades", 0),
                "degradation_score": degradation_score,
            }
        )
    fold_frame = pd.DataFrame(rows)
    if fold_frame.empty:
        return walk_forward_id, fold_frame, {
            "average_test_cagr": 0.0,
            "median_test_cagr": 0.0,
            "profitable_test_fold_pct": 0.0,
            "worst_test_fold": 0.0,
            "best_test_fold": 0.0,
            "average_train_to_test_degradation": 0.0,
            "consistency_score": 0.0,
        }
    profitable_pct = float((fold_frame["test_cagr"] > 0).mean())
    consistency_score = max(0.0, 1.0 - float(fold_frame["test_cagr"].std(ddof=0) if len(fold_frame) > 1 else 0.0))
    summary = {
        "average_test_cagr": float(fold_frame["test_cagr"].mean()),
        "median_test_cagr": float(fold_frame["test_cagr"].median()),
        "profitable_test_fold_pct": profitable_pct,
        "worst_test_fold": float(fold_frame["test_cagr"].min()),
        "best_test_fold": float(fold_frame["test_cagr"].max()),
        "average_train_to_test_degradation": float(fold_frame["degradation_score"].mean()),
        "consistency_score": consistency_score,
    }
    return walk_forward_id, fold_frame, summary
