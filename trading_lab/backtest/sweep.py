from __future__ import annotations

import json
from itertools import product
from uuid import uuid4

import pandas as pd

from trading_lab.backtest.audit import generate_strategy_audit
from trading_lab.backtest.engine import BacktestConfig, BacktestEngine
from trading_lab.backtest.robustness import parameter_stability_summary


def expand_parameter_grid(param_grid: dict[str, list[int | float]]) -> list[dict[str, int | float]]:
    keys = list(param_grid.keys())
    if not keys:
        return [{}]
    return [dict(zip(keys, values, strict=False)) for values in product(*[param_grid[key] for key in keys])]


def run_parameter_sweep(
    engine: BacktestEngine,
    strategy_builder,
    data_by_symbol: dict[str, pd.DataFrame],
    config: BacktestConfig,
    param_grid: dict[str, list[int | float]],
    benchmark_symbol: str,
    sort_metric: str = "CAGR",
    *,
    strategy_name: str | None = None,
    notes: str = "",
    tags: str = "",
    sweep_context: dict[str, object] | None = None,
) -> tuple[str, pd.DataFrame]:
    sweep_id = str(uuid4())
    rows: list[dict] = []
    for params in expand_parameter_grid(param_grid):
        strategy = strategy_builder(params)
        sweep_config = config.model_copy(update={"sweep_id": sweep_id})
        result = engine.run(data_by_symbol=data_by_symbol, strategy=strategy, config=sweep_config, benchmark_symbol=benchmark_symbol)
        metrics = result.metrics
        rows.append(
            {
                "sweep_id": sweep_id,
                "run_id": result.run_id,
                "strategy_name": strategy.name,
                "parameters_json": params,
                "CAGR": metrics.get("CAGR", 0.0),
                "Sharpe Ratio": metrics.get("Sharpe Ratio", 0.0),
                "Sortino Ratio": metrics.get("Sortino Ratio", 0.0),
                "Max Drawdown": metrics.get("Max Drawdown", 0.0),
                "Calmar Ratio": metrics.get("Calmar Ratio", 0.0),
                "Total Return": metrics.get("Total Return", 0.0),
                "Profit Factor": metrics.get("Profit Factor", 0.0),
                "Win Rate": metrics.get("Win Rate", 0.0),
                "Number of Trades": metrics.get("Number of Trades", 0),
                "Exposure %": metrics.get("Exposure %", 0.0),
                "Excess CAGR": metrics.get("Excess CAGR", 0.0),
                "audit_summary": " ".join(generate_strategy_audit(metrics, result.trade_log, result.equity_curve, strategy_parameters=params)),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty and sort_metric in frame.columns:
        ascending = sort_metric == "Max Drawdown"
        frame = frame.sort_values(sort_metric, ascending=ascending).reset_index(drop=True)
    if engine.database is not None:
        engine.database.replace_sweep_run(
            _build_sweep_run_payload(
                sweep_id=sweep_id,
                data_by_symbol=data_by_symbol,
                config=config,
                benchmark_symbol=benchmark_symbol,
                strategy_name=strategy_name or (frame.iloc[0]["strategy_name"] if not frame.empty else "strategy"),
                notes=notes,
                tags=tags,
                param_grid=param_grid,
                sweep_context=sweep_context,
            ),
            _build_sweep_result_frame(sweep_id, frame),
            _build_sweep_parameter_frame(sweep_id, param_grid),
        )
    return sweep_id, frame


def summarize_parameter_stability(sweep_results: pd.DataFrame, drawdown_threshold: float = -0.25) -> dict[str, object]:
    return parameter_stability_summary(sweep_results, drawdown_threshold=drawdown_threshold)


def _build_sweep_run_payload(
    *,
    sweep_id: str,
    data_by_symbol: dict[str, pd.DataFrame],
    config: BacktestConfig,
    benchmark_symbol: str,
    strategy_name: str,
    notes: str,
    tags: str,
    param_grid: dict[str, list[int | float]],
    sweep_context: dict[str, object] | None,
) -> dict[str, object]:
    non_benchmark_symbols = [symbol for symbol in data_by_symbol if symbol != benchmark_symbol]
    symbol_frames = [data_by_symbol[symbol] for symbol in non_benchmark_symbols if not data_by_symbol[symbol].empty]
    start_date = min(pd.to_datetime(frame["timestamp"]).min().date() for frame in symbol_frames) if symbol_frames else None
    end_date = max(pd.to_datetime(frame["timestamp"]).max().date() for frame in symbol_frames) if symbol_frames else None
    risk_settings = {
        "slippage_pct": config.slippage_pct,
        "commission_per_trade": config.commission_per_trade,
        "stop_loss_pct": config.stop_loss_pct,
        "take_profit_pct": config.take_profit_pct,
        "trailing_stop_pct": config.trailing_stop_pct,
        "return_mode": config.return_mode,
    }
    return {
        "sweep_id": sweep_id,
        "created_at": pd.Timestamp.now("UTC").tz_localize(None),
        "strategy_name": strategy_name,
        "tickers": ",".join(non_benchmark_symbols),
        "start_date": start_date,
        "end_date": end_date,
        "benchmark_symbol": benchmark_symbol,
        "initial_capital": config.initial_capital,
        "price_mode": config.price_mode,
        "position_sizing_method": config.position_sizing_method,
        "risk_settings_json": json.dumps(risk_settings),
        "sweep_config_json": json.dumps({"param_grid": param_grid, **(sweep_context or {})}),
        "notes": notes,
        "tags": tags,
    }


def _build_sweep_result_frame(sweep_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "sweep_result_id",
                "sweep_id",
                "backtest_run_id",
                "parameter_json",
                "total_return",
                "cagr",
                "max_drawdown",
                "sharpe",
                "sortino",
                "calmar",
                "win_rate",
                "profit_factor",
                "number_of_trades",
                "exposure_pct",
                "robustness_score",
                "beats_benchmark_flag",
                "created_at",
            ]
        )
    result_frame = frame.copy()
    result_frame["sweep_result_id"] = [str(uuid4()) for _ in range(len(result_frame))]
    result_frame["backtest_run_id"] = result_frame["run_id"]
    result_frame["parameter_json"] = result_frame["parameters_json"].apply(json.dumps)
    result_frame["robustness_score"] = None
    result_frame["beats_benchmark_flag"] = result_frame["Excess CAGR"].fillna(0.0) > 0
    result_frame["created_at"] = pd.Timestamp.now("UTC").tz_localize(None)
    return result_frame.rename(
        columns={
            "Total Return": "total_return",
            "CAGR": "cagr",
            "Max Drawdown": "max_drawdown",
            "Sharpe Ratio": "sharpe",
            "Sortino Ratio": "sortino",
            "Calmar Ratio": "calmar",
            "Win Rate": "win_rate",
            "Profit Factor": "profit_factor",
            "Number of Trades": "number_of_trades",
            "Exposure %": "exposure_pct",
        }
    )[
        [
            "sweep_result_id",
            "sweep_id",
            "backtest_run_id",
            "parameter_json",
            "total_return",
            "cagr",
            "max_drawdown",
            "sharpe",
            "sortino",
            "calmar",
            "win_rate",
            "profit_factor",
            "number_of_trades",
            "exposure_pct",
            "robustness_score",
            "beats_benchmark_flag",
            "created_at",
        ]
    ]


def _build_sweep_parameter_frame(sweep_id: str, param_grid: dict[str, list[int | float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"sweep_id": sweep_id, "parameter_name": name, "parameter_values_json": json.dumps(values)} for name, values in param_grid.items()]
    )
