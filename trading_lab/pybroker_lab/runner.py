from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pybroker import Strategy, StrategyConfig
from pybroker.common import FeeInfo
from pybroker.slippage import SlippageModel
from pybroker.strategy import WalkforwardWindow

from trading_lab.pybroker_lab.audit import (
    AUDIT_COLUMNS,
    build_actual_data_used_frame,
    build_benchmark_summary_frame,
    build_data_quality_audit,
    build_higher_timeframe_check,
    build_indicator_debug_table,
    build_raw_bars_export_name,
    build_strategy_debug_frame,
    build_trade_audit_frame,
    filter_actual_data_row,
)
from trading_lab.pybroker_lab.benchmarks import build_buy_and_hold_curve
from trading_lab.pybroker_lab.config import PyBrokerLabConfig, PyBrokerStrategyDefinition
from trading_lab.pybroker_lab.data import load_market_data, symbol_bars
from trading_lab.pybroker_lab.metrics import (
    bootstrap_reliability_note,
    bootstrap_to_frame,
    build_delta_metrics,
    compute_equity_metrics,
    decision_to_row,
    evaluate_strategy,
    normalize_portfolio_curve,
    normalize_trade_log,
)
from trading_lab.pybroker_lab.reporting import write_csv, write_report_markdown
from trading_lab.pybroker_lab.strategy_registry import strategy_registry


class FixedBpsSlippage(SlippageModel):
    def __init__(self, slippage_bps: float):
        self._multiplier = slippage_bps / 10000.0

    def apply_slippage(self, ctx, buy_shares=None, sell_shares=None):
        if self._multiplier <= 0:
            return
        if buy_shares:
            ctx.buy_fill_price = float(ctx.close[-1]) * (1.0 + self._multiplier)
        if sell_shares:
            ctx.sell_fill_price = float(ctx.close[-1]) * (1.0 - self._multiplier)


@dataclass(frozen=True)
class PyBrokerRunResult:
    output_dir: Path
    summary: pd.DataFrame
    strategy_metrics: pd.DataFrame
    benchmark_metrics: pd.DataFrame
    actual_data_used: pd.DataFrame
    benchmark_summary: pd.DataFrame
    trade_audit: pd.DataFrame
    indicator_debug_tables: dict[str, pd.DataFrame]
    data_quality_audits: dict[str, pd.DataFrame]
    higher_timeframe_checks: dict[str, pd.DataFrame]
    actual_bars: dict[str, pd.DataFrame]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    bootstrap_metrics: pd.DataFrame
    walkforward_windows: pd.DataFrame
    debug_frames: dict[str, pd.DataFrame]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PyBroker research lab.")
    parser.add_argument("--strategy", default="all", choices=["all", *strategy_registry().keys()])
    parser.add_argument("--start", dest="start_date", default="2010-01-01")
    parser.add_argument("--end", dest="end_date", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument("--symbols", nargs="*", default=["SPY"])
    parser.add_argument("--benchmark-symbol", default="SPY")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--output-dir", default="outputs/pybroker_lab")
    return parser


def run_pybroker_lab(
    config: PyBrokerLabConfig,
    *,
    strategy_name: str = "all",
    data_frame: pd.DataFrame | None = None,
    statuses: list[Any] | None = None,
) -> PyBrokerRunResult:
    registry = strategy_registry()
    strategy_names = list(registry) if strategy_name == "all" else [strategy_name]
    definitions = [registry[name](config) for name in strategy_names]
    extra_symbols = {symbol for definition in definitions for symbol in definition.symbols}
    market_data = load_market_data(config, data_frame=data_frame, extra_symbols=extra_symbols)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    strategy_metric_rows: list[dict[str, Any]] = []
    benchmark_metric_rows: list[dict[str, Any]] = []
    actual_data_frames: list[pd.DataFrame] = []
    benchmark_summary_frames: list[pd.DataFrame] = []
    trade_audit_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    window_rows: list[dict[str, Any]] = []
    debug_frames: dict[str, pd.DataFrame] = {}
    indicator_debug_tables: dict[str, pd.DataFrame] = {}
    data_quality_audits: dict[str, pd.DataFrame] = {}
    higher_timeframe_checks: dict[str, pd.DataFrame] = {}
    actual_bars: dict[str, pd.DataFrame] = {}

    for definition in definitions:
        payload = _run_single_strategy(config=config, market_data=market_data, definition=definition, statuses=statuses or [])
        summary_rows.append(payload["summary"])
        strategy_metric_rows.append(payload["strategy_metrics"])
        benchmark_metric_rows.append(payload["benchmark_metrics"])
        actual_data_frames.append(payload["actual_data_used"])
        benchmark_summary_frames.append(payload["benchmark_summary"])
        trade_audit_frames.append(payload["trade_audit"])
        trade_frames.append(payload["trades"])
        equity_frames.extend([payload["strategy_curve"], payload["benchmark_curve"]])
        bootstrap_frames.append(payload["bootstrap"])
        window_rows.extend(payload["walkforward_rows"])
        debug_frames[definition.name] = payload["debug_frame"]
        indicator_debug_tables[definition.name] = payload["indicator_debug_table"]
        data_quality_audits[definition.name] = payload["data_quality_audit"]
        higher_timeframe_checks[definition.name] = payload["higher_timeframe_check"]
        actual_bars[definition.name] = payload["actual_bars"]
        raw_bars_filename = f"{definition.name}_{build_raw_bars_export_name(definition.symbols[0], config.timeframe, payload['actual_bars'])}"
        write_csv(payload["actual_bars"], output_dir / raw_bars_filename)

    summary = pd.DataFrame(summary_rows)
    strategy_metrics = pd.DataFrame(strategy_metric_rows)
    benchmark_metrics = pd.DataFrame(benchmark_metric_rows)
    non_empty_actual_data = [frame for frame in actual_data_frames if not frame.empty]
    non_empty_benchmark_summary = [frame for frame in benchmark_summary_frames if not frame.empty]
    non_empty_trade_audit = [frame for frame in trade_audit_frames if not frame.empty]
    actual_data_used = pd.concat(non_empty_actual_data, ignore_index=True) if non_empty_actual_data else pd.DataFrame()
    benchmark_summary = pd.concat(non_empty_benchmark_summary, ignore_index=True) if non_empty_benchmark_summary else pd.DataFrame()
    trade_audit = pd.concat(non_empty_trade_audit, ignore_index=True) if non_empty_trade_audit else pd.DataFrame(columns=AUDIT_COLUMNS)
    non_empty_trades = [frame for frame in trade_frames if not frame.empty]
    non_empty_equity = [frame for frame in equity_frames if not frame.empty]
    non_empty_bootstrap = [frame for frame in bootstrap_frames if not frame.empty]
    trades = pd.concat(non_empty_trades, ignore_index=True) if non_empty_trades else pd.DataFrame()
    equity_curve = pd.concat(non_empty_equity, ignore_index=True) if non_empty_equity else pd.DataFrame()
    bootstrap_metrics = pd.concat(non_empty_bootstrap, ignore_index=True) if non_empty_bootstrap else pd.DataFrame()
    walkforward_windows = pd.DataFrame(window_rows)

    write_csv(summary, output_dir / "summary.csv")
    write_csv(strategy_metrics, output_dir / "strategy_metrics.csv")
    write_csv(benchmark_metrics, output_dir / "benchmark_metrics.csv")
    write_csv(actual_data_used, output_dir / "actual_data_used.csv")
    write_csv(benchmark_summary, output_dir / "benchmark_summary.csv")
    write_csv(trade_audit, output_dir / "trade_audit.csv")
    write_csv(trades, output_dir / "trades.csv")
    write_csv(equity_curve, output_dir / "equity_curve.csv")
    write_csv(bootstrap_metrics, output_dir / "bootstrap_metrics.csv")
    write_csv(walkforward_windows, output_dir / "walkforward_windows.csv")
    write_report_markdown(
        path=output_dir / "report.md",
        config=asdict(config),
        summary=summary,
        strategy_metrics=strategy_metrics,
        benchmark_metrics=benchmark_metrics,
        bootstrap_metrics=bootstrap_metrics,
        walkforward_windows=walkforward_windows,
    )
    return PyBrokerRunResult(
        output_dir=output_dir,
        summary=summary,
        strategy_metrics=strategy_metrics,
        benchmark_metrics=benchmark_metrics,
        actual_data_used=actual_data_used,
        benchmark_summary=benchmark_summary,
        trade_audit=trade_audit,
        indicator_debug_tables=indicator_debug_tables,
        data_quality_audits=data_quality_audits,
        higher_timeframe_checks=higher_timeframe_checks,
        actual_bars=actual_bars,
        trades=trades,
        equity_curve=equity_curve,
        bootstrap_metrics=bootstrap_metrics,
        walkforward_windows=walkforward_windows,
        debug_frames=debug_frames,
    )


def _run_single_strategy(
    *,
    config: PyBrokerLabConfig,
    market_data: pd.DataFrame,
    definition: PyBrokerStrategyDefinition,
    statuses: list[Any],
) -> dict[str, Any]:
    bars = market_data[market_data["symbol"].isin(definition.symbols)].copy().reset_index(drop=True)
    if bars.empty:
        raise ValueError(f"No market data was available for {definition.name}.")
    result = _run_walkforward(config=config, definition=definition, frame=bars)
    debug_frame = build_strategy_debug_frame(definition.name, bars=bars, config=config)
    strategy_curve = normalize_portfolio_curve(result.portfolio, strategy_name=definition.name, curve_type="strategy")
    trades = normalize_trade_log(result.trades, strategy_name=definition.name)
    benchmark_symbol = definition.symbols[0] if len(definition.symbols) == 1 else config.benchmark_symbol
    benchmark_prices = symbol_bars(market_data, benchmark_symbol)
    benchmark_curve = _benchmark_for_dates(
        benchmark_prices,
        strategy_curve["date"] if not strategy_curve.empty else benchmark_prices["date"],
        initial_cash=config.initial_cash,
        strategy_name=definition.name,
    )
    strategy_metrics = compute_equity_metrics(
        equity_curve=strategy_curve,
        trades=trades,
        initial_cash=config.initial_cash,
        strategy_name=definition.name,
    )
    benchmark_metrics = compute_equity_metrics(
        equity_curve=benchmark_curve,
        trades=pd.DataFrame(),
        initial_cash=config.initial_cash,
        strategy_name=definition.name,
    )
    actual_data_used = build_actual_data_used_frame(
        market_data=market_data[market_data["symbol"].isin(tuple(dict.fromkeys([*definition.symbols, benchmark_symbol])))].copy(),
        config=config,
        statuses=statuses,
        symbols=tuple(dict.fromkeys([*definition.symbols, benchmark_symbol])),
        provider_name=definition.data_source_name,
    )
    if not actual_data_used.empty:
        actual_data_used.insert(0, "strategy_id", definition.name)
    actual_data_row = filter_actual_data_row(actual_data_used, definition.symbols[0])
    trade_audit = build_trade_audit_frame(
        trades=trades,
        strategy_id=definition.name,
        bars=bars,
        config=config,
    )
    indicator_debug_table = build_indicator_debug_table(trade_audit)
    data_quality_audit = build_data_quality_audit(
        bars=bars,
        strategy_id=definition.name,
        config=config,
        actual_data_row=actual_data_row,
    )
    higher_timeframe_check = build_higher_timeframe_check(definition.name, debug_frame)
    benchmark_summary = build_benchmark_summary_frame(
        strategy_metrics=strategy_metrics,
        benchmark_metrics=benchmark_metrics,
        actual_data_row=actual_data_row,
        sizing_method=config.sizing_method,
        sizing_value=config.sizing_value,
    )
    delta_metrics = build_delta_metrics(strategy_metrics, benchmark_metrics)
    decision = evaluate_strategy(
        strategy_metrics=strategy_metrics,
        benchmark_metrics=benchmark_metrics,
        cagr_tolerance=config.materially_worse_cagr_threshold,
    )
    bootstrap = bootstrap_to_frame(result.bootstrap, strategy_name=definition.name)
    walkforward_rows = _walkforward_window_rows(config=config, definition=definition, frame=bars, benchmark_prices=benchmark_prices)
    summary = {
        "strategy_name": definition.name,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "actual_first_bar_timestamp": None if bars.empty else pd.to_datetime(bars["date"].min()),
        "actual_last_bar_timestamp": None if bars.empty else pd.to_datetime(bars["date"].max()),
        "bar_count": int(len(bars)),
        "timeframe": config.timeframe,
        "sizing_method": config.sizing_method,
        "sizing_value": config.sizing_value,
        "data_source": definition.data_source_name,
        "assumptions": " | ".join(definition.assumptions),
        **delta_metrics,
        **decision_to_row(decision, strategy_name=definition.name),
        "bootstrap_note": bootstrap_reliability_note(result.bootstrap),
    }
    return {
        "summary": summary,
        "strategy_metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics,
        "actual_data_used": actual_data_used,
        "benchmark_summary": benchmark_summary,
        "trade_audit": trade_audit,
        "indicator_debug_table": indicator_debug_table,
        "data_quality_audit": data_quality_audit,
        "higher_timeframe_check": higher_timeframe_check,
        "actual_bars": bars.copy().reset_index(drop=True),
        "trades": trades,
        "strategy_curve": strategy_curve,
        "benchmark_curve": benchmark_curve,
        "bootstrap": bootstrap,
        "walkforward_rows": walkforward_rows,
        "debug_frame": debug_frame,
    }


def _run_walkforward(*, config: PyBrokerLabConfig, definition: PyBrokerStrategyDefinition, frame: pd.DataFrame):
    strategy = Strategy(frame, config.start_date, config.end_date, config=_strategy_config(config, definition))
    strategy.add_execution(definition.execution, definition.symbols, indicators=definition.indicators, models=definition.models)
    strategy.set_slippage_model(FixedBpsSlippage(config.slippage_bps))
    return strategy.walkforward(
        windows=config.walkforward_windows,
        lookahead=definition.lookahead,
        train_size=config.train_size,
        warmup=min(config.warmup_bars, max(frame["date"].nunique() - 2, 1)),
        calc_bootstrap=True,
        disable_parallel=True,
    )


def _strategy_config(config: PyBrokerLabConfig, definition: PyBrokerStrategyDefinition) -> StrategyConfig:
    def fee_fn(info: FeeInfo):
        return float(info.shares) * float(info.fill_price) * config.commission_bps / 10000.0

    return StrategyConfig(
        initial_cash=config.initial_cash,
        fee_mode=fee_fn,
        enable_fractional_shares=True,
        bootstrap_sample_size=config.bootstrap_sample_size,
        position_mode=definition.position_mode,
        max_long_positions=definition.max_long_positions,
        max_short_positions=definition.max_short_positions,
        return_signals=True,
        exit_on_last_bar=True,
    )


def _benchmark_for_dates(
    benchmark_bars: pd.DataFrame,
    dates: pd.Series,
    *,
    initial_cash: float,
    strategy_name: str,
) -> pd.DataFrame:
    if benchmark_bars.empty:
        return pd.DataFrame()
    aligned = benchmark_bars[benchmark_bars["date"].isin(pd.to_datetime(dates))].copy()
    if aligned.empty:
        aligned = benchmark_bars.copy()
    curve = build_buy_and_hold_curve(aligned, initial_cash=initial_cash, strategy_name=strategy_name)
    return curve


def _walkforward_window_rows(
    *,
    config: PyBrokerLabConfig,
    definition: PyBrokerStrategyDefinition,
    frame: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
) -> list[dict[str, Any]]:
    strategy = Strategy(frame, config.start_date, config.end_date, config=_strategy_config(config, definition))
    windows = list(strategy.walkforward_split(frame, config.walkforward_windows, definition.lookahead, train_size=config.train_size))
    rows: list[dict[str, Any]] = []
    for index, window in enumerate(windows, start=1):
        rows.append(_evaluate_window(index, window, config=config, definition=definition, frame=frame, benchmark_prices=benchmark_prices))
    return rows


def _evaluate_window(
    window_number: int,
    window: WalkforwardWindow,
    *,
    config: PyBrokerLabConfig,
    definition: PyBrokerStrategyDefinition,
    frame: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
) -> dict[str, Any]:
    train_idx = window.train_data
    test_idx = window.test_data
    slice_start = min(train_idx.min(), test_idx.min())
    slice_end = max(train_idx.max(), test_idx.max())
    window_frame = frame.iloc[slice_start : slice_end + 1].copy()
    test_frame = frame.iloc[test_idx].copy()
    result = _run_backtest_window(config=config, definition=definition, frame=window_frame)
    curve = normalize_portfolio_curve(result.portfolio, strategy_name=definition.name, curve_type="strategy")
    benchmark_curve = _benchmark_for_dates(benchmark_prices, curve["date"] if not curve.empty else test_frame["date"], initial_cash=config.initial_cash, strategy_name=definition.name)
    strategy_metrics = compute_equity_metrics(
        equity_curve=curve,
        trades=normalize_trade_log(result.trades, strategy_name=definition.name),
        initial_cash=config.initial_cash,
        strategy_name=definition.name,
    )
    benchmark_metrics = compute_equity_metrics(
        equity_curve=benchmark_curve,
        trades=pd.DataFrame(),
        initial_cash=config.initial_cash,
        strategy_name=definition.name,
    )
    return {
        "strategy_name": definition.name,
        "window": window_number,
        "train_start": pd.to_datetime(window_frame["date"].min()).date().isoformat(),
        "train_end": pd.to_datetime(test_frame["date"].min()).date().isoformat(),
        "test_start": pd.to_datetime(test_frame["date"].min()).date().isoformat(),
        "test_end": pd.to_datetime(test_frame["date"].max()).date().isoformat(),
        "cagr": strategy_metrics["cagr"],
        "sharpe": strategy_metrics["sharpe"],
        "max_drawdown": strategy_metrics["max_drawdown"],
        "benchmark_cagr": benchmark_metrics["cagr"],
    }


def _run_backtest_window(*, config: PyBrokerLabConfig, definition: PyBrokerStrategyDefinition, frame: pd.DataFrame):
    start_date = pd.to_datetime(frame["date"].min()).date().isoformat()
    end_date = pd.to_datetime(frame["date"].max()).date().isoformat()
    strategy = Strategy(frame, start_date, end_date, config=_strategy_config(config, definition))
    strategy.add_execution(definition.execution, definition.symbols, indicators=definition.indicators, models=definition.models)
    strategy.set_slippage_model(FixedBpsSlippage(config.slippage_bps))
    return strategy.backtest(
        train_size=config.train_size,
        lookahead=definition.lookahead,
        warmup=min(config.warmup_bars, max(frame["date"].nunique() - 2, 1)),
        calc_bootstrap=False,
        disable_parallel=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    config = PyBrokerLabConfig(
        symbols=tuple(args.symbols),
        benchmark_symbol=args.benchmark_symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_cash=args.initial_cash,
        timeframe=args.timeframe,
        output_dir=Path(args.output_dir),
    )
    run_pybroker_lab(config, strategy_name=args.strategy)


if __name__ == "__main__":
    main()
