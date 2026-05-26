from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_report_markdown(
    *,
    path: Path,
    config: dict[str, Any],
    summary: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    benchmark_metrics: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    walkforward_windows: pd.DataFrame,
) -> None:
    lines = [
        "# PyBroker Lab Report",
        "",
        "## Run Configuration",
        "",
    ]
    for key, value in config.items():
        lines.append(f"- **{key}**: {value}")
    for _, summary_row in summary.iterrows():
        strategy_name = summary_row["strategy_name"]
        metrics_row = strategy_metrics[strategy_metrics["strategy_name"] == strategy_name].iloc[0]
        benchmark_row = benchmark_metrics[benchmark_metrics["strategy_name"] == strategy_name].iloc[0]
        lines.extend(
            [
                "",
                f"## {strategy_name}",
                "",
                f"- Data source: {summary_row['data_source']}",
                f"- Assumptions: {summary_row['assumptions']}",
                f"- Date range: {summary_row['start_date']} to {summary_row['end_date']}",
                f"- Total return: {metrics_row['total_return']:.2%}",
                f"- CAGR: {metrics_row['cagr']:.2%}",
                f"- Sharpe: {metrics_row['sharpe']:.2f}",
                f"- Sortino: {metrics_row['sortino']:.2f}",
                f"- Max drawdown: {metrics_row['max_drawdown']:.2%}",
                f"- Calmar: {metrics_row['calmar']:.2f}",
                f"- Trade count: {int(metrics_row['trade_count'])}",
                f"- Win rate: {metrics_row['win_rate']:.2%}",
                f"- Profit factor: {metrics_row['profit_factor']:.2f}",
                f"- Average win: {metrics_row['average_win']:.2%}",
                f"- Average loss: {metrics_row['average_loss']:.2%}",
                f"- Exposure: {metrics_row['exposure']:.2%}",
                f"- Comparison to SPY buy-and-hold: total return delta {summary_row['total_return_delta']:.2%}, CAGR delta {summary_row['cagr_delta']:.2%}, Sharpe delta {summary_row['sharpe_delta']:.2f}, Drawdown delta {summary_row['max_drawdown_delta']:.2%}",
                f"- Benchmark SPY CAGR: {benchmark_row['cagr']:.2%}",
                f"- Benchmark SPY total return: {benchmark_row['total_return']:.2%}",
                f"- PASS/FAIL: {summary_row['status']}",
                f"- Decision note: {summary_row['decision_reason']}",
                f"- Bootstrap note: {summary_row['bootstrap_note']}",
            ]
        )
        strategy_bootstrap = bootstrap_metrics[bootstrap_metrics["strategy_name"] == strategy_name]
        if not strategy_bootstrap.empty:
            lines.extend(["", "### Bootstrap Confidence Intervals", ""])
            for _, row in strategy_bootstrap.iterrows():
                lines.append(f"- {row['metric']} {row['confidence']}: [{row['lower']:.4f}, {row['upper']:.4f}]")
        strategy_windows = walkforward_windows[walkforward_windows["strategy_name"] == strategy_name]
        if not strategy_windows.empty:
            lines.extend(["", "### Walk-Forward Windows", ""])
            for _, row in strategy_windows.iterrows():
                lines.append(
                    f"- Window {int(row['window'])}: {row['test_start']} to {row['test_end']} | CAGR {row['cagr']:.2%}, Sharpe {row['sharpe']:.2f}, Max DD {row['max_drawdown']:.2%}"
                )
    path.write_text("\n".join(lines), encoding="utf-8")
