from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json

import pandas as pd

from trading_lab.data.market_calendar import MarketCalendar, get_default_calendar


@dataclass
class BenchmarkDiagnostics:
    benchmark_symbol: str
    coverage_ratio: float
    missing_session_count: int
    dropped_strategy_dates: int
    zero_return_days: int
    status: str
    warnings: list[str]
    created_at: datetime

    def to_record(self, run_id: str) -> dict[str, object]:
        return {
            "run_id": run_id,
            "benchmark_symbol": self.benchmark_symbol,
            "coverage_ratio": self.coverage_ratio,
            "missing_session_count": self.missing_session_count,
            "dropped_strategy_dates": self.dropped_strategy_dates,
            "zero_return_days": self.zero_return_days,
            "status": self.status,
            "warnings_json": json.dumps(self.warnings),
            "created_at": self.created_at.replace(tzinfo=None),
        }


def evaluate_benchmark_diagnostics(
    equity_curve: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    benchmark_curve: pd.DataFrame,
    benchmark_symbol: str,
    market_calendar: MarketCalendar | None = None,
) -> BenchmarkDiagnostics:
    calendar = market_calendar or get_default_calendar()
    warnings: list[str] = []

    if equity_curve.empty or benchmark_bars.empty or benchmark_curve.empty:
        warnings.append("Benchmark data is missing or empty for the selected period.")
        return BenchmarkDiagnostics(
            benchmark_symbol=benchmark_symbol,
            coverage_ratio=0.0,
            missing_session_count=0,
            dropped_strategy_dates=int(len(equity_curve.index)),
            zero_return_days=0,
            status="critical",
            warnings=warnings,
            created_at=datetime.now(UTC),
        )

    strategy_dates = pd.to_datetime(equity_curve["timestamp"]).dt.normalize()
    benchmark_dates = pd.to_datetime(benchmark_bars["timestamp"]).dt.normalize()
    benchmark_curve_dates = pd.to_datetime(benchmark_curve["timestamp"]).dt.normalize()
    range_start = strategy_dates.min().date()
    range_end = strategy_dates.max().date()
    expected_sessions = calendar.expected_sessions(range_start, range_end)
    observed_sessions = benchmark_dates.dt.date.tolist()
    missing_sessions = calendar.missing_sessions(observed_sessions, range_start, range_end)
    aligned_dates = strategy_dates.isin(benchmark_curve_dates).sum()
    coverage_ratio = float(aligned_dates / len(strategy_dates)) if len(strategy_dates) else 0.0
    dropped_strategy_dates = int(len(strategy_dates) - aligned_dates)

    benchmark_equity = benchmark_curve["benchmark_equity"].astype(float)
    zero_returns = benchmark_equity.pct_change().fillna(0.0).eq(0.0).sum()
    zero_return_days = int(max(zero_returns - 1, 0))

    if missing_sessions:
        warnings.append(f"Benchmark is missing {len(missing_sessions)} NYSE sessions in the requested window.")
    if coverage_ratio < 0.95:
        warnings.append("Benchmark date coverage is weaker than the strategy date coverage.")
    if dropped_strategy_dates > max(3, int(0.05 * len(strategy_dates))):
        warnings.append("Benchmark alignment required dropping many strategy dates.")
    if zero_return_days > max(10, int(0.25 * len(expected_sessions))):
        warnings.append("Benchmark has an unusual number of zero-return days.")

    status = "fresh"
    if warnings:
        status = "warning"
    if not expected_sessions or coverage_ratio == 0:
        status = "critical"

    return BenchmarkDiagnostics(
        benchmark_symbol=benchmark_symbol,
        coverage_ratio=coverage_ratio,
        missing_session_count=len(missing_sessions),
        dropped_strategy_dates=dropped_strategy_dates,
        zero_return_days=zero_return_days,
        status=status,
        warnings=warnings,
        created_at=datetime.now(UTC),
    )
