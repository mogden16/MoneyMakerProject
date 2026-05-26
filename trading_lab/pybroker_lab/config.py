from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

from pybroker.common import PositionMode
from pybroker.indicator import Indicator
from pybroker.model import ModelSource


@dataclass(frozen=True)
class PyBrokerLabConfig:
    symbols: tuple[str, ...] = ("SPY",)
    benchmark_symbol: str = "SPY"
    start_date: str = "2010-01-01"
    end_date: str = field(default_factory=lambda: date.today().isoformat())
    initial_cash: float = 100000.0
    timeframe: str = "1d"
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    warmup_bars: int = 200
    train_size: float = 0.7
    walkforward_windows: int = 3
    bootstrap_sample_size: int = 1000
    output_dir: Path = Path("outputs/pybroker_lab")
    strategy_params: dict[str, Any] = field(default_factory=dict)
    materially_worse_cagr_threshold: float = 0.02
    sizing_method: str = "percent_equity"
    sizing_value: float = 1.0


@dataclass(frozen=True)
class PyBrokerStrategyDefinition:
    name: str
    symbols: tuple[str, ...]
    indicators: tuple[Indicator, ...]
    execution: Callable[..., None]
    models: tuple[ModelSource, ...] = ()
    lookahead: int = 1
    description: str = ""
    data_source_name: str = "PyBroker YFinance"
    assumptions: tuple[str, ...] = ()
    max_long_positions: int | None = 1
    max_short_positions: int | None = None
    position_mode: PositionMode = PositionMode.LONG_ONLY
