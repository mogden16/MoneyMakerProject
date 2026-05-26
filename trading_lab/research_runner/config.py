from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchRunnerConfig:
    symbol: str
    timeframe: str
    start: str
    end: str
    include_models: bool
    max_combinations: int | None
    output_dir: str
    min_trades: int
    label_horizon_days: int
    target_r_multiple: float
    stop_r_multiple: float

