from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run unattended offline SPY research.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--timeframe", default="1d", choices=["1d", "15m", "5m"])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", default="today")
    parser.add_argument("--include-models", action="store_true")
    parser.add_argument("--max-combinations", type=int, default=None)
    parser.add_argument("--output-dir", default="reports/research_runs")
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--label-horizon-days", type=int, default=10)
    parser.add_argument("--target-r-multiple", type=float, default=1.5)
    parser.add_argument("--stop-r-multiple", type=float, default=1.0)
    return parser


def normalize_end_date(raw_end: str) -> str:
    if raw_end.lower() == "today":
        return str(date.today())
    return raw_end


def resolve_output_dir(base_dir: str) -> Path:
    return Path(base_dir)
