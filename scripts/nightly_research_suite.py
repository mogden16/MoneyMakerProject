from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trading_lab.research_runner.config import ResearchRunnerConfig
from trading_lab.research_runner.runner import run_research_pipeline


def main() -> None:
    config = ResearchRunnerConfig(
        symbol="SPY",
        timeframe="1d",
        start="2000-01-01",
        end=str(date.today()),
        include_models=True,
        max_combinations=None,
        output_dir="reports/research_runs",
        min_trades=20,
        label_horizon_days=10,
        target_r_multiple=1.5,
        stop_r_multiple=1.0,
    )
    result = run_research_pipeline(config)
    print(f"Nightly research suite completed: {result.output_path}")
    print(f"Candidates: {len(result.candidates)}")
    print(f"Signals: {len(result.signal_dataset)}")
    print(f"Model rows: {len(result.model_summary)}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
