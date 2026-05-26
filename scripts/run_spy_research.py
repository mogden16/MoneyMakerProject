from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trading_lab.research_runner import ResearchRunnerConfig, run_research_pipeline
from trading_lab.research_runner.cli import build_parser, normalize_end_date


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = ResearchRunnerConfig(
        symbol=args.symbol.upper(),
        timeframe=args.timeframe,
        start=args.start,
        end=normalize_end_date(args.end),
        include_models=bool(args.include_models),
        max_combinations=args.max_combinations,
        output_dir=args.output_dir,
        min_trades=args.min_trades,
        label_horizon_days=args.label_horizon_days,
        target_r_multiple=args.target_r_multiple,
        stop_r_multiple=args.stop_r_multiple,
    )
    result = run_research_pipeline(config)
    print(f"Research run completed: {result.output_path}")
    print(f"Candidates: {len(result.candidates)}")
    print(f"Signals: {len(result.signal_dataset)}")
    if config.include_models:
        print(f"Model rows: {len(result.model_summary)}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
