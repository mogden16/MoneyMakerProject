from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trading_lab.research_runner.models import train_time_series_models
from trading_lab.research_runner.reporting import write_csv, write_warnings_markdown


def parse_args() -> tuple[Path, Path]:
    import argparse

    parser = argparse.ArgumentParser(description="Train offline signal filter models from a saved signal dataset.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    return Path(args.dataset_path), Path(args.output_dir)


def main() -> None:
    dataset_path, output_dir = parse_args()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    result = train_time_series_models(dataset)
    write_csv(result.summary, output_dir / "model_results.csv")
    write_csv(result.folds, output_dir / "model_fold_results.csv")
    write_csv(result.feature_importance, output_dir / "feature_importance.csv")
    write_csv(result.approved_signals, output_dir / "model_approved_signals.csv")
    write_csv(result.comparison, output_dir / "model_comparison.csv")
    write_csv(result.approved_breakdown, output_dir / "approved_signal_breakdown.csv")
    write_warnings_markdown(output_dir / "warnings.md", result.warnings)
    print(f"Model training completed: {output_dir}")
    print(f"Dataset rows: {len(dataset)}")
    print(f"Model summary rows: {len(result.summary)}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
