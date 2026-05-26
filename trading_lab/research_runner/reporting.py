from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    frame.to_csv(path, index=False)


def write_summary_markdown(
    *,
    output_path: Path,
    config: dict[str, Any],
    candidates: pd.DataFrame,
    rejected: pd.DataFrame,
    top_candidates: pd.DataFrame,
    highlights: dict[str, dict[str, Any]],
    signal_dataset: pd.DataFrame,
    dataset_warnings: list[str],
    model_summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    approved_breakdown: pd.DataFrame,
    manual_review_shortlist: pd.DataFrame,
    recommendation: dict[str, Any],
    model_warnings: list[str],
    run_warnings: list[str],
) -> None:
    lines = [
        "# SPY Research Runner Summary",
        "",
        "## Run Configuration",
        "",
    ]
    for key, value in config.items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Combination Summary", ""])
    lines.append(f"- Tested combinations: {len(candidates) + len(rejected)}")
    lines.append(f"- Candidate rows: {len(candidates)}")
    lines.append(f"- Rejected rows: {len(rejected)}")
    lines.extend(["", "## Top Candidate Highlights", ""])
    if highlights:
        for category, row in highlights.items():
            lines.append(f"- **{category}**: {row.get('entry_preset_label', row.get('entry_strategy_name'))} + {row.get('exit_preset_label', row.get('exit_structure_name'))} | {row.get('summary_comment', '')}")
    else:
        lines.append("- No ranked highlights were available.")
    lines.extend(["", "## Top Candidates", ""])
    if top_candidates.empty:
        lines.append("- No top candidates met the filter criteria.")
    else:
        for _, row in top_candidates.head(10).iterrows():
            lines.append(
                f"- {row['entry_preset_label']} + {row['exit_preset_label']}: "
                f"CAGR {row['cagr']:.2%}, Excess CAGR {row['excess_cagr']:.2%}, "
                f"Max DD {row['max_drawdown']:.2%}, Trades {int(row['number_of_trades'])}, "
                f"Label {row['candidate_label']}"
            )
    lines.extend(["", "## Signal Dataset", ""])
    lines.append(f"- Dataset rows: {len(signal_dataset)}")
    if not signal_dataset.empty:
        positive_rate = float(signal_dataset["label_good_signal"].mean())
        lines.append(f"- Positive label rate: {positive_rate:.2%}")
        lines.append(f"- Time range: {signal_dataset['timestamp'].min()} to {signal_dataset['timestamp'].max()}")
    for warning in dataset_warnings:
        lines.append(f"- Warning: {warning}")
    lines.extend(["", "## Model Results", ""])
    if model_summary.empty:
        lines.append("- No models were run.")
    else:
        for _, row in model_summary.iterrows():
            lines.append(
                f"- {row['model_name']}: F1 {row['f1']:.3f}, ROC AUC {row['roc_auc']:.3f}, "
                f"Approved signals {int(row['approved_signal_count'])}, "
                f"Approved avg forward return 10d {row['approved_avg_forward_return_10d']:.2%}"
            )
    lines.extend(["", "## Model vs Baseline", ""])
    if model_comparison.empty:
        lines.append("- No model-vs-baseline comparison was available.")
    else:
        for _, row in model_comparison.iterrows():
            lines.append(
                f"- {row['model_name']}: forward-return edge {row['forward_return_edge_10d']:.2%}, "
                f"win-rate edge {row['win_rate_edge']:.2%}, approval rate {row['approval_rate']:.2%}"
            )
    lines.extend(["", "## Approved Signal Breakdown", ""])
    if approved_breakdown.empty:
        lines.append("- No approved-signal breakdown was available.")
    else:
        for _, row in approved_breakdown.head(10).iterrows():
            lines.append(
                f"- {row['model_name']} | {row['entry_strategy_name']} + {row['exit_structure_name']}: "
                f"signals {int(row['approved_signal_count'])}, avg forward return 10d {row['avg_forward_return_10d']:.2%}, "
                f"win rate {row['win_rate']:.2%}"
            )
    lines.extend(["", "## Manual Review Shortlist", ""])
    if manual_review_shortlist.empty:
        lines.append("- No model-backed shortlist met the manual review criteria.")
    else:
        for _, row in manual_review_shortlist.head(10).iterrows():
            lines.append(
                f"- {row['model_name']} | {row['entry_strategy_name']} + {row['exit_structure_name']}: "
                f"signals {int(row['approved_signal_count'])}, avg forward return 10d {row['avg_forward_return_10d']:.2%}, "
                f"candidate {row['candidate_label']} | {row['manual_review_comment']}"
            )
    lines.extend(["", "## Recommendation", ""])
    if recommendation:
        model_name = recommendation.get("model_name")
        model_text = f" backed by {model_name}" if model_name else ""
        lines.append(
            f"- Recommended next implementation candidate: {recommendation.get('entry_strategy_name')} + "
            f"{recommendation.get('exit_structure_name')} on {recommendation.get('timeframe')}{model_text}."
        )
        lines.append(f"- Reason: {recommendation.get('recommendation_reason', '')}")
        if recommendation.get("manual_review_comment"):
            lines.append(f"- Review note: {recommendation.get('manual_review_comment')}")
    else:
        lines.append("- No single candidate was strong enough to recommend automatically.")
    lines.extend(["", "## Warnings and Limitations", ""])
    combined_warnings = [*run_warnings, *dataset_warnings, *model_warnings]
    if combined_warnings:
        for warning in combined_warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- No major warnings were generated.")
    lines.extend(
        [
            "",
            "## Suggested Next Manual Review Steps",
            "",
            "- Review the best overall and best low-drawdown candidates against buy-and-hold SPY.",
            "- Ignore suspicious high-return candidates with weak trade counts or unstable fold behavior.",
            "- Inspect the signal dataset for the leading entry/exit combinations before trusting model output.",
            "- Promote at most one candidate to forward paper trading after manual review.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_warnings_markdown(output_path: Path, warnings: list[str]) -> None:
    lines = ["# Research Runner Warnings", ""]
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- No major warnings were generated.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_recommendation_markdown(output_path: Path, recommendation: dict[str, Any]) -> None:
    lines = ["# Research Recommendation", ""]
    if not recommendation:
        lines.append("- No candidate was strong enough to recommend automatically.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return
    lines.append(f"- **Source**: {recommendation.get('source')}")
    lines.append(f"- **Timeframe**: {recommendation.get('timeframe')}")
    lines.append(f"- **Entry strategy**: {recommendation.get('entry_strategy_name')}")
    lines.append(f"- **Entry archetype**: {recommendation.get('strategy_archetype')}")
    lines.append(f"- **Exit structure**: {recommendation.get('exit_structure_name')}")
    lines.append(f"- **Exit archetype**: {recommendation.get('exit_archetype')}")
    lines.append(f"- **Candidate label**: {recommendation.get('candidate_label')}")
    if recommendation.get("model_name"):
        lines.append(f"- **Model backing**: {recommendation.get('model_name')}")
    if recommendation.get("approved_signal_count") is not None:
        lines.append(f"- **Approved signal count**: {recommendation.get('approved_signal_count')}")
    if recommendation.get("avg_forward_return_10d") is not None:
        lines.append(f"- **Approved avg forward return 10d**: {recommendation.get('avg_forward_return_10d'):.2%}")
    lines.extend(
        [
            "",
            "## Reason",
            "",
            f"- {recommendation.get('recommendation_reason', 'No reason generated.')}",
        ]
    )
    if recommendation.get("manual_review_comment"):
        lines.extend(["", "## Review Note", "", f"- {recommendation.get('manual_review_comment')}"])
    output_path.write_text("\n".join(lines), encoding="utf-8")
