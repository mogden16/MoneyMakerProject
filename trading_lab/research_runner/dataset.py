from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from trading_lab.research_runner.features import build_feature_frame, extract_signal_feature_row
from trading_lab.research_runner.labeling import LabelingConfig, apply_labels


@dataclass(frozen=True)
class SignalDatasetResult:
    frame: pd.DataFrame
    warnings: list[str]


def build_signal_dataset(
    *,
    bars: pd.DataFrame,
    signal_frame: pd.DataFrame,
    strategy_name: str,
    entry_parameters: dict[str, Any],
    exit_structure_name: str,
    exit_parameters: dict[str, Any],
    timeframe: str,
    labeling_config: LabelingConfig,
) -> SignalDatasetResult:
    """Create a signal-level research dataset for one strategy/exit configuration."""
    if signal_frame.empty or not signal_frame.get("entry_signal", pd.Series(dtype=bool)).astype(bool).any():
        return SignalDatasetResult(frame=pd.DataFrame(), warnings=["No entry signals were generated for this strategy configuration."])

    feature_frame = build_feature_frame(signal_frame)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    signal_indices = feature_frame.index[feature_frame["entry_signal"].astype(bool)].tolist()
    for signal_index in signal_indices:
        if signal_index >= len(feature_frame) - 1:
            continue
        signal_row = extract_signal_feature_row(
            feature_frame,
            signal_index,
            strategy_name=strategy_name,
            exit_structure_name=exit_structure_name,
            timeframe=timeframe,
            entry_parameters=entry_parameters,
            exit_parameters=exit_parameters,
        )
        signal_row["planned_entry_timestamp"] = pd.Timestamp(feature_frame.iloc[signal_index + 1]["timestamp"])
        signal_row["planned_entry_price"] = float(feature_frame.iloc[signal_index + 1]["open"])
        labels = apply_labels(pd.Series(signal_row), feature_frame, signal_index, labeling_config)
        signal_row.update(labels)
        rows.append(signal_row)
    if not rows:
        warnings.append("Signals were present, but none had enough forward bars for labeling.")
        return SignalDatasetResult(frame=pd.DataFrame(), warnings=warnings)
    frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    for column in ["entry_parameters_json", "exit_parameters_json"]:
        frame[column] = frame[column].apply(lambda value: json.dumps(value, default=str) if isinstance(value, dict) else value)
    return SignalDatasetResult(frame=frame, warnings=warnings)
