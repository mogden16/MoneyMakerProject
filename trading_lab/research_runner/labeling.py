from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LabelingConfig:
    label_horizon_days: int = 10
    target_r_multiple: float = 1.5
    stop_r_multiple: float = 1.0


def infer_initial_risk_pct(signal_row: pd.Series) -> float:
    exit_params = signal_row.get("exit_parameters_json") or {}
    if isinstance(exit_params, str):
        exit_params = {}
    for key in ["stop_loss_pct", "trailing_stop_pct"]:
        value = float(exit_params.get(key) or 0.0)
        if value > 0:
            return value
    atr_pct = float(signal_row.get("atr_pct") or 0.0)
    if atr_pct > 0:
        return min(max(atr_pct, 0.005), 0.05)
    return 0.02


def apply_labels(signal_row: pd.Series, bars: pd.DataFrame, signal_index: int, config: LabelingConfig) -> dict[str, float | int | bool]:
    """Label one signal row using future bars only."""
    signal_price = float(signal_row["signal_price"])
    future = bars.iloc[signal_index + 1 : signal_index + 1 + max(config.label_horizon_days, 20)].copy()
    if future.empty:
        return {
            "forward_return_5d": 0.0,
            "forward_return_10d": 0.0,
            "forward_return_20d": 0.0,
            "max_favorable_excursion_10d": 0.0,
            "max_adverse_excursion_10d": 0.0,
            "target_before_stop_10d": False,
            "stop_before_target_10d": False,
            "positive_r_multiple_10d": False,
            "label_good_signal": 0,
        }

    def _forward_return(days: int) -> float:
        sample = future.iloc[:days]
        if sample.empty:
            return 0.0
        return float(sample["close"].iloc[-1] / signal_price - 1)

    horizon = future.iloc[: config.label_horizon_days]
    mfe = float(horizon["high"].max() / signal_price - 1) if not horizon.empty else 0.0
    mae = float(horizon["low"].min() / signal_price - 1) if not horizon.empty else 0.0
    risk_pct = infer_initial_risk_pct(signal_row)
    target_pct = risk_pct * config.target_r_multiple
    stop_pct = risk_pct * config.stop_r_multiple
    target_price = signal_price * (1 + target_pct)
    stop_price = signal_price * (1 - stop_pct)
    target_before_stop = False
    stop_before_target = False
    for _, bar in horizon.iterrows():
        high_hit = float(bar["high"]) >= target_price
        low_hit = float(bar["low"]) <= stop_price
        if high_hit and low_hit:
            stop_before_target = True
            break
        if low_hit:
            stop_before_target = True
            break
        if high_hit:
            target_before_stop = True
            break
    forward_10d = _forward_return(10)
    label_good_signal = int(target_before_stop or (forward_10d > 0 and mae > -(stop_pct * 1.1)))
    return {
        "forward_return_5d": _forward_return(5),
        "forward_return_10d": forward_10d,
        "forward_return_20d": _forward_return(20),
        "max_favorable_excursion_10d": mfe,
        "max_adverse_excursion_10d": mae,
        "target_before_stop_10d": target_before_stop,
        "stop_before_target_10d": stop_before_target,
        "positive_r_multiple_10d": bool(forward_10d > 0 and (forward_10d / max(risk_pct, 1e-9)) > 0),
        "label_good_signal": label_good_signal,
    }
