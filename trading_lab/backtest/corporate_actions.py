from __future__ import annotations

import pandas as pd


def summarize_corporate_action_warnings(
    actions: pd.DataFrame,
    *,
    price_mode: str,
    adjusted_available: bool,
) -> list[str]:
    if actions.empty:
        return []

    warnings: list[str] = []
    splits = actions[actions["action_type"] == "split"]
    dividends = actions[actions["action_type"] == "dividend"]

    if price_mode == "adjusted_price_mode":
        if not adjusted_available:
            warnings.append("Adjusted-price mode was selected, but adjusted close was unavailable for at least one symbol.")
        if not splits.empty:
            warnings.append("Split events occurred during this backtest. Adjusted data should be preferred to avoid distorted return math.")
        if adjusted_available and not splits.empty:
            warnings.append("Adjusted data appears necessary because split events were present during the test window.")
    else:
        if not splits.empty:
            warnings.append("Raw-price mode includes split events in this window. Long-history returns may be distorted.")
        if not dividends.empty and dividends["cash_amount"].fillna(0.0).abs().max() >= 1.0:
            warnings.append("Raw-price mode includes large dividend events. Price-only comparisons may understate total return.")

    return warnings
