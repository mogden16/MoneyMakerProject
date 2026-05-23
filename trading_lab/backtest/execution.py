from __future__ import annotations


def apply_slippage(price: float, slippage_pct: float, side: str) -> float:
    if side == "buy":
        return price * (1 + slippage_pct)
    if side == "sell":
        return price * (1 - slippage_pct)
    raise ValueError(f"Unsupported side: {side}")

