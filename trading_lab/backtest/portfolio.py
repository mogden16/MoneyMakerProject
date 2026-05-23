from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Position:
    symbol: str
    shares: float
    entry_price: float
    entry_timestamp: object
    highest_close: float
    holding_days: int = 0
    entry_signal_timestamp: object | None = None


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    def positions_value(self, prices: dict[str, float]) -> float:
        total = 0.0
        for symbol, position in self.positions.items():
            mark = prices.get(symbol, position.entry_price)
            total += position.shares * mark
        return total

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.positions_value(prices)

