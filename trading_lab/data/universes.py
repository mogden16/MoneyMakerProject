from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UniverseDefinition:
    name: str
    tickers: list[str]


PREDEFINED_UNIVERSES: dict[str, UniverseDefinition] = {
    "Single benchmark": UniverseDefinition(name="Single benchmark", tickers=["SPY"]),
    "Large-cap tech": UniverseDefinition(name="Large-cap tech", tickers=["QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"]),
    "Sector ETFs": UniverseDefinition(name="Sector ETFs", tickers=["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU"]),
    "Broad ETFs": UniverseDefinition(name="Broad ETFs", tickers=["SPY", "QQQ", "IWM", "DIA"]),
    "Defensive ETFs": UniverseDefinition(name="Defensive ETFs", tickers=["XLP", "XLU", "XLV"]),
    "Cyclical ETFs": UniverseDefinition(name="Cyclical ETFs", tickers=["XLY", "XLI", "XLF", "XLE"]),
    "Custom": UniverseDefinition(name="Custom", tickers=[]),
}


def list_universe_names() -> list[str]:
    """Return the supported universe names in display order."""
    return list(PREDEFINED_UNIVERSES.keys())


def get_universe_tickers(universe_name: str) -> list[str]:
    """Return a copy of the tickers for a predefined universe."""
    definition = PREDEFINED_UNIVERSES.get(universe_name)
    return list(definition.tickers) if definition else []


def normalize_ticker_list(raw_tickers: str | list[str]) -> list[str]:
    """Normalize a comma-separated ticker string into a de-duplicated uppercase list."""
    if isinstance(raw_tickers, list):
        candidates = raw_tickers
    else:
        candidates = raw_tickers.split(",")
    normalized: list[str] = []
    for item in candidates:
        ticker = str(item).strip().upper()
        if ticker and ticker not in normalized:
            normalized.append(ticker)
    return normalized
