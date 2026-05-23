from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class MarketDataProvider(ABC):
    @abstractmethod
    def get_stock_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_corporate_actions(self, symbol: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self, symbol: str) -> dict:
        raise NotImplementedError

