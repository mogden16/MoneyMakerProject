from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class StrategyBase(ABC):
    name = "base"

    @abstractmethod
    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def parameters(self) -> dict:
        return {}

