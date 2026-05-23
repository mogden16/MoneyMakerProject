from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StockBarRecord(BaseModel):
    source_vendor: str
    symbol: str
    timeframe: str = "1d"
    timestamp: str
    session_date: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float
    dividends: float = 0.0
    stock_splits: float = 0.0
    adjusted_flag: bool = False
    retrieved_at: str


class CorporateActionRecord(BaseModel):
    source_vendor: str
    symbol: str
    action_type: Literal["dividend", "split"]
    effective_date: str
    cash_amount: float | None = None
    split_ratio: float | None = None
    split_from: float | None = None
    split_to: float | None = None
    retrieved_at: str


class BacktestRunRecord(BaseModel):
    run_id: str
    strategy_name: str
    parameters_json: str
    symbols_csv: str
    start_date: str
    end_date: str
    created_at: str
    initial_capital: float
    timeframe: str = "1d"
    total_return: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    number_of_trades: int = 0


class BacktestTradeRecord(BaseModel):
    run_id: str
    symbol: str
    entry_timestamp: str
    exit_timestamp: str
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    return_pct: float
    holding_days: int
    exit_reason: str


class BacktestEquityCurveRecord(BaseModel):
    run_id: str
    timestamp: str
    equity: float
    cash: float
    positions_value: float
    drawdown: float


class PositionSizingSpec(BaseModel):
    method: Literal["fixed_dollar", "percent_of_portfolio"] = "percent_of_portfolio"
    value: float = Field(gt=0)
