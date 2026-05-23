from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def build_equity_chart(equity_curve: pd.DataFrame, benchmark_curve: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    if not equity_curve.empty:
        fig.add_trace(go.Scatter(x=equity_curve["timestamp"], y=equity_curve["equity"], mode="lines", name="Strategy"))
    if benchmark_curve is not None and not benchmark_curve.empty:
        fig.add_trace(go.Scatter(x=benchmark_curve["timestamp"], y=benchmark_curve["benchmark_equity"], mode="lines", name="Benchmark"))
    fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Equity")
    return fig


def build_drawdown_chart(equity_curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not equity_curve.empty:
        drawdown = equity_curve["equity"] / equity_curve["equity"].cummax() - 1
        fig.add_trace(go.Scatter(x=equity_curve["timestamp"], y=drawdown, fill="tozeroy", mode="lines", name="Drawdown"))
    fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown")
    return fig


def build_multi_equity_chart(curves: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for label, curve in curves.items():
        if curve.empty:
            continue
        fig.add_trace(go.Scatter(x=curve["timestamp"], y=curve["equity"], mode="lines", name=label))
    fig.update_layout(title="Backtest Comparison", xaxis_title="Date", yaxis_title="Equity")
    return fig


def build_multi_drawdown_chart(curves: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for label, curve in curves.items():
        if curve.empty:
            continue
        drawdown = curve["equity"] / curve["equity"].cummax() - 1
        fig.add_trace(go.Scatter(x=curve["timestamp"], y=drawdown, mode="lines", name=label))
    fig.update_layout(title="Drawdown Comparison", xaxis_title="Date", yaxis_title="Drawdown")
    return fig
