import pandas as pd

from trading_lab.backtest.metrics import calculate_benchmark_metrics, calculate_cagr, calculate_max_drawdown


def test_calculate_cagr():
    equity_curve = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01", "2021-01-01"]),
            "equity": [100000.0, 121000.0],
        }
    )
    result = calculate_cagr(equity_curve, 100000.0)
    assert round(result, 4) == 0.2095


def test_calculate_max_drawdown():
    equity_curve = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=5, freq="D"),
            "equity": [100000.0, 110000.0, 90000.0, 95000.0, 120000.0],
        }
    )
    result = calculate_max_drawdown(equity_curve)
    assert round(result, 4) == -0.1818


def test_calculate_benchmark_metrics_alignment():
    strategy = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=4, freq="D"), "equity": [100, 105, 110, 120]})
    benchmark = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=4, freq="D"), "benchmark_equity": [100, 101, 103, 106]})
    metrics = calculate_benchmark_metrics(strategy, benchmark, 100.0)
    assert round(metrics["benchmark_total_return"], 4) == 0.06
    assert metrics["correlation"] > 0
