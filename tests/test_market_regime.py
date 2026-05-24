from __future__ import annotations

from datetime import date

import pandas as pd

from trading_lab.market_regime.indicators import compute_moving_average_distance
from trading_lab.market_regime.narrative import generate_analyst_summary
from trading_lab.market_regime.scoring import build_regime_summary
from trading_lab.market_regime.schema import DataSourceInfo, MarketRegimeReport, RegimeSummary, ReportSection
from trading_lab.market_regime.seasonality import build_seasonality_rows, compute_forward_returns


def make_close_series(length: int = 260, start: float = 100.0, step: float = 1.0) -> pd.Series:
    index = pd.date_range("2021-01-01", periods=length, freq="B")
    values = [start + step * idx for idx in range(length)]
    return pd.Series(values, index=index, dtype=float)


def test_compute_moving_average_distance_matches_expected_ratio():
    series = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=pd.date_range("2024-01-01", periods=5, freq="B"))
    distance = compute_moving_average_distance(series, 5)
    expected = 14.0 / 12.0 - 1.0
    assert distance is not None
    assert abs(distance - expected) < 1e-9


def test_build_regime_summary_produces_bullish_label_with_positive_inputs():
    index_rows = [
        {"symbol": "SPY", "return_20d": 0.06, "distance_50dma": 0.05, "distance_200dma": 0.11, "realized_vol_20d": 0.14, "overextended_flag": False},
        {"symbol": "QQQ", "return_20d": 0.07, "distance_50dma": 0.08, "distance_200dma": 0.16, "realized_vol_20d": 0.18, "overextended_flag": False},
        {"symbol": "IWM", "return_20d": 0.03, "distance_50dma": 0.02, "distance_200dma": 0.06, "realized_vol_20d": 0.20, "overextended_flag": False},
        {"symbol": "SOXX", "return_20d": 0.08, "distance_50dma": 0.10, "distance_200dma": 0.22, "realized_vol_20d": 0.21, "overextended_flag": False},
    ]
    breadth_rows = [{"pct_above_50dma": 68.0, "pct_above_200dma": 57.0, "new_highs": 90, "new_lows": 24}]
    sector_rows = [
        {"symbol": "XLK", "trend_vs_50dma": "Above", "relative_strength_vs_spy": 0.03},
        {"symbol": "XLF", "trend_vs_50dma": "Above", "relative_strength_vs_spy": 0.02},
        {"symbol": "XLE", "trend_vs_50dma": "Above", "relative_strength_vs_spy": 0.01},
        {"symbol": "XLU", "trend_vs_50dma": "Above", "relative_strength_vs_spy": -0.01},
        {"symbol": "XLV", "trend_vs_50dma": "Above", "relative_strength_vs_spy": 0.00},
        {"symbol": "XLP", "trend_vs_50dma": "Below", "relative_strength_vs_spy": -0.02},
        {"symbol": "XLY", "trend_vs_50dma": "Above", "relative_strength_vs_spy": 0.02},
        {"symbol": "XLI", "trend_vs_50dma": "Above", "relative_strength_vs_spy": 0.02},
        {"symbol": "XLC", "trend_vs_50dma": "Above", "relative_strength_vs_spy": 0.01},
    ]
    summary = build_regime_summary(index_rows, breadth_rows, sector_rows, options_rows=[], options_source_is_demo=True)
    assert summary.regime_label == "Bullish"
    assert summary.trend_direction == "Up"
    assert summary.recommended_posture == "Fully Invested"
    assert summary.total_score >= 5


def test_seasonality_uses_prior_year_samples_only():
    series = make_close_series(length=800, start=100.0, step=0.25)
    rows = build_seasonality_rows(series, as_of_date=date(2024, 5, 24))
    assert len(rows) == 4
    assert all("sample_size" in row for row in rows)
    assert any(row["sample_size"] > 0 for row in rows)
    assert all("methodology" in row for row in rows)


def test_compute_forward_returns_aligns_with_horizon():
    series = pd.Series([100.0, 110.0, 121.0, 133.1], index=pd.date_range("2024-01-01", periods=4, freq="B"))
    forward = compute_forward_returns(series, 1)
    assert round(float(forward.iloc[0]), 6) == 0.1
    assert round(float(forward.iloc[1]), 6) == 0.1


def test_narrative_skips_unavailable_metrics():
    summary = RegimeSummary(
        regime_label="Neutral",
        trend_direction="Mixed",
        recommended_posture="Normal Exposure",
        short_summary="Neutral backdrop with balanced signals.",
        total_score=1,
        components=[],
    )
    report = MarketRegimeReport(
        generated_at="2026-05-24T10:00:00-04:00",
        summary=summary,
        analyst_summary="",
        index_momentum=ReportSection(title="Index Momentum", rows=[{"symbol": "SPY", "distance_50dma": None, "return_20d": None, "overextended_flag": False}], source=DataSourceInfo(label="Demo", detail="Demo index data.", is_demo=True)),
        breadth=ReportSection(title="Breadth", rows=[], source=DataSourceInfo(label="Demo", detail="Demo breadth.", is_demo=True)),
        options_positioning=ReportSection(title="Options", rows=[], source=DataSourceInfo(label="Demo", detail="Demo options.", is_demo=True)),
        sector_leadership=ReportSection(title="Sectors", rows=[], source=DataSourceInfo(label="Demo", detail="Demo sectors.", is_demo=True)),
        seasonality=ReportSection(title="Seasonality", rows=[], source=DataSourceInfo(label="Demo", detail="Demo seasonality.", is_demo=True)),
        macro_calendar=ReportSection(title="Macro", rows=[], source=DataSourceInfo(label="Demo", detail="Demo macro.", is_demo=True)),
        earnings_watch=ReportSection(title="Earnings", rows=[], source=DataSourceInfo(label="Demo", detail="Demo earnings.", is_demo=True)),
        warnings=[],
    )
    narrative = generate_analyst_summary(report)
    assert "not available" not in narrative.lower()
    assert "average 20-day return" not in narrative.lower()
    assert "dealer positioning reads" not in narrative.lower()
