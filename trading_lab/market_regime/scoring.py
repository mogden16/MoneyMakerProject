from __future__ import annotations

from typing import Iterable

from trading_lab.market_regime.schema import RegimeSummary, ScoreComponent


def _safe_values(rows: Iterable[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def score_trend(index_rows: list[dict[str, object]]) -> ScoreComponent:
    above_50 = sum(1 for row in index_rows if isinstance(row.get("distance_50dma"), (int, float)) and float(row["distance_50dma"]) > 0)
    above_200 = sum(1 for row in index_rows if isinstance(row.get("distance_200dma"), (int, float)) and float(row["distance_200dma"]) > 0)
    if above_50 >= 4 and above_200 >= 3:
        score = 2
    elif above_50 >= 3 and above_200 >= 2:
        score = 1
    elif above_50 <= 1 and above_200 == 0:
        score = -2
    elif above_50 <= 2 and above_200 <= 1:
        score = -1
    else:
        score = 0
    return ScoreComponent("Trend", score, True, f"{above_50} of 4 indexes are above the 50 DMA and {above_200} are above the 200 DMA.")


def score_momentum(index_rows: list[dict[str, object]]) -> ScoreComponent:
    returns_20 = _safe_values(index_rows, "return_20d")
    average_return = sum(returns_20) / len(returns_20) if returns_20 else 0.0
    overextended = any(bool(row.get("overextended_flag")) for row in index_rows)
    if average_return >= 0.05:
        score = 2
    elif average_return > 0.0:
        score = 1
    elif average_return <= -0.05:
        score = -2
    elif average_return < 0.0:
        score = -1
    else:
        score = 0
    if overextended and score > -2:
        score -= 1
    return ScoreComponent("Momentum", max(-2, min(2, score)), True, f"Average 20-day return across tracked indexes is {average_return:.2%}; SOXX overextended flag is {'on' if overextended else 'off'}.")


def score_breadth(breadth_rows: list[dict[str, object]]) -> ScoreComponent:
    above_50 = _safe_values(breadth_rows, "pct_above_50dma")
    above_200 = _safe_values(breadth_rows, "pct_above_200dma")
    net_highs = [float(row.get("new_highs", 0)) - float(row.get("new_lows", 0)) for row in breadth_rows if isinstance(row.get("new_highs"), (int, float)) and isinstance(row.get("new_lows"), (int, float))]
    avg_50 = sum(above_50) / len(above_50) if above_50 else 0.0
    avg_200 = sum(above_200) / len(above_200) if above_200 else 0.0
    avg_net_highs = sum(net_highs) / len(net_highs) if net_highs else 0.0
    if avg_50 >= 65 and avg_200 >= 55 and avg_net_highs > 25:
        score = 2
    elif avg_50 >= 55 and avg_200 >= 45 and avg_net_highs >= 0:
        score = 1
    elif avg_50 <= 35 and avg_200 <= 30 and avg_net_highs < 0:
        score = -2
    elif avg_50 <= 45 and avg_200 <= 40:
        score = -1
    else:
        score = 0
    return ScoreComponent("Breadth", score, True, f"Average breadth is {avg_50:.1f}% above the 50 DMA and {avg_200:.1f}% above the 200 DMA, with average net highs of {avg_net_highs:.1f}.")


def score_volatility(index_rows: list[dict[str, object]]) -> ScoreComponent:
    spy_row = next((row for row in index_rows if row.get("symbol") == "SPY"), None)
    realized_vol = float(spy_row["realized_vol_20d"]) if spy_row and isinstance(spy_row.get("realized_vol_20d"), (int, float)) else None
    if realized_vol is None:
        return ScoreComponent("Volatility", 0, False, "SPY realized volatility is not available.")
    if realized_vol < 0.12:
        score = 2
    elif realized_vol < 0.20:
        score = 1
    elif realized_vol < 0.28:
        score = 0
    elif realized_vol < 0.36:
        score = -1
    else:
        score = -2
    return ScoreComponent("Volatility", score, True, f"SPY 20-day realized volatility is {realized_vol:.2%}.")


def score_sector_leadership(sector_rows: list[dict[str, object]]) -> ScoreComponent:
    positive_relative = sum(1 for row in sector_rows if isinstance(row.get("relative_strength_vs_spy"), (int, float)) and float(row["relative_strength_vs_spy"]) > 0)
    above_50 = sum(1 for row in sector_rows if row.get("trend_vs_50dma") == "Above")
    cyclical_symbols = {"XLK", "XLY", "XLI", "XLF", "XLC", "XLE"}
    defensive_symbols = {"XLU", "XLP", "XLV"}
    cyclical_strength = [float(row["relative_strength_vs_spy"]) for row in sector_rows if row.get("symbol") in cyclical_symbols and isinstance(row.get("relative_strength_vs_spy"), (int, float))]
    defensive_strength = [float(row["relative_strength_vs_spy"]) for row in sector_rows if row.get("symbol") in defensive_symbols and isinstance(row.get("relative_strength_vs_spy"), (int, float))]
    cyclical_avg = sum(cyclical_strength) / len(cyclical_strength) if cyclical_strength else 0.0
    defensive_avg = sum(defensive_strength) / len(defensive_strength) if defensive_strength else 0.0
    if positive_relative >= 6 and above_50 >= 6 and cyclical_avg >= defensive_avg:
        score = 2
    elif positive_relative >= 5 and above_50 >= 5:
        score = 1
    elif positive_relative <= 2 and above_50 <= 3 and cyclical_avg < defensive_avg:
        score = -2
    elif positive_relative <= 3 and above_50 <= 4:
        score = -1
    else:
        score = 0
    return ScoreComponent("Sector Leadership", score, True, f"{positive_relative} of 9 sectors are outperforming SPY on a 1-month basis and {above_50} are above the 50 DMA.")


def score_options_positioning(options_rows: list[dict[str, object]], *, source_is_demo: bool) -> ScoreComponent:
    if source_is_demo or not options_rows:
        return ScoreComponent("Options Positioning", 0, False, "Options positioning is demo-only and excluded from the regime score.")
    row = options_rows[0]
    percentile = float(row["historical_percentile"]) if isinstance(row.get("historical_percentile"), (int, float)) else 50.0
    regime = str(row.get("dealer_regime", "Neutral"))
    if regime == "Accumulating" and percentile >= 60:
        score = 2
    elif regime == "Accumulating":
        score = 1
    elif regime == "Distributing" and percentile <= 40:
        score = -2
    elif regime == "Distributing":
        score = -1
    else:
        score = 0
    return ScoreComponent("Options Positioning", score, True, f"Dealer regime is {regime} with a historical percentile of {percentile:.1f}.")


def determine_trend_direction(index_rows: list[dict[str, object]]) -> str:
    above_50 = sum(1 for row in index_rows if isinstance(row.get("distance_50dma"), (int, float)) and float(row["distance_50dma"]) > 0)
    if above_50 >= 3:
        return "Up"
    if above_50 <= 1:
        return "Down"
    return "Mixed"


def map_regime_label(total_score: int) -> str:
    if total_score >= 5:
        return "Bullish"
    if total_score >= 1:
        return "Neutral"
    if total_score >= -3:
        return "Defensive"
    return "Risk-Off"


def map_posture(regime_label: str) -> str:
    return {
        "Bullish": "Fully Invested",
        "Neutral": "Normal Exposure",
        "Defensive": "Reduced Exposure",
        "Risk-Off": "Defensive",
    }[regime_label]


def build_regime_summary(index_rows: list[dict[str, object]], breadth_rows: list[dict[str, object]], sector_rows: list[dict[str, object]], options_rows: list[dict[str, object]], *, options_source_is_demo: bool) -> RegimeSummary:
    components = [
        score_trend(index_rows),
        score_momentum(index_rows),
        score_breadth(breadth_rows),
        score_volatility(index_rows),
        score_sector_leadership(sector_rows),
        score_options_positioning(options_rows, source_is_demo=options_source_is_demo),
    ]
    total_score = sum(component.score for component in components if component.active)
    regime_label = map_regime_label(total_score)
    trend_direction = determine_trend_direction(index_rows)
    posture = map_posture(regime_label)
    active_components = [component for component in components if component.active]
    strongest_positive = max(active_components, key=lambda component: component.score, default=None)
    strongest_negative = min(active_components, key=lambda component: component.score, default=None)
    positives = strongest_positive.name if strongest_positive and strongest_positive.score > 0 else "no strong positive factor"
    negatives = strongest_negative.name if strongest_negative and strongest_negative.score < 0 else "no major risk factor"
    short_summary = f"{regime_label} backdrop with a {trend_direction.lower()} trend bias. Strongest tailwind: {positives}. Main caution: {negatives}."
    return RegimeSummary(
        regime_label=regime_label,
        trend_direction=trend_direction,
        recommended_posture=posture,
        short_summary=short_summary,
        total_score=total_score,
        components=components,
    )
