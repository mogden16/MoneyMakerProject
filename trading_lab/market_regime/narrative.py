from __future__ import annotations

from trading_lab.market_regime.schema import MarketRegimeReport


def _format_pct(value: object) -> str | None:
    if isinstance(value, (int, float)):
        return f"{float(value):.1%}"
    return None


def generate_analyst_summary(report: MarketRegimeReport) -> str:
    lines: list[str] = []
    summary = report.summary
    lines.append(
        f"Overall regime is {summary.regime_label} with a {summary.trend_direction.lower()} trend and a {summary.recommended_posture.lower()} posture."
    )
    lines.append(summary.short_summary)

    if report.index_momentum.rows:
        above_50 = sum(1 for row in report.index_momentum.rows if isinstance(row.get("distance_50dma"), (int, float)) and float(row["distance_50dma"]) > 0)
        avg_20_values = [float(row["return_20d"]) for row in report.index_momentum.rows if isinstance(row.get("return_20d"), (int, float))]
        momentum_text = f"Momentum shows {above_50} of {len(report.index_momentum.rows)} tracked indexes above their 50 DMA"
        if avg_20_values:
            momentum_text += f" with an average 20-day return of {sum(avg_20_values) / len(avg_20_values):.1%}"
        soxx_overextended = any(bool(row.get("overextended_flag")) for row in report.index_momentum.rows if row.get("symbol") == "SOXX")
        if soxx_overextended:
            momentum_text += "; SOXX is flagged as overextended versus its 200 DMA"
        lines.append(momentum_text + ".")

    if report.breadth.rows:
        first_breadth = report.breadth.rows[0]
        breadth_50 = _format_pct((float(first_breadth["pct_above_50dma"]) / 100.0) if isinstance(first_breadth.get("pct_above_50dma"), (int, float)) else None)
        breadth_200 = _format_pct((float(first_breadth["pct_above_200dma"]) / 100.0) if isinstance(first_breadth.get("pct_above_200dma"), (int, float)) else None)
        breadth_text = "Breadth"
        if breadth_50 is not None and breadth_200 is not None:
            breadth_text += f" has {breadth_50} of the lead universe above the 50 DMA and {breadth_200} above the 200 DMA"
        if report.breadth.source.is_demo:
            breadth_text += "; this section is using demo placeholder data"
        lines.append(breadth_text + ".")

    if report.sector_leadership.rows:
        leaders = [row["symbol"] for row in report.sector_leadership.rows if isinstance(row.get("relative_strength_vs_spy"), (int, float)) and float(row["relative_strength_vs_spy"]) > 0]
        laggards = [row["symbol"] for row in report.sector_leadership.rows if isinstance(row.get("relative_strength_vs_spy"), (int, float)) and float(row["relative_strength_vs_spy"]) < 0]
        if leaders:
            lines.append(f"Sector leadership is strongest in {', '.join(leaders[:3])}.")
        elif laggards:
            lines.append(f"Sector leadership is weak, with relative lag in {', '.join(laggards[:3])}.")

    if report.options_positioning.rows:
        options_row = report.options_positioning.rows[0]
        if report.options_positioning.source.is_demo:
            lines.append("Dealer positioning is shown as demo-only placeholder data and is excluded from scoring.")
        else:
            regime = options_row.get("dealer_regime")
            percentile = options_row.get("historical_percentile")
            if regime is not None and percentile is not None:
                lines.append(f"Dealer positioning reads {regime} at the {float(percentile):.0f}th percentile of its historical range.")

    bullish_factors = [component.name for component in summary.components if component.active and component.score > 0]
    risk_factors = [component.name for component in summary.components if component.active and component.score < 0]
    if bullish_factors:
        lines.append(f"Bullish factors: {', '.join(bullish_factors)}.")
    if risk_factors:
        lines.append(f"Key risks: {', '.join(risk_factors)}.")

    watch_items: list[str] = []
    if any(bool(row.get("overextended_flag")) for row in report.index_momentum.rows):
        watch_items.append("SOXX mean reversion risk")
    if report.macro_calendar.rows:
        watch_items.append(str(report.macro_calendar.rows[0].get("event_name", "macro calendar")))
    if report.earnings_watch.rows:
        watch_items.append(str(report.earnings_watch.rows[0].get("ticker", "earnings")))
    if watch_items:
        lines.append(f"Watchlist for the upcoming week: {', '.join(watch_items)}.")

    return " ".join(line.strip() for line in lines if line.strip())
