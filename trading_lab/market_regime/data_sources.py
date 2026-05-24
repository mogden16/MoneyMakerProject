from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from trading_lab.data.providers.yfinance_provider import CacheStatus, YFinanceDataProvider
from trading_lab.market_regime.indicators import build_index_momentum_rows, build_sector_leadership_rows, extract_close_series
from trading_lab.market_regime.mock_data import (
    build_demo_breadth_rows,
    build_demo_earnings_rows,
    build_demo_macro_rows,
    build_demo_options_rows,
    build_demo_price_histories,
)
from trading_lab.market_regime.narrative import generate_analyst_summary
from trading_lab.market_regime.scoring import build_regime_summary
from trading_lab.market_regime.schema import DataSourceInfo, MarketRegimeReport, ReportSection
from trading_lab.market_regime.seasonality import build_seasonality_rows

INDEX_SYMBOLS = ["SPY", "QQQ", "IWM", "SOXX"]
SECTOR_SYMBOLS = ["XLK", "XLF", "XLE", "XLU", "XLV", "XLP", "XLY", "XLI", "XLC"]


def _status_to_source_info(statuses: list[CacheStatus], *, detail_prefix: str) -> DataSourceInfo:
    refreshed = sum(1 for status in statuses if status.performed_refresh)
    cached = sum(1 for status in statuses if status.used_cached_data and not status.performed_refresh)
    label = "Live" if refreshed else "Cached"
    detail = f"{detail_prefix}; refreshed={refreshed}, cached={cached}, yfinance-backed"
    return DataSourceInfo(label=label, detail=detail, is_demo=False)


def _fetch_price_histories(
    provider: YFinanceDataProvider,
    *,
    symbols: list[str],
    start_date: str,
    end_date: str,
    refresh_data: bool,
) -> tuple[dict[str, pd.DataFrame], DataSourceInfo, list[str]]:
    histories: dict[str, pd.DataFrame] = {}
    statuses: list[CacheStatus] = []
    warnings: list[str] = []
    failed_symbols: list[str] = []
    for symbol in symbols:
        try:
            histories[symbol] = provider.get_stock_bars(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe="1d",
                force_refresh=refresh_data,
            )
            status = provider.get_last_fetch_status(symbol)
            if status is not None:
                statuses.append(status)
        except Exception as exc:
            failed_symbols.append(symbol)
            warnings.append(f"{symbol}: live data unavailable, using demo placeholder ({exc})")
    if failed_symbols:
        histories.update(build_demo_price_histories(failed_symbols, end_date=end_date))
    if len(failed_symbols) == len(symbols):
        return histories, DataSourceInfo(label="Demo", detail="All ETF price histories are synthetic demo data.", is_demo=True), warnings
    if failed_symbols:
        return histories, DataSourceInfo(label="Live", detail=f"Mixed live/demo price set; demo fallback for {', '.join(failed_symbols)}.", is_demo=True), warnings
    return histories, _status_to_source_info(statuses, detail_prefix="ETF price histories"), warnings


def build_market_regime_report(
    provider: YFinanceDataProvider,
    *,
    start_date: str,
    end_date: str,
    refresh_data: bool,
    as_of_date: date | None = None,
) -> MarketRegimeReport:
    report_date = as_of_date or pd.Timestamp(end_date).date()
    price_symbols = list(dict.fromkeys(INDEX_SYMBOLS + SECTOR_SYMBOLS))
    price_histories, price_source, warnings = _fetch_price_histories(
        provider,
        symbols=price_symbols,
        start_date=start_date,
        end_date=end_date,
        refresh_data=refresh_data,
    )
    index_rows = build_index_momentum_rows(price_histories)
    sector_rows = build_sector_leadership_rows(price_histories, benchmark_symbol="SPY")
    spy_history = price_histories.get("SPY")
    spy_series = extract_close_series(spy_history) if spy_history is not None and not spy_history.empty else pd.Series(dtype=float)
    seasonality_source = price_source if not price_source.is_demo else DataSourceInfo(label="Demo", detail="Seasonality is computed from demo SPY history.", is_demo=True)
    breadth_source = DataSourceInfo(label="Demo", detail="Placeholder breadth adapter until constituent-level breadth inputs are connected.", is_demo=True)
    options_source = DataSourceInfo(label="Demo", detail="Placeholder dealer positioning section; hook this to CSV or options APIs later.", is_demo=True)
    macro_source = DataSourceInfo(label="Demo", detail="Local demo macro calendar rows.", is_demo=True)
    earnings_source = DataSourceInfo(label="Demo", detail="Local demo earnings watch rows.", is_demo=True)

    breadth_rows = build_demo_breadth_rows()
    options_rows = build_demo_options_rows()
    seasonality_rows = build_seasonality_rows(spy_series, as_of_date=report_date) if not spy_series.empty else []
    macro_rows = build_demo_macro_rows(report_date)
    earnings_rows = build_demo_earnings_rows(report_date)

    summary = build_regime_summary(
        index_rows=index_rows,
        breadth_rows=breadth_rows,
        sector_rows=sector_rows,
        options_rows=options_rows,
        options_source_is_demo=options_source.is_demo,
    )
    report = MarketRegimeReport(
        generated_at=datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        summary=summary,
        analyst_summary="",
        index_momentum=ReportSection(title="Index Momentum", rows=index_rows, source=price_source),
        breadth=ReportSection(title="Breadth", rows=breadth_rows, source=breadth_source, notes=["Constituent breadth adapter is not connected yet."]),
        options_positioning=ReportSection(title="Options / Dealer Positioning", rows=options_rows, source=options_source, notes=["Demo values do not affect the regime score."]),
        sector_leadership=ReportSection(title="Sector / Leadership", rows=sector_rows, source=price_source),
        seasonality=ReportSection(title="Seasonality", rows=seasonality_rows, source=seasonality_source),
        macro_calendar=ReportSection(title="Macro Calendar", rows=macro_rows, source=macro_source),
        earnings_watch=ReportSection(title="Earnings Watch", rows=earnings_rows, source=earnings_source),
        warnings=warnings,
    )
    analyst_summary = generate_analyst_summary(report)
    return MarketRegimeReport(
        generated_at=report.generated_at,
        summary=report.summary,
        analyst_summary=analyst_summary,
        index_momentum=report.index_momentum,
        breadth=report.breadth,
        options_positioning=report.options_positioning,
        sector_leadership=report.sector_leadership,
        seasonality=report.seasonality,
        macro_calendar=report.macro_calendar,
        earnings_watch=report.earnings_watch,
        warnings=report.warnings,
    )
