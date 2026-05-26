from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from dotenv import load_dotenv
from plotly.subplots import make_subplots

from trading_lab.backtest.audit import generate_audit_findings
from trading_lab.backtest.benchmark import BenchmarkDiagnostics, evaluate_benchmark_diagnostics
from trading_lab.backtest.corporate_actions import summarize_corporate_action_warnings
from trading_lab.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from trading_lab.backtest.metrics import calculate_sharpe_ratio, compute_summary_metrics, monthly_returns_table
from trading_lab.backtest.qualification import (
    evaluate_options_overlay_candidate,
    run_slippage_sensitivity,
    summarize_saved_sweep_stability,
    summarize_slippage_warnings,
)
from trading_lab.backtest.regime import compute_regime_metrics, summarize_regime_comments
from trading_lab.backtest.robustness import compute_robustness_score, parameter_stability_summary, profit_concentration_analysis
from trading_lab.backtest.sweep import run_parameter_sweep, summarize_parameter_stability
from trading_lab.backtest.train_test import run_train_test_analysis, split_data_by_date, split_data_by_percentage
from trading_lab.backtest.walk_forward import run_walk_forward_analysis
from trading_lab.data.database import TradingLabDatabase
from trading_lab.data.intraday import INTRADAY_MAX_HISTORY_DAYS, is_intraday_timeframe
from trading_lab.data.providers.yfinance_provider import CacheStatus, YFinanceDataProvider
from trading_lab.data.universes import get_universe_tickers, list_universe_names, normalize_ticker_list
from trading_lab.indicators.hma import hull_moving_average
from trading_lab.indicators.qqe import qqe_indicator
from trading_lab.indicators.rsi import relative_strength_index
from trading_lab.market_regime import build_market_regime_report
from trading_lab.paper.analytics import closed_trade_analytics
from trading_lab.paper.forward_engine import (
    ForwardPaperEngine,
    build_active_paper_strategy_payload,
    build_promotion_checklist,
    compare_forward_to_backtest,
    display_strategy_name,
    parse_strategy_parameters,
)
from trading_lab.paper.journal import close_paper_trade_payload, create_paper_trade_payload, open_paper_trade_payload, update_post_trade_review
from trading_lab.pybroker_lab import PyBrokerLabConfig, run_pybroker_lab
from trading_lab.pybroker_lab.audit import build_chart_metadata, build_raw_bars_export_name
from trading_lab.pybroker_lab.runner import PyBrokerRunResult, strategy_registry
from trading_lab.pybroker_lab.parity import CandleComparisonResult, compare_candle_frames, parse_tradingview_csv
from trading_lab.pybroker_lab.strategy_registry import fixed_strategy_library
from trading_lab.reports.charts import build_drawdown_chart, build_equity_chart, build_multi_drawdown_chart, build_multi_equity_chart
from trading_lab.signals.scanner import plan_trade_from_signal, scan_symbol_strategy
from trading_lab.spy_lab import (
    build_spy_backtest_config,
    build_spy_robustness_checklist,
    build_spy_search_summary_comment,
    build_spy_strategy,
    build_spy_workbench_config,
    generate_approved_spy_entry_presets,
    generate_approved_spy_exit_presets,
    generate_spy_search_combinations,
    get_spy_strategy_preset,
    get_spy_exit_structure,
    grade_spy_search_candidate,
    list_spy_exit_structures,
    list_spy_strategy_presets,
    prepare_spy_timeframe_bars,
    rank_spy_search_results,
    run_automated_spy_search,
    run_spy_exit_comparison,
    run_spy_parameter_stability,
    run_spy_robustness_checks,
    average_r_multiple,
    apply_spy_exit_structure,
    spy_daily_signal_status,
    spy_strategy_summary,
    spy_summary_commentary,
    summarize_exit_comparison_results,
    summarize_profit_concentration,
)
from trading_lab.strategies.breakout import BreakoutStrategy
from trading_lab.strategies.moving_average import MovingAverageCrossStrategy
from trading_lab.strategies.qqe_hma_strategy import QQEHMAStrategy
from trading_lab.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from trading_lab.strategies.trend_filter import TrendFilterStrategy


SESSION_RESULT_KEY = "ptl_current_result"
SESSION_DATA_KEY = "ptl_current_data"
SESSION_STATUSES_KEY = "ptl_current_statuses"
SESSION_WARNINGS_KEY = "ptl_current_warnings"
SESSION_RESEARCH_KEY = "ptl_current_research"
SESSION_META_KEY = "ptl_current_meta"
SESSION_SWEEP_KEY = "ptl_current_sweep"
SESSION_TRAIN_TEST_KEY = "ptl_current_train_test"
SESSION_WALK_FORWARD_KEY = "ptl_current_walk_forward"
SESSION_QUALIFICATION_KEY = "ptl_current_qualification"
SESSION_SLIPPAGE_KEY = "ptl_current_slippage"
SESSION_SCANNER_KEY = "ptl_current_scanner"
SESSION_TRADE_PLAN_KEY = "ptl_current_trade_plan"
SESSION_SCANNER_SNAPSHOT_KEY = "ptl_current_scanner_snapshot"
SESSION_FORWARD_UPDATE_KEY = "ptl_forward_update_result"
SESSION_SPY_LAB_KEY = "ptl_spy_lab_state"
SESSION_SPY_LAB_STABILITY_KEY = "ptl_spy_lab_stability"
SESSION_SPY_LAB_ROBUSTNESS_KEY = "ptl_spy_lab_robustness"
SESSION_SPY_LAB_EXIT_KEY = "ptl_spy_lab_exit_comparison"
SESSION_SPY_SEARCH_KEY = "ptl_spy_search_state"
SESSION_PYBROKER_LAB_KEY = "ptl_pybroker_lab_state"
PRIMARY_TAB_LABELS = ["SPY Workbench", "PyBroker Lab", "Forward Paper", "Research History", "Market Regime Report", "Data & Settings"]


def default_show_advanced_tools() -> bool:
    """Advanced mode stays off by default so the app opens in the simplified SPY workflow."""
    return False


def get_primary_tab_labels() -> list[str]:
    """Return the simplified top-level navigation labels."""
    return list(PRIMARY_TAB_LABELS)


def load_settings() -> dict[str, Any]:
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_strategy(strategy_name: str, params: dict[str, Any]):
    if strategy_name == "SPY 200-Day Trend Filter":
        return TrendFilterStrategy(**params)
    if strategy_name == "Moving Average Crossover":
        return MovingAverageCrossStrategy(**params)
    if strategy_name == "RSI Mean Reversion":
        return RSIMeanReversionStrategy(**params)
    if strategy_name == "Daily Breakout":
        return BreakoutStrategy(**params)
    if strategy_name == "QQE/HMA Daily":
        return QQEHMAStrategy(**params)
    raise ValueError(f"Unsupported strategy: {strategy_name}")


def default_strategy_params(strategy_name: str) -> dict[str, Any]:
    """Return a conservative baseline parameter set for cross-strategy comparison."""
    if strategy_name == "SPY 200-Day Trend Filter":
        return {"sma_length": 200}
    if strategy_name == "Moving Average Crossover":
        return {"fast_window": 20, "slow_window": 50}
    if strategy_name == "RSI Mean Reversion":
        return {"rsi_length": 14, "buy_threshold": 30.0, "sell_threshold": 55.0, "max_holding_days": 10, "trend_sma_window": 200}
    if strategy_name == "Daily Breakout":
        return {"lookback_window": 20}
    if strategy_name == "QQE/HMA Daily":
        return {
            "hma_length": 21,
            "rsi_length": 14,
            "rsi_smoothing": 5,
            "qqe_factor": 4.236,
            "atr_smoothing": 5,
            "require_hma_slope": True,
            "exit_on_hma_break": True,
            "exit_on_qqe_bearish": True,
        }
    raise ValueError(f"Unsupported strategy: {strategy_name}")


def strategy_internal_name(strategy_name: str) -> str:
    """Map a UI display name to the persisted internal strategy name."""
    return build_strategy(strategy_name, default_strategy_params(strategy_name)).name


def indicator_price_series(bars: pd.DataFrame, price_mode: str) -> pd.Series:
    price_column = "adj_close" if price_mode == "adjusted_price_mode" and "adj_close" in bars.columns else "close"
    return bars[price_column].astype(float)


def parse_range_input(raw: str, cast_type=float) -> list[int | float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [cast_type(value) for value in values]


def safe_json_loads(raw: str | None, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def extract_saved_run_strategy_params(run_record: dict[str, Any] | None) -> dict[str, Any]:
    """Extract strategy-only parameters from a stored backtest run record."""
    if not run_record:
        return {}
    parameters = safe_json_loads(run_record.get("parameters_json"), {})
    if not isinstance(parameters, dict):
        return {}
    config_keys = set(BacktestConfig.model_fields.keys())
    return {key: value for key, value in parameters.items() if key not in config_keys}


def extract_saved_run_config_defaults(run_record: dict[str, Any] | None) -> dict[str, Any]:
    """Extract forward-paper defaults from a stored backtest run record."""
    if not run_record:
        return {}
    parameters = safe_json_loads(run_record.get("parameters_json"), {})
    if not isinstance(parameters, dict):
        return {}
    config_keys = set(BacktestConfig.model_fields.keys())
    defaults = {key: parameters[key] for key in config_keys if key in parameters}
    defaults["initial_capital"] = float(run_record.get("initial_capital", defaults.get("initial_capital", 100000.0)) or 100000.0)
    defaults["price_mode"] = str(run_record.get("price_mode", defaults.get("price_mode", "raw_price_mode")) or "raw_price_mode")
    return defaults


def normalize_strategy_display_name(name: str) -> str:
    """Normalize internal strategy identifiers into the UI display names."""
    return display_strategy_name(name)


def collect_data(
    provider: YFinanceDataProvider,
    symbols: list[str],
    start_date: str,
    end_date: str,
    refresh_data: bool,
    benchmark_symbol: str | None = None,
    timeframe: str = "1d",
) -> tuple[dict[str, pd.DataFrame], list[CacheStatus], list[str]]:
    data_by_symbol: dict[str, pd.DataFrame] = {}
    statuses: list[CacheStatus] = []
    validation_warnings: list[str] = []
    fetch_symbols = list(dict.fromkeys(symbols + ([benchmark_symbol] if benchmark_symbol else [])))
    for symbol in fetch_symbols:
        bars = provider.get_stock_bars(symbol=symbol, start_date=start_date, end_date=end_date, timeframe=timeframe, force_refresh=refresh_data)
        data_by_symbol[symbol] = bars
        status = provider.get_last_fetch_status(symbol)
        if status is not None:
            statuses.append(status)
            validation_warnings.extend([f"{symbol}: {warning}" for warning in status.validation_warnings])
    return data_by_symbol, statuses, validation_warnings


def collect_pybroker_data(
    provider: YFinanceDataProvider,
    symbols: list[str],
    start_date: str,
    end_date: str,
    refresh_data: bool,
    timeframe: str,
) -> tuple[pd.DataFrame, list[CacheStatus], list[str]]:
    data_by_symbol, statuses, validation_warnings = collect_data(
        provider,
        symbols,
        start_date,
        end_date,
        refresh_data,
        benchmark_symbol=None,
        timeframe=timeframe,
    )
    frames = [bars.copy() for bars in data_by_symbol.values() if not bars.empty]
    if not frames:
        return pd.DataFrame(), statuses, validation_warnings
    combined = pd.concat(frames, ignore_index=True)
    return combined, statuses, validation_warnings


def pybroker_strategy_labels() -> dict[str, str]:
    labels = {"all": "All default strategies"}
    labels.update({strategy_id: template.display_name for strategy_id, template in fixed_strategy_library().items()})
    return labels


def render_pybroker_strategy_description(strategy_name: str) -> None:
    if strategy_name == "all":
        st.info("This runs the full fixed strategy library with hard-coded defaults. Indicator settings remain in code and are not editable in the app.")
        return
    template = fixed_strategy_library()[strategy_name]
    with st.container(border=True):
        st.subheader(template.display_name)
        st.write(template.description)
        st.caption("Fixed logic and indicator settings remain hard-coded in the codebase. The UI only exposes strategy selection, data range, benchmark, and backtest-level execution controls.")


def render_pybroker_run_result(state: dict[str, Any]) -> None:
    result: PyBrokerRunResult = state["result"]
    summary = result.summary
    if summary.empty:
        st.info("No PyBroker result is loaded yet.")
        return
    st.success(f"PyBroker run completed. Output folder: `{state['output_dir']}`")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Strategies run", int(summary["strategy_name"].nunique()))
    metric_cols[1].metric("PASS count", int(summary["status"].eq("PASS").sum()))
    metric_cols[2].metric("FAIL count", int(summary["status"].eq("FAIL").sum()))
    metric_cols[3].metric("Date range", f"{state['config']['start_date']} to {state['config']['end_date']}")
    st.subheader("Actual Data Used")
    if result.actual_data_used.empty:
        st.info("No actual-data rows were recorded for this run.")
    else:
        st.dataframe(result.actual_data_used, use_container_width=True, hide_index=True)
    if result.data_quality_audits:
        st.subheader("Data Audit")
        for strategy_id, audit_frame in result.data_quality_audits.items():
            with st.expander(f"{fixed_strategy_library()[strategy_id].display_name} Data Audit", expanded=False):
                st.dataframe(audit_frame, use_container_width=True, hide_index=True)

    st.subheader("Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    if not result.benchmark_summary.empty:
        st.subheader("Benchmark Comparison")
        st.dataframe(result.benchmark_summary, use_container_width=True, hide_index=True)
    st.subheader("Strategy Metrics")
    st.dataframe(result.strategy_metrics, use_container_width=True, hide_index=True)
    st.subheader("Benchmark Metrics")
    st.dataframe(result.benchmark_metrics, use_container_width=True, hide_index=True)
    if not result.equity_curve.empty:
        chart_frame = result.equity_curve.copy()
        chart_frame["series"] = chart_frame["strategy_name"] + " (" + chart_frame["curve_type"] + ")"
        st.plotly_chart(px.line(chart_frame, x="date", y="equity", color="series", title="PyBroker Equity Curves"), use_container_width=True)
    if not result.walkforward_windows.empty:
        st.subheader("Walk-Forward Windows")
        st.dataframe(result.walkforward_windows, use_container_width=True, hide_index=True)
    if not result.bootstrap_metrics.empty:
        st.subheader("Bootstrap Confidence Intervals")
        st.dataframe(result.bootstrap_metrics, use_container_width=True, hide_index=True)
    st.subheader("Trade Audit")
    if not result.trade_audit.empty:
        st.dataframe(result.trade_audit, use_container_width=True, hide_index=True)
    else:
        st.info("No trade audit rows were generated for this run.")
        st.dataframe(pd.DataFrame(columns=[
            "strategy_id",
            "symbol",
            "timeframe",
            "signal_timestamp",
            "entry_timestamp",
            "entry_price",
            "entry_reason",
            "exit_timestamp",
            "exit_price",
            "exit_reason",
            "shares_contracts",
            "position_value",
            "percent_return",
            "dollar_pnl",
            "holding_period",
            "holding_period_bars",
            "entry_indicator_values",
            "exit_indicator_values",
        ]), use_container_width=True, hide_index=True)
    if not result.trades.empty:
        st.subheader("Trades")
        st.dataframe(result.trades, use_container_width=True, hide_index=True)

    if result.debug_frames:
        st.subheader("Debug Charts")
        for strategy_id, debug_frame in result.debug_frames.items():
            if debug_frame.empty:
                continue
            trade_audit = result.trade_audit[result.trade_audit["strategy_id"] == strategy_id] if not result.trade_audit.empty else pd.DataFrame()
            with st.container(border=True):
                st.markdown(f"**{fixed_strategy_library()[strategy_id].display_name}**")
                range_cols = st.columns(3)
                range_mode = range_cols[0].selectbox(
                    "Chart range",
                    ["full_backtest_range", "last_1_trading_day", "last_5_trading_days", "around_selected_trade"],
                    format_func=lambda value: {
                        "full_backtest_range": "Full backtest range",
                        "last_1_trading_day": "Last 1 trading day",
                        "last_5_trading_days": "Last 5 trading days",
                        "around_selected_trade": "Around selected trade",
                    }[value],
                    key=f"pybroker_debug_range_{strategy_id}",
                )
                trade_index = None
                if range_mode == "around_selected_trade" and not trade_audit.empty:
                    trade_options = trade_audit.reset_index(drop=True).index.tolist()
                    trade_index = range_cols[1].selectbox(
                        "Trade",
                        trade_options,
                        format_func=lambda value: format_debug_trade_label(trade_audit.reset_index(drop=True).iloc[int(value)]),
                        key=f"pybroker_debug_trade_{strategy_id}",
                    )
                show_legend = range_cols[2].checkbox("Show legend", value=False, key=f"pybroker_debug_legend_{strategy_id}")
                filtered_frame, filtered_trade_audit = filter_pybroker_debug_window(
                    debug_frame,
                    trade_audit,
                    range_mode=range_mode,
                    selected_trade_index=trade_index,
                )
                actual_data_row = None
                if not result.actual_data_used.empty:
                    actual_match = result.actual_data_used[result.actual_data_used["strategy_id"] == strategy_id]
                    if not actual_match.empty:
                        actual_data_row = actual_match.iloc[0].to_dict()
                metadata_symbol = str(filtered_frame["symbol"].iloc[0]) if "symbol" in filtered_frame.columns and not filtered_frame.empty else "SPY"
                chart_metadata = build_chart_metadata(actual_data_row=actual_data_row, chart_frame=filtered_frame, symbol=metadata_symbol)
                st.dataframe(chart_metadata, use_container_width=True, hide_index=True)
                st.plotly_chart(build_pybroker_debug_chart(filtered_frame, filtered_trade_audit, strategy_id, show_legend=show_legend), use_container_width=True, config={"displayModeBar": True, "displaylogo": False, "responsive": True})
                indicator_debug = result.indicator_debug_tables.get(strategy_id, pd.DataFrame())
                st.caption("Entry and exit indicator values used by the strategy on the audited trades.")
                st.dataframe(indicator_debug, use_container_width=True, hide_index=True)
                higher_timeframe_check = result.higher_timeframe_checks.get(strategy_id, pd.DataFrame())
                with st.expander("Higher Timeframe Lookahead Check", expanded=False):
                    if higher_timeframe_check.empty:
                        st.info("This strategy does not use higher timeframe data.")
                    else:
                        st.dataframe(higher_timeframe_check, use_container_width=True, hide_index=True)
                raw_bars = result.actual_bars.get(strategy_id, pd.DataFrame())
                with st.expander("Raw Bars Preview", expanded=False):
                    if raw_bars.empty:
                        st.info("No raw bars were retained for this strategy.")
                    else:
                        preview_columns = [column for column in ["timestamp", "open", "high", "low", "close", "volume"] if column in raw_bars.columns or column == "timestamp"]
                        raw_preview = raw_bars.copy()
                        if "date" in raw_preview.columns and "timestamp" not in raw_preview.columns:
                            raw_preview["timestamp"] = pd.to_datetime(raw_preview["date"])
                        st.caption("First 10 rows")
                        st.dataframe(raw_preview.loc[:, preview_columns].head(10), use_container_width=True, hide_index=True)
                        st.caption("Last 10 rows")
                        st.dataframe(raw_preview.loc[:, preview_columns].tail(10), use_container_width=True, hide_index=True)
                        export_name = build_raw_bars_export_name(metadata_symbol, str(actual_data_row.get("timeframe") if actual_data_row else state["config"]["timeframe"]), raw_preview)
                        st.download_button(
                            "Download raw bars CSV",
                            data=raw_preview.loc[:, preview_columns].to_csv(index=False).encode("utf-8"),
                            file_name=export_name,
                            mime="text/csv",
                            key=f"pybroker_raw_bars_download_{strategy_id}",
                        )

    report_path = Path(state["output_dir"]) / "report.md"
    if report_path.exists():
        with st.expander("Report Markdown", expanded=False):
            st.markdown(report_path.read_text(encoding="utf-8"))

    st.subheader("Downloads")
    for filename in [
        "summary.csv",
        "strategy_metrics.csv",
        "benchmark_metrics.csv",
        "actual_data_used.csv",
        "benchmark_summary.csv",
        "trade_audit.csv",
        "trades.csv",
        "equity_curve.csv",
        "bootstrap_metrics.csv",
        "walkforward_windows.csv",
        "report.md",
    ]:
        path = Path(state["output_dir"]) / filename
        if not path.exists():
            continue
        mime = "text/markdown" if path.suffix == ".md" else "text/csv"
        st.download_button(
            f"Download {filename}",
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            key=f"pybroker_download_{filename}",
        )    


def format_debug_trade_label(trade_row: pd.Series) -> str:
    entry_timestamp = pd.to_datetime(trade_row.get("entry_timestamp"))
    exit_timestamp = pd.to_datetime(trade_row.get("exit_timestamp"))
    entry_text = "unknown" if pd.isna(entry_timestamp) else entry_timestamp.strftime("%Y-%m-%d %H:%M")
    exit_text = "open" if pd.isna(exit_timestamp) else exit_timestamp.strftime("%Y-%m-%d %H:%M")
    return f"{entry_text} to {exit_text}"


def filter_pybroker_debug_window(
    debug_frame: pd.DataFrame,
    trade_audit: pd.DataFrame,
    *,
    range_mode: str,
    selected_trade_index: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chart_frame = debug_frame.copy().sort_values("timestamp").reset_index(drop=True)
    filtered_trade_audit = trade_audit.copy().reset_index(drop=True)
    if chart_frame.empty:
        return chart_frame, filtered_trade_audit
    if range_mode == "last_1_trading_day":
        last_session = chart_frame["session_date"].iloc[-1]
        chart_frame = chart_frame[chart_frame["session_date"] == last_session].reset_index(drop=True)
    elif range_mode == "last_5_trading_days":
        sessions = chart_frame["session_date"].dropna().drop_duplicates().tolist()
        selected_sessions = set(sessions[-5:])
        chart_frame = chart_frame[chart_frame["session_date"].isin(selected_sessions)].reset_index(drop=True)
    elif range_mode == "around_selected_trade" and selected_trade_index is not None and not filtered_trade_audit.empty:
        trade_row = filtered_trade_audit.iloc[int(selected_trade_index)]
        entry_timestamp = pd.to_datetime(trade_row.get("entry_timestamp"))
        exit_timestamp = pd.to_datetime(trade_row.get("exit_timestamp"))
        entry_matches = chart_frame.index[chart_frame["timestamp"] == entry_timestamp].tolist()
        exit_matches = chart_frame.index[chart_frame["timestamp"] == exit_timestamp].tolist()
        entry_idx = entry_matches[0] if entry_matches else max(chart_frame["timestamp"].searchsorted(entry_timestamp) - 1, 0)
        exit_idx = exit_matches[0] if exit_matches else min(chart_frame["timestamp"].searchsorted(exit_timestamp), len(chart_frame) - 1)
        start_idx = max(entry_idx - 20, 0)
        end_idx = min(exit_idx + 20, len(chart_frame) - 1)
        chart_frame = chart_frame.iloc[start_idx : end_idx + 1].reset_index(drop=True)
        window_start = chart_frame["timestamp"].min()
        window_end = chart_frame["timestamp"].max()
        if not filtered_trade_audit.empty:
            filtered_trade_audit = filtered_trade_audit[
                pd.to_datetime(filtered_trade_audit["entry_timestamp"]).between(window_start, window_end)
                | pd.to_datetime(filtered_trade_audit["exit_timestamp"]).between(window_start, window_end)
            ].reset_index(drop=True)
    return chart_frame, filtered_trade_audit


def build_pybroker_debug_chart(debug_frame: pd.DataFrame, trade_audit: pd.DataFrame, strategy_id: str, *, show_legend: bool = False):
    chart_frame = debug_frame.copy().sort_values("timestamp").reset_index(drop=True)
    has_oscillator_panel = any(column in chart_frame.columns for column in ["combined_momentum", "qqe_rsi_ma", "qqe_trailing_line", "rsi_ma"])
    fig = make_subplots(
        rows=3 if has_oscillator_panel else 2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.15, 0.2] if has_oscillator_panel else [0.75, 0.25],
    )
    fig.add_trace(
        go.Candlestick(
            x=chart_frame["timestamp"],
            open=chart_frame["open"],
            high=chart_frame["high"],
            low=chart_frame["low"],
            close=chart_frame["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    overlay_styles = {
        "trail": dict(name="Blackflag Trail", color="#1f77b4", dash="solid"),
        "fib1": dict(name="Fib1", color="#17becf", dash="dot"),
        "fib2": dict(name="Fib2", color="#2ca02c", dash="dot"),
        "fib3": dict(name="Fib3", color="#9467bd", dash="dot"),
        "higher_hma": dict(name="HMA", color="#ff7f0e", dash="solid"),
        "fast_ema": dict(name="Fast EMA", color="#d62728", dash="solid"),
        "slow_ema": dict(name="Slow EMA", color="#8c564b", dash="solid"),
    }
    for column, style in overlay_styles.items():
        if column in chart_frame.columns:
            fig.add_trace(
                go.Scatter(
                    x=chart_frame["timestamp"],
                    y=chart_frame[column],
                    mode="lines",
                    name=style["name"],
                    line=dict(color=style["color"], dash=style["dash"]),
                ),
                row=1,
                col=1,
            )

    if not trade_audit.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(trade_audit["entry_timestamp"]),
                y=pd.to_numeric(trade_audit["entry_price"]),
                mode="markers",
                marker=dict(symbol="triangle-up", size=11, color="#2ca02c"),
                name="Entries",
                customdata=trade_audit[["entry_reason"]].to_numpy(),
                hovertemplate="Entry<br>%{x}<br>Price=%{y:.2f}<br>Reason=%{customdata[0]}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(trade_audit["exit_timestamp"]),
                y=pd.to_numeric(trade_audit["exit_price"]),
                mode="markers",
                marker=dict(symbol="triangle-down", size=11, color="#d62728"),
                name="Exits",
                customdata=trade_audit[["exit_reason"]].to_numpy(),
                hovertemplate="Exit<br>%{x}<br>Price=%{y:.2f}<br>Reason=%{customdata[0]}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    volume_colors = [
        "#2ca02c" if float(close) >= float(open_) else "#d62728"
        for open_, close in zip(chart_frame["open"], chart_frame["close"], strict=False)
    ]
    fig.add_trace(
        go.Bar(x=chart_frame["timestamp"], y=chart_frame["volume"], marker_color=volume_colors, name="Volume"),
        row=2,
        col=1,
    )

    if has_oscillator_panel:
        oscillator_row = 3
        oscillator_columns = [
            ("combined_momentum", "Combined Momentum", "#1f77b4"),
            ("rsi_ma", "QQE RSI MA", "#17becf"),
            ("qqe_rsi_ma", "QQE RSI MA", "#17becf"),
            ("qqe_trailing_line", "QQE Trailing Line", "#ff7f0e"),
        ]
        plotted = set()
        for column, label, color in oscillator_columns:
            if column in chart_frame.columns and column not in plotted:
                fig.add_trace(
                    go.Scatter(x=chart_frame["timestamp"], y=chart_frame[column], mode="lines", name=label, line=dict(color=color)),
                    row=oscillator_row,
                    col=1,
                )
                plotted.add(column)
        fig.add_hline(y=50.0, line_dash="dash", line_color="#7f7f7f", row=oscillator_row, col=1)
        strategy_settings = fixed_strategy_library()[strategy_id].fixed_settings
        for level_key in ["rsi_super_oversold", "rsi_oversold", "rsi_low_neutral", "rsi_high_neutral", "rsi_overbought", "rsi_super_overbought"]:
            if level_key in strategy_settings:
                fig.add_hline(y=float(strategy_settings[level_key]), line_dash="dot", line_color="#bdbdbd", row=oscillator_row, col=1)

    fig.update_layout(
        title=dict(text=f"{fixed_strategy_library()[strategy_id].display_name} Debug Chart", y=0.985, x=0.02, xanchor="left"),
        legend_orientation="h",
        showlegend=show_legend,
        legend=dict(yanchor="bottom", y=1.01, xanchor="left", x=0.0, bgcolor="rgba(255,255,255,0.7)", font=dict(size=10)),
        margin=dict(l=70, r=24, t=95, b=28),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=980 if has_oscillator_panel else 840,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, title_standoff=8, automargin=True, tickfont=dict(size=11))
    fig.update_yaxes(title_text="Volume", row=2, col=1, title_standoff=8, automargin=True, tickfont=dict(size=10))
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=not has_oscillator_panel, row=2, col=1)
    if has_oscillator_panel:
        fig.update_yaxes(title_text="QQE / RSI", row=3, col=1, title_standoff=8, automargin=True, tickfont=dict(size=10))
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=2, col=1)
        fig.update_xaxes(showticklabels=True, row=3, col=1)
    return fig


def build_candle_comparison_chart(comparison: CandleComparisonResult):
    frame = comparison.merged.copy().sort_values("timestamp").reset_index(drop=True)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.38, 0.38, 0.24],
        subplot_titles=("yfinance Candles", "TradingView Candles", "Close Difference"),
    )
    fig.add_trace(
        go.Candlestick(
            x=frame["timestamp"],
            open=frame["open_yfinance"],
            high=frame["high_yfinance"],
            low=frame["low_yfinance"],
            close=frame["close_yfinance"],
            name="yfinance",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Candlestick(
            x=frame["timestamp"],
            open=frame["open_tradingview"],
            high=frame["high_tradingview"],
            low=frame["low_tradingview"],
            close=frame["close_tradingview"],
            name="TradingView",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=frame["timestamp"], y=frame["close_diff"], mode="lines", name="Close Diff"),
        row=3,
        col=1,
    )
    fig.update_layout(title="yfinance vs TradingView Candle Comparison", margin=dict(l=20, r=20, t=60, b=20), hovermode="x unified", xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="yfinance", row=1, col=1)
    fig.update_yaxes(title_text="TradingView", row=2, col=1)
    fig.update_yaxes(title_text="Close Diff", row=3, col=1)
    return fig


def load_tradingview_csv_payload(uploaded_file, local_path: str) -> bytes | None:
    if uploaded_file is not None:
        return uploaded_file.getvalue()
    if local_path.strip():
        path = Path(local_path.strip())
        if not path.exists():
            raise FileNotFoundError(f"TradingView CSV path was not found: {path}")
        return path.read_bytes()
    return None


def render_tradingview_parity_tool(
    *,
    provider: YFinanceDataProvider,
    symbols_raw: str,
    start_date: str,
    end_date: str,
    refresh_data: bool,
    timeframe: str,
) -> None:
    with st.expander("yfinance vs TradingView Candle Parity", expanded=False):
        st.caption("Use this to compare the app's yfinance candles against a TradingView CSV export before changing any strategy logic.")
        if timeframe != "5m":
            st.info("This parity tool is currently configured for 5-minute candle comparison. Switch the PyBroker Lab timeframe to `5m` for a like-for-like comparison.")
        compare_cols = st.columns(4)
        timezone = compare_cols[0].selectbox("Timezone", ["America/New_York", "UTC"], index=0, key="tv_parity_timezone")
        timestamp_basis = compare_cols[1].selectbox("TradingView timestamp basis", ["bar_start", "bar_end"], index=0, key="tv_parity_timestamp_basis")
        shift_dataset = compare_cols[2].selectbox(
            "Alignment shift",
            ["none", "shift_yfinance_forward_1_bar", "shift_tradingview_forward_1_bar"],
            index=0,
            format_func=lambda value: {
                "none": "No shift",
                "shift_yfinance_forward_1_bar": "Shift yfinance forward 1 bar",
                "shift_tradingview_forward_1_bar": "Shift TradingView forward 1 bar",
            }[value],
            key="tv_parity_shift",
        )
        regular_hours_only = compare_cols[3].checkbox("Regular hours only", value=True, key="tv_parity_regular_hours")
        upload_cols = st.columns(2)
        uploaded_file = upload_cols[0].file_uploader("TradingView CSV export", type=["csv"], key="tv_parity_upload")
        local_path = upload_cols[1].text_input("Or local CSV path", value="", key="tv_parity_local_path")
        if st.button("Run Candle Parity Comparison", key="tv_parity_run_button"):
            symbols = normalize_ticker_list(symbols_raw)
            if not symbols:
                st.error("A symbol is required to compare candles.")
                return
            try:
                payload = load_tradingview_csv_payload(uploaded_file, local_path)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
            if payload is None:
                st.error("Upload a TradingView CSV or provide a local CSV path.")
                return
            try:
                tradingview_frame = parse_tradingview_csv(
                    payload,
                    timeframe="5m",
                    timezone=timezone,
                    timestamp_basis=timestamp_basis,
                    symbol=symbols[0],
                )
                yfinance_frame = provider.get_stock_bars(
                    symbols[0],
                    start_date,
                    end_date,
                    timeframe="5m",
                    force_refresh=refresh_data,
                )
                comparison = compare_candle_frames(
                    yfinance_frame,
                    tradingview_frame,
                    symbol=symbols[0],
                    timeframe="5m",
                    timezone=timezone,
                    regular_hours_only=regular_hours_only,
                    timestamp_basis=timestamp_basis,
                    shift_dataset=shift_dataset,
                )
            except Exception as exc:  # pragma: no cover - surfaced to the UI
                st.error(f"Parity comparison failed: {exc}")
                return
            st.subheader("Comparison Summary")
            st.dataframe(comparison.summary, use_container_width=True, hide_index=True)
            st.subheader("Worst 25 Mismatched Bars")
            st.dataframe(comparison.worst_mismatches, use_container_width=True, hide_index=True)
            st.plotly_chart(build_candle_comparison_chart(comparison), use_container_width=True)
            st.download_button(
                "Export candle comparison mismatches CSV",
                data=comparison.worst_mismatches.to_csv(index=False).encode("utf-8"),
                file_name="tradingview_candle_mismatches.csv",
                mime="text/csv",
                key="tv_parity_download_mismatches",
            )


def render_pybroker_lab_workspace(
    *,
    provider: YFinanceDataProvider,
    start_date: str,
    end_date: str,
    refresh_data: bool,
    base_config: BacktestConfig,
) -> None:
    st.header("PyBroker Lab")
    st.caption("Server-side PyBroker backtests for a fixed library of default Thinkorswim-era strategies, evaluated against buy-and-hold on the same symbol and period.")
    st.info("Indicator settings are hard-coded in the codebase. The UI only exposes strategy selection and normal backtest inputs.")

    labels = pybroker_strategy_labels()
    registry = strategy_registry()
    strategy_options = ["all", *registry.keys()]
    selected_strategy = st.selectbox("Strategy", strategy_options, format_func=lambda value: labels.get(value, value), key="pybroker_strategy")
    render_pybroker_strategy_description(selected_strategy)

    library = fixed_strategy_library()
    timeframe_options = ["15m", "5m"]
    if selected_strategy != "all":
        timeframe_options = list(library[selected_strategy].supported_timeframes)
    default_symbols = "SPY"
    setup_cols = st.columns(4)
    symbols_raw = setup_cols[0].text_input("Symbols", value=default_symbols, key="pybroker_symbols", help="These fixed strategy templates run on the symbols you provide and compare the result to the benchmark symbol.")
    benchmark_symbol = setup_cols[1].text_input("Benchmark", value="SPY", key="pybroker_benchmark").upper().strip()
    timeframe = setup_cols[2].selectbox("Timeframe", timeframe_options, key="pybroker_timeframe", help="These strategy templates are intraday and execute on completed bars with next-bar fills.")
    walkforward_windows = int(setup_cols[3].number_input("Walk-forward windows", min_value=2, max_value=10, value=3, key="pybroker_windows"))
    if is_intraday_timeframe(timeframe):
        requested_days = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days, 0)
        if requested_days > INTRADAY_MAX_HISTORY_DAYS:
            effective_start = (pd.Timestamp(end_date) - pd.Timedelta(days=INTRADAY_MAX_HISTORY_DAYS)).date().isoformat()
            st.warning(
                f"Intraday bars are fetched on demand and cached, but the current yfinance path only supports roughly the most recent "
                f"{INTRADAY_MAX_HISTORY_DAYS} days. This run will clamp the effective intraday start date to about `{effective_start}`."
            )
        else:
            st.caption(
                f"Intraday bars are fetched on demand and cached in DuckDB. The current provider supports about the most recent "
                f"{INTRADAY_MAX_HISTORY_DAYS} days of `15m`/`5m` history."
            )

    config_cols = st.columns(5)
    initial_cash = float(config_cols[0].number_input("Initial cash", min_value=1000.0, value=float(base_config.initial_capital), key="pybroker_initial_cash"))
    train_size = float(config_cols[1].number_input("Train size", min_value=0.5, max_value=0.95, value=0.7, step=0.05, key="pybroker_train_size"))
    warmup_bars = int(config_cols[2].number_input("Warmup bars", min_value=20, max_value=400, value=200, key="pybroker_warmup"))
    slippage_bps = float(config_cols[3].number_input("Slippage bps", min_value=0.0, value=float(base_config.slippage_pct * 10000.0), format="%.2f", key="pybroker_slippage_bps"))
    commission_bps = float(config_cols[4].number_input("Commission bps", min_value=0.0, value=1.0, format="%.2f", key="pybroker_commission_bps"))
    bootstrap_sample_size = int(st.number_input("Bootstrap sample size", min_value=50, max_value=5000, value=1000, step=50, key="pybroker_bootstrap_size"))
    sizing_cols = st.columns(2)
    sizing_method = sizing_cols[0].selectbox(
        "Position sizing",
        ["percent_equity", "fixed_dollar", "fixed_shares"],
        index=0,
        format_func=lambda value: {
            "percent_equity": "100% equity allocation",
            "fixed_dollar": "Fixed dollar allocation",
            "fixed_shares": "Fixed share quantity",
        }[value],
        key="pybroker_sizing_method",
    )
    if sizing_method == "percent_equity":
        percent_label = sizing_cols[1].selectbox("Equity allocation", [1.0, 0.5], index=0, format_func=lambda value: "100% equity" if float(value) == 1.0 else "50% equity", key="pybroker_sizing_percent")
        sizing_value = float(percent_label)
    elif sizing_method == "fixed_dollar":
        sizing_value = float(sizing_cols[1].number_input("Fixed dollar allocation", min_value=0.0, value=float(initial_cash), key="pybroker_sizing_fixed_dollar"))
    else:
        sizing_value = float(sizing_cols[1].number_input("Fixed share quantity", min_value=0.0, value=100.0, key="pybroker_sizing_fixed_shares"))

    st.caption("No optimizer is used here. Fixed strategy logic stays in code, while sizing is a backtest-level control recorded in the results alongside the actual data range and benchmark comparison.")
    render_tradingview_parity_tool(
        provider=provider,
        symbols_raw=symbols_raw,
        start_date=start_date,
        end_date=end_date,
        refresh_data=refresh_data,
        timeframe=timeframe,
    )
    if st.button("Run PyBroker Lab", key="pybroker_run_button", type="primary"):
        symbols = normalize_ticker_list(symbols_raw)
        if benchmark_symbol and benchmark_symbol not in symbols:
            symbols.append(benchmark_symbol)
        if not symbols:
            st.error("At least one symbol is required.")
        else:
            output_dir = Path("outputs/pybroker_lab") / f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
            with st.spinner("Running PyBroker research on the Streamlit server..."):
                price_data, statuses, validation_warnings = collect_pybroker_data(
                    provider,
                    symbols,
                    start_date,
                    end_date,
                    refresh_data,
                    timeframe,
                )
                if price_data.empty:
                    st.error("No bars were available for the requested symbols, timeframe, and date range.")
                else:
                    config = PyBrokerLabConfig(
                        symbols=tuple(symbols),
                        benchmark_symbol=benchmark_symbol or "SPY",
                        start_date=start_date,
                        end_date=end_date,
                        initial_cash=initial_cash,
                        timeframe=timeframe,
                        commission_bps=commission_bps,
                        slippage_bps=slippage_bps,
                        warmup_bars=warmup_bars,
                        train_size=train_size,
                        walkforward_windows=walkforward_windows,
                        bootstrap_sample_size=bootstrap_sample_size,
                        output_dir=output_dir,
                        strategy_params={},
                        sizing_method=sizing_method,
                        sizing_value=sizing_value,
                    )
                    result = run_pybroker_lab(config, strategy_name=selected_strategy, data_frame=price_data, statuses=statuses)
                    st.session_state[SESSION_PYBROKER_LAB_KEY] = {
                        "result": result,
                        "output_dir": str(result.output_dir),
                        "config": {
                            "strategy": selected_strategy,
                            "symbols": ",".join(symbols),
                            "benchmark_symbol": benchmark_symbol,
                            "timeframe": timeframe,
                            "sizing_method": sizing_method,
                            "sizing_value": sizing_value,
                            "start_date": start_date,
                            "end_date": end_date,
                        },
                        "statuses": statuses,
                        "validation_warnings": validation_warnings,
                        "price_data": price_data,
                    }

    state = st.session_state.get(SESSION_PYBROKER_LAB_KEY)
    if state:
        render_pybroker_run_result(state)
        render_warning_list(state.get("validation_warnings", []), "No PyBroker data warnings were generated.")


def build_indicator_preview_frame(
    bars: pd.DataFrame,
    indicator_name: str,
    *,
    price_mode: str,
    indicator_params: dict[str, Any],
) -> pd.DataFrame:
    frame = bars.copy().sort_values("timestamp").reset_index(drop=True)
    price_series = indicator_price_series(frame, price_mode)
    frame["display_price"] = price_series
    if indicator_name == "HMA":
        frame["hma"] = hull_moving_average(price_series, int(indicator_params["length"]))
    elif indicator_name == "RSI":
        frame["rsi"] = relative_strength_index(price_series, int(indicator_params["length"]))
    elif indicator_name == "QQE":
        qqe_frame = qqe_indicator(
            price_series,
            rsi_length=int(indicator_params["rsi_length"]),
            rsi_smoothing=int(indicator_params["rsi_smoothing"]),
            qqe_factor=float(indicator_params["qqe_factor"]),
            atr_smoothing=int(indicator_params["atr_smoothing"]),
        )
        frame = pd.concat([frame, qqe_frame.reset_index(drop=True)], axis=1)
    else:
        raise ValueError(f"Unsupported indicator preview: {indicator_name}")
    return frame


def render_metrics(metrics: dict[str, float | int], limit: int | None = None) -> None:
    items = list(metrics.items())[:limit] if limit is not None else list(metrics.items())
    metric_columns = st.columns(4)
    for idx, (label, value) in enumerate(items):
        metric_columns[idx % 4].metric(label, f"{value:.4f}" if isinstance(value, float) else value)


def render_freshness_status(statuses: list[CacheStatus]) -> None:
    if not statuses:
        st.info("No data has been loaded yet.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "symbol": status.symbol,
                    "timeframe": status.timeframe,
                    "cache_status": status.cache_status,
                    "latest_cached_session": status.latest_cached_session,
                    "expected_latest_session": status.expected_latest_session,
                    "calendar": status.calendar_name,
                    "calendar_fallback": status.using_calendar_fallback,
                    "used_cached_data": status.used_cached_data,
                    "performed_refresh": status.performed_refresh,
                }
                for status in statuses
            ]
        ),
        use_container_width=True,
    )


def render_corporate_actions(db: TradingLabDatabase, symbols: list[str]) -> None:
    actions = db.read_corporate_actions(symbols)
    if actions.empty:
        st.info("No recent dividends or splits stored for the selected tickers yet.")
        return
    st.dataframe(actions, use_container_width=True)


def render_warning_list(warnings: list[str], empty_message: str) -> None:
    if warnings:
        for warning in warnings:
            st.write(f"- {warning}")
    else:
        st.info(empty_message)


def persist_research_outputs(
    db: TradingLabDatabase,
    result: BacktestResult,
    benchmark_symbol: str,
    audit_findings,
    regime_metrics: pd.DataFrame,
    robustness,
    benchmark_diagnostics: BenchmarkDiagnostics,
) -> None:
    audit_frame = pd.DataFrame(
        [
            {"run_id": result.run_id, "severity": finding.severity, "message": finding.message, "created_at": datetime.now(UTC).replace(tzinfo=None)}
            for finding in audit_findings
        ]
    )
    db.replace_audit_results(result.run_id, audit_frame)
    if not regime_metrics.empty:
        stored_regimes = regime_metrics.copy()
        stored_regimes.insert(0, "benchmark_symbol", benchmark_symbol)
        stored_regimes.insert(0, "run_id", result.run_id)
        db.replace_regime_metrics(result.run_id, stored_regimes)
    db.replace_robustness_score(
        {
            "run_id": result.run_id,
            "score": robustness.score,
            "label": robustness.label,
            "strengths_json": json.dumps(robustness.strengths),
            "red_flags_json": json.dumps(robustness.red_flags),
            "explanation_bullets_json": json.dumps(robustness.explanation_bullets),
            "created_at": datetime.now(UTC).replace(tzinfo=None),
        }
    )
    db.replace_benchmark_diagnostics(result.run_id, benchmark_diagnostics.to_record(result.run_id))


def analyze_current_result(
    db: TradingLabDatabase,
    data_by_symbol: dict[str, pd.DataFrame],
    result: BacktestResult,
    config: BacktestConfig,
    strategy_params: dict[str, Any],
    benchmark_symbol: str,
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, object]:
    benchmark_bars = data_by_symbol.get(benchmark_symbol, pd.DataFrame())
    benchmark_price_column = "adj_close" if config.price_mode == "adjusted_price_mode" else "close"
    regime_metrics = compute_regime_metrics(
        result.equity_curve,
        result.trade_log,
        benchmark_bars,
        config.initial_capital,
        benchmark_curve=result.benchmark_curve,
        price_column=benchmark_price_column,
    )
    regime_comments = summarize_regime_comments(regime_metrics)
    concentration = profit_concentration_analysis(result.trade_log)
    benchmark_diagnostics = evaluate_benchmark_diagnostics(result.equity_curve, benchmark_bars, result.benchmark_curve, benchmark_symbol)
    corporate_actions = db.read_corporate_actions_for_period(symbols, start_date, end_date)
    adjusted_available = all("adj_close" in frame.columns and frame["adj_close"].notna().any() for symbol, frame in data_by_symbol.items() if symbol in symbols)
    corporate_action_warnings = summarize_corporate_action_warnings(corporate_actions, price_mode=config.price_mode, adjusted_available=adjusted_available)
    audit_findings = generate_audit_findings(
        result.metrics,
        result.trade_log,
        result.equity_curve,
        benchmark_metrics={"benchmark_cagr": float(result.metrics.get("Benchmark CAGR", 0.0))},
        strategy_parameters=strategy_params,
        regime_comments=regime_comments + benchmark_diagnostics.warnings + corporate_action_warnings,
    )
    robustness = compute_robustness_score(
        result.metrics,
        concentration=concentration,
        regime_metrics=regime_metrics,
    )
    persist_research_outputs(db, result, benchmark_symbol, audit_findings, regime_metrics, robustness, benchmark_diagnostics)
    return {
        "regime_metrics": regime_metrics,
        "regime_comments": regime_comments,
        "concentration": concentration,
        "audit_findings": audit_findings,
        "robustness": robustness,
        "benchmark_diagnostics": benchmark_diagnostics,
        "corporate_action_warnings": corporate_action_warnings,
        "corporate_actions": corporate_actions,
    }


def load_run_train_test_payload(db: TradingLabDatabase, run_id: str) -> dict[str, object] | None:
    frame = db.read_train_test_summary(run_id)
    if frame.empty:
        return None
    row = frame.iloc[0]
    return {
        "split_method": row["split_method"],
        "split_value": row["split_value"],
        "train_metrics": safe_json_loads(row["train_metrics_json"], {}),
        "test_metrics": safe_json_loads(row["test_metrics_json"], {}),
        "degradation": safe_json_loads(row["degradation_json"], {}),
    }


def load_run_walk_forward_payload(db: TradingLabDatabase, run_id: str) -> dict[str, object] | None:
    frame = db.read_walk_forward_runs(run_id)
    if frame.empty:
        return None
    row = frame.iloc[0]
    return safe_json_loads(row["summary_json"], {})


def latest_parameter_stability_for_strategy(db: TradingLabDatabase, strategy_name: str) -> dict[str, object] | None:
    sweeps = db.list_sweep_runs(limit=250)
    if sweeps.empty:
        return None
    strategy_sweeps = sweeps[sweeps["strategy_name"] == strategy_name]
    if strategy_sweeps.empty:
        return None
    latest_sweep_id = strategy_sweeps.iloc[0]["sweep_id"]
    results = db.read_sweep_results(latest_sweep_id)
    if results.empty:
        return None
    return parameter_stability_summary(results.rename(columns={"cagr": "CAGR", "max_drawdown": "Max Drawdown", "total_return": "Total Return"}))


def latest_saved_run_for_strategy(db: TradingLabDatabase, strategy_name: str) -> dict[str, Any] | None:
    internal_name = strategy_internal_name(strategy_name)
    runs = db.list_backtest_runs(limit=500)
    if runs.empty:
        return None
    matches = runs[runs["strategy_name"] == internal_name]
    if matches.empty:
        return None
    return db.get_backtest_run(str(matches.iloc[0]["run_id"]))


def latest_qualification_result_for_strategy(db: TradingLabDatabase, strategy_name: str) -> tuple[str | None, pd.Series | None]:
    qualification_runs = db.list_strategy_qualification_runs(limit=250)
    if qualification_runs.empty:
        return None, None
    for qualification_id in qualification_runs["qualification_id"].tolist():
        results = db.read_strategy_qualification_results(str(qualification_id))
        if results.empty:
            continue
        match = results[results["strategy_name"] == strategy_name]
        if not match.empty:
            return str(qualification_id), match.iloc[0]
    return None, None


def build_options_candidate_assessment(
    db: TradingLabDatabase,
    run_id: str,
    metrics: dict[str, float | int],
    concentration: dict[str, object],
    robustness_score: int,
    strategy_name: str,
) -> object:
    return evaluate_options_overlay_candidate(
        metrics,
        robustness_score=robustness_score,
        concentration=concentration,
        train_test_summary=load_run_train_test_payload(db, run_id),
        walk_forward_summary=load_run_walk_forward_payload(db, run_id),
        parameter_stability=latest_parameter_stability_for_strategy(db, strategy_name),
    )


def render_concentration(concentration: dict[str, object]) -> None:
    cols = st.columns(3)
    cols[0].metric("Best Trade Profit %", f"{float(concentration.get('best_trade_profit_share', 0.0)):.1%}")
    cols[1].metric("Top 3 Profit %", f"{float(concentration.get('top_3_profit_share', 0.0)):.1%}")
    cols[2].metric("Top 5 Profit %", f"{float(concentration.get('top_5_profit_share', 0.0)):.1%}")
    ticker = pd.DataFrame(list((concentration.get("ticker_contribution") or {}).items()), columns=["ticker", "profit_share"])
    year = pd.DataFrame(list((concentration.get("year_contribution") or {}).items()), columns=["year", "profit_share"])
    if not ticker.empty:
        st.write("Ticker Contribution")
        st.dataframe(ticker, use_container_width=True)
    if not year.empty:
        st.write("Year Contribution")
        st.dataframe(year, use_container_width=True)


def render_robustness_frame(robustness_row: pd.Series) -> None:
    st.metric("Robustness Score", f"{int(robustness_row['score'])}/100", robustness_row["label"])
    for section_name, raw in [("Strengths", robustness_row["strengths_json"]), ("Red Flags", robustness_row["red_flags_json"]), ("Explanation", robustness_row["explanation_bullets_json"])]:
        values = safe_json_loads(raw, [])
        if not values:
            continue
        st.write(section_name)
        for item in values:
            st.write(f"- {item}")


def render_robustness(robustness) -> None:
    st.metric("Robustness Score", f"{robustness.score}/100", robustness.label)
    if robustness.strengths:
        st.write("Strengths")
        for item in robustness.strengths:
            st.write(f"- {item}")
    if robustness.red_flags:
        st.write("Red Flags")
        for item in robustness.red_flags:
            st.write(f"- {item}")
    for item in robustness.explanation_bullets:
        st.write(f"- {item}")


def render_options_candidate(assessment) -> None:
    flag_text = "Yes" if assessment.flag else "No"
    st.metric("Options Overlay Candidate", flag_text, assessment.label)
    for bullet in assessment.explanation_bullets:
        st.write(f"- {bullet}")


def render_audit_findings(audit_findings) -> None:
    for finding in audit_findings:
        st.write(f"- [{finding.severity.upper()}] {finding.message}")


def render_benchmark_diagnostics(diagnostics: BenchmarkDiagnostics | pd.DataFrame | None) -> None:
    if diagnostics is None:
        st.info("Benchmark diagnostics are not available.")
        return
    if isinstance(diagnostics, pd.DataFrame):
        if diagnostics.empty:
            st.info("Benchmark diagnostics are not available.")
            return
        row = diagnostics.iloc[0]
        warnings = safe_json_loads(row["warnings_json"], [])
        cols = st.columns(4)
        cols[0].metric("Coverage Ratio", f"{float(row['coverage_ratio']):.1%}")
        cols[1].metric("Missing Sessions", int(row["missing_session_count"]))
        cols[2].metric("Dropped Dates", int(row["dropped_strategy_dates"]))
        cols[3].metric("Zero-Return Days", int(row["zero_return_days"]))
        render_warning_list(warnings, "No benchmark diagnostics warnings were saved.")
        return
    cols = st.columns(4)
    cols[0].metric("Coverage Ratio", f"{diagnostics.coverage_ratio:.1%}")
    cols[1].metric("Missing Sessions", diagnostics.missing_session_count)
    cols[2].metric("Dropped Dates", diagnostics.dropped_strategy_dates)
    cols[3].metric("Zero-Return Days", diagnostics.zero_return_days)
    render_warning_list(diagnostics.warnings, "No benchmark diagnostics warnings were detected.")


def render_saved_run_detail(db: TradingLabDatabase, run_id: str, *, editor_prefix: str) -> None:
    selected_run = db.get_backtest_run(run_id)
    if selected_run is None:
        st.warning("Selected run could not be loaded.")
        return
    trades = db.read_backtest_trades(run_id)
    equity_curve = db.read_backtest_equity_curve(run_id)
    benchmark_curve = db.read_backtest_benchmark_curve(run_id)
    audit = db.read_audit_results(run_id)
    regimes = db.read_regime_metrics(run_id)
    robustness = db.read_robustness_score(run_id)
    train_test = db.read_train_test_summary(run_id)
    walk_runs = db.read_walk_forward_runs(run_id)
    benchmark_diag = db.read_benchmark_diagnostics(run_id)
    concentration = profit_concentration_analysis(trades)

    st.subheader("Saved Run Summary")
    summary_cols = st.columns(2)
    summary_cols[0].json(selected_run, expanded=False)
    notes_value = st.text_area("Run Notes", value=selected_run.get("notes") or "", key=f"{editor_prefix}_notes")
    tags_value = st.text_input("Run Tags", value=selected_run.get("tags") or "", key=f"{editor_prefix}_tags", help="Comma-separated tags for filtering.")
    if st.button("Save Run Notes/Tags", key=f"{editor_prefix}_save_annotations"):
        db.update_backtest_run_annotations(run_id, notes_value, tags_value)
        st.success("Run notes and tags updated.")
    if not robustness.empty:
        st.subheader("Robustness")
        render_robustness_frame(robustness.iloc[0])
        st.subheader("Options Overlay Candidate")
        assessment = build_options_candidate_assessment(
            db,
            run_id,
            {
                "Number of Trades": selected_run.get("number_of_trades", 0),
                "CAGR": selected_run.get("cagr", 0.0),
                "Excess CAGR": selected_run.get("excess_cagr", 0.0),
                "Max Drawdown": selected_run.get("max_drawdown", 0.0),
                "Benchmark Max Drawdown": selected_run.get("benchmark_max_drawdown", 0.0),
                "Beta": selected_run.get("beta", 0.0),
                "Exposure %": selected_run.get("exposure_pct", 0.0),
            },
            concentration,
            int(robustness.iloc[0]["score"]),
            str(selected_run.get("strategy_name") or ""),
        )
        render_options_candidate(assessment)
    st.subheader("Benchmark Diagnostics")
    render_benchmark_diagnostics(benchmark_diag)
    st.subheader("Saved Parameters")
    st.json(safe_json_loads(selected_run.get("parameters_json"), {}), expanded=False)
    if not equity_curve.empty:
        st.plotly_chart(build_equity_chart(equity_curve, benchmark_curve), use_container_width=True)
        st.plotly_chart(build_drawdown_chart(equity_curve), use_container_width=True)
    if not audit.empty:
        st.subheader("Audit Findings")
        st.dataframe(audit, use_container_width=True)
    if not regimes.empty:
        st.subheader("Regime Metrics")
        st.dataframe(regimes, use_container_width=True)
    if not train_test.empty:
        st.subheader("Train/Test Summary")
        st.dataframe(train_test, use_container_width=True)
    if not walk_runs.empty:
        st.subheader("Walk-Forward Runs")
        st.dataframe(walk_runs, use_container_width=True)
        selected_walk = st.selectbox("Saved walk-forward run", walk_runs["walk_forward_id"].tolist(), key=f"{editor_prefix}_saved_walk_select")
        st.dataframe(db.read_walk_forward_folds(selected_walk), use_container_width=True)
    st.subheader("Saved Trade Log")
    st.dataframe(trades, use_container_width=True)
    st.download_button("Export saved trade log", data=trades.to_csv(index=False).encode("utf-8"), file_name=f"{run_id}_trades.csv", mime="text/csv", key=f"{editor_prefix}_trade_export")
    st.download_button(
        "Export saved run summary",
        data=json.dumps(selected_run, indent=2, default=str).encode("utf-8"),
        file_name=f"{run_id}_summary.json",
        mime="application/json",
        key=f"{editor_prefix}_summary_export",
    )


def render_saved_backtests(db: TradingLabDatabase, *, key_prefix: str = "saved_backtests") -> None:
    st.header("Saved Backtests")
    filter_tag = st.text_input("Filter saved runs by tag", value="", key=f"{key_prefix}_saved_runs_tag_filter")
    saved_runs = db.list_backtest_runs(limit=100, tag=filter_tag or None)
    if saved_runs.empty:
        st.info("No saved backtests found yet.")
        return
    st.dataframe(saved_runs, use_container_width=True)
    selected_run_id = st.selectbox("Select saved run", saved_runs["run_id"].tolist(), key=f"{key_prefix}_saved_run_select")
    render_saved_run_detail(db, selected_run_id, editor_prefix=f"{key_prefix}_saved_run")


def render_compare_backtests(db: TradingLabDatabase, *, key_prefix: str = "compare_backtests") -> None:
    st.header("Compare Backtests")
    saved_runs = db.list_backtest_runs(limit=100)
    if saved_runs.empty:
        st.info("No saved backtests available for comparison yet.")
        return
    selected_runs = st.multiselect(
        "Select runs to compare",
        saved_runs["run_id"].tolist(),
        default=saved_runs["run_id"].head(2).tolist(),
        key=f"{key_prefix}_selected_runs",
    )
    if not selected_runs:
        st.info("Select at least one run.")
        return
    comparison = db.compare_backtest_runs(selected_runs)
    robustness_rows: list[dict[str, object]] = []
    benchmark_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for run_id in selected_runs:
        robust = db.read_robustness_score(run_id)
        if not robust.empty:
            score = int(robust.iloc[0]["score"])
            robustness_rows.append({"run_id": run_id, "robustness_score": score, "robustness_label": robust.iloc[0]["label"]})
            selected_run = db.get_backtest_run(run_id) or {}
            assessment = build_options_candidate_assessment(
                db,
                run_id,
                {
                    "Number of Trades": selected_run.get("number_of_trades", 0),
                    "CAGR": selected_run.get("cagr", 0.0),
                    "Excess CAGR": selected_run.get("excess_cagr", 0.0),
                    "Max Drawdown": selected_run.get("max_drawdown", 0.0),
                    "Benchmark Max Drawdown": selected_run.get("benchmark_max_drawdown", 0.0),
                    "Beta": selected_run.get("beta", 0.0),
                    "Exposure %": selected_run.get("exposure_pct", 0.0),
                },
                profit_concentration_analysis(db.read_backtest_trades(run_id)),
                score,
                str(selected_run.get("strategy_name") or ""),
            )
            candidate_rows.append(
                {
                    "run_id": run_id,
                    "options_candidate_flag": assessment.flag,
                    "options_candidate_label": assessment.label,
                }
            )
        diag = db.read_benchmark_diagnostics(run_id)
        if not diag.empty:
            benchmark_rows.append({"run_id": run_id, "benchmark_diag_status": diag.iloc[0]["status"]})
    if robustness_rows:
        comparison = comparison.merge(pd.DataFrame(robustness_rows), on="run_id", how="left")
    if benchmark_rows:
        comparison = comparison.merge(pd.DataFrame(benchmark_rows), on="run_id", how="left")
    if candidate_rows:
        comparison = comparison.merge(pd.DataFrame(candidate_rows), on="run_id", how="left")
    st.dataframe(comparison, use_container_width=True)
    curves = {run_id: db.read_backtest_equity_curve(run_id) for run_id in selected_runs}
    st.plotly_chart(build_multi_equity_chart(curves), use_container_width=True)
    st.plotly_chart(build_multi_drawdown_chart(curves), use_container_width=True)
    selected_run_for_diag = st.selectbox("Show benchmark diagnostics for", selected_runs, key=f"{key_prefix}_compare_diag_run")
    render_benchmark_diagnostics(db.read_benchmark_diagnostics(selected_run_for_diag))
    st.download_button(
        "Export comparison CSV",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="backtest_comparison.csv",
        mime="text/csv",
        key=f"{key_prefix}_comparison_export",
    )


def render_saved_sweeps(db: TradingLabDatabase, *, key_prefix: str = "saved_sweeps") -> None:
    st.subheader("Saved Sweeps")
    tag_filter = st.text_input("Filter sweeps by tag", value="", key=f"{key_prefix}_saved_sweeps_tag_filter")
    saved_sweeps = db.list_sweep_runs(limit=100, tag=tag_filter or None)
    if saved_sweeps.empty:
        st.info("No saved sweeps found yet.")
        return
    st.dataframe(saved_sweeps, use_container_width=True)
    selected_sweep_id = st.selectbox("Select saved sweep", saved_sweeps["sweep_id"].tolist(), key=f"{key_prefix}_saved_sweep_select")
    selected_sweep = db.get_sweep_run(selected_sweep_id)
    if selected_sweep is None:
        st.warning("Selected sweep could not be loaded.")
        return
    results = db.read_sweep_results(selected_sweep_id)
    parameters = db.read_sweep_parameters(selected_sweep_id)
    st.json(selected_sweep, expanded=False)
    notes_value = st.text_area("Sweep Notes", value=selected_sweep.get("notes") or "", key=f"{key_prefix}_saved_sweep_notes")
    tags_value = st.text_input("Sweep Tags", value=selected_sweep.get("tags") or "", key=f"{key_prefix}_saved_sweep_tags")
    if st.button("Save Sweep Notes/Tags", key=f"{key_prefix}_saved_sweep_annotations"):
        db.update_sweep_annotations(selected_sweep_id, notes_value, tags_value)
        st.success("Sweep notes and tags updated.")
    if not parameters.empty:
        st.subheader("Sweep Parameters")
        st.dataframe(parameters, use_container_width=True)
    if results.empty:
        st.info("This sweep has no saved parameter results.")
        return
    st.subheader("Sweep Results")
    st.dataframe(results, use_container_width=True)
    selected_result_id = st.selectbox("Select sweep result", results["sweep_result_id"].tolist(), key=f"{key_prefix}_saved_sweep_result_select")
    selected_result = results.loc[results["sweep_result_id"] == selected_result_id].iloc[0]
    linked_run_id = selected_result.get("backtest_run_id")
    if linked_run_id:
        st.write(f"Linked Backtest Run: `{linked_run_id}`")
        if st.checkbox("Show linked backtest details", key=f"{key_prefix}_show_linked_run"):
            render_saved_run_detail(db, linked_run_id, editor_prefix=f"{key_prefix}_linked_run")
    st.download_button(
        "Export sweep results CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_sweep_id}_sweep_results.csv",
        mime="text/csv",
        key=f"{key_prefix}_sweep_results_export",
    )


def build_strategy_sweep_stability(db: TradingLabDatabase) -> pd.DataFrame:
    saved_sweeps = db.list_sweep_runs(limit=500)
    if saved_sweeps.empty:
        return pd.DataFrame()
    grouped: dict[str, list[pd.DataFrame]] = {}
    for _, row in saved_sweeps.iterrows():
        grouped.setdefault(str(row["strategy_name"]), []).append(db.read_sweep_results(str(row["sweep_id"])))
    return summarize_saved_sweep_stability(grouped)


def run_strategy_qualification(
    *,
    db: TradingLabDatabase,
    provider: YFinanceDataProvider,
    universe_name: str,
    tickers: list[str],
    benchmark_symbol: str,
    start_date: str,
    end_date: str,
    refresh_data: bool,
    config: BacktestConfig,
    strategy_names: list[str],
    notes: str,
    tags: str,
) -> dict[str, object]:
    data_by_symbol, statuses, warnings = collect_data(provider, tickers, start_date, end_date, refresh_data, benchmark_symbol=benchmark_symbol)
    engine = BacktestEngine(database=db)
    rows: list[dict[str, object]] = []
    run_lookup: dict[str, str] = {}
    research_lookup: dict[str, dict[str, object]] = {}
    qualification_id = str(uuid4())
    for strategy_name in strategy_names:
        params = default_strategy_params(strategy_name)
        strategy = build_strategy(strategy_name, params)
        result = engine.run(data_by_symbol=data_by_symbol, strategy=strategy, config=config, benchmark_symbol=benchmark_symbol)
        research = analyze_current_result(db, data_by_symbol, result, config, params, benchmark_symbol, tickers, start_date, end_date)
        robust = research["robustness"]
        assessment = build_options_candidate_assessment(db, result.run_id, result.metrics, research["concentration"], robust.score, strategy_name)
        red_flag_count = len(robust.red_flags) + sum(1 for finding in research["audit_findings"] if finding.severity in {"warning", "critical"})
        rows.append(
            {
                "qualification_result_id": str(uuid4()),
                "qualification_id": qualification_id,
                "strategy_name": strategy_name,
                "backtest_run_id": result.run_id,
                "total_return": float(result.metrics.get("Total Return", 0.0) or 0.0),
                "cagr": float(result.metrics.get("CAGR", 0.0) or 0.0),
                "max_drawdown": float(result.metrics.get("Max Drawdown", 0.0) or 0.0),
                "sharpe": float(result.metrics.get("Sharpe Ratio", 0.0) or 0.0),
                "sortino": float(result.metrics.get("Sortino Ratio", 0.0) or 0.0),
                "calmar": float(result.metrics.get("Calmar Ratio", 0.0) or 0.0),
                "win_rate": float(result.metrics.get("Win Rate", 0.0) or 0.0),
                "profit_factor": float(result.metrics.get("Profit Factor", 0.0) or 0.0),
                "number_of_trades": int(result.metrics.get("Number of Trades", 0) or 0),
                "exposure_pct": float(result.metrics.get("Exposure %", 0.0) or 0.0),
                "excess_cagr": float(result.metrics.get("Excess CAGR", 0.0) or 0.0),
                "robustness_score": robust.score,
                "red_flag_count": red_flag_count,
                "options_candidate_flag": assessment.flag,
                "candidate_label": assessment.label,
                "candidate_explanation_json": json.dumps(assessment.explanation_bullets),
                "created_at": datetime.now(UTC).replace(tzinfo=None),
            }
        )
        run_lookup[strategy_name] = result.run_id
        research_lookup[strategy_name] = research
    results_frame = pd.DataFrame(rows)
    db.replace_strategy_qualification_run(
        {
            "qualification_id": qualification_id,
            "created_at": datetime.now(UTC).replace(tzinfo=None),
            "universe_name": universe_name,
            "tickers": ",".join(tickers),
            "benchmark_symbol": benchmark_symbol,
            "start_date": pd.Timestamp(start_date).date(),
            "end_date": pd.Timestamp(end_date).date(),
            "price_mode": config.price_mode,
            "initial_capital": config.initial_capital,
            "risk_settings_json": json.dumps(
                {
                    "slippage_pct": config.slippage_pct,
                    "commission_per_trade": config.commission_per_trade,
                    "max_positions": config.max_positions,
                    "stop_loss_pct": config.stop_loss_pct,
                    "take_profit_pct": config.take_profit_pct,
                    "trailing_stop_pct": config.trailing_stop_pct,
                }
            ),
            "notes": notes,
            "tags": tags,
        },
        results_frame,
    )
    return {
        "qualification_id": qualification_id,
        "results": results_frame,
        "run_lookup": run_lookup,
        "research_lookup": research_lookup,
        "statuses": statuses,
        "warnings": warnings,
        "data_by_symbol": data_by_symbol,
    }


def render_saved_qualification_runs(db: TradingLabDatabase, *, key_prefix: str = "saved_qualifications") -> None:
    st.subheader("Saved Qualification Runs")
    tag_filter = st.text_input("Filter qualifications by tag", value="", key=f"{key_prefix}_qualification_tag_filter")
    qualifications = db.list_strategy_qualification_runs(limit=100, tag=tag_filter or None)
    if qualifications.empty:
        st.info("No saved strategy qualification runs are available yet.")
        return
    st.dataframe(qualifications, use_container_width=True)
    selected_id = st.selectbox("Select saved qualification", qualifications["qualification_id"].tolist(), key=f"{key_prefix}_qualification_select")
    selected = db.get_strategy_qualification_run(selected_id)
    if selected is None:
        st.warning("Selected qualification could not be loaded.")
        return
    notes_value = st.text_area("Qualification Notes", value=selected.get("notes") or "", key=f"{key_prefix}_qualification_notes")
    tags_value = st.text_input("Qualification Tags", value=selected.get("tags") or "", key=f"{key_prefix}_qualification_tags")
    if st.button("Save Qualification Notes/Tags", key=f"{key_prefix}_qualification_annotations"):
        db.update_strategy_qualification_annotations(selected_id, notes_value, tags_value)
        st.success("Qualification notes and tags updated.")
    st.json(selected, expanded=False)
    results = db.read_strategy_qualification_results(selected_id)
    if results.empty:
        st.info("This qualification has no saved strategy results.")
        return
    st.dataframe(results, use_container_width=True)
    selected_result_id = st.selectbox("Select qualification result", results["qualification_result_id"].tolist(), key=f"{key_prefix}_qualification_result_select")
    selected_result = results.loc[results["qualification_result_id"] == selected_result_id].iloc[0]
    bullets = safe_json_loads(selected_result.get("candidate_explanation_json"), [])
    if bullets:
        st.write("Candidate Assessment")
        for bullet in bullets:
            st.write(f"- {bullet}")
    linked_run_id = selected_result.get("backtest_run_id")
    if linked_run_id and st.checkbox("Show linked backtest details", key=f"{key_prefix}_qualification_show_run"):
        render_saved_run_detail(db, str(linked_run_id), editor_prefix=f"{key_prefix}_qualification_linked_run")
    st.download_button(
        "Export qualification results CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_id}_qualification_results.csv",
        mime="text/csv",
        key=f"{key_prefix}_qualification_export",
    )


def run_signal_scan(
    *,
    db: TradingLabDatabase,
    provider: YFinanceDataProvider,
    universe_name: str,
    tickers: list[str],
    strategy_names: list[str],
    benchmark_symbol: str,
    config: BacktestConfig,
    refresh_data: bool,
) -> dict[str, object]:
    scan_end = pd.Timestamp.today().normalize()
    scan_start = (scan_end - pd.Timedelta(days=500)).date().isoformat()
    data_by_symbol, statuses, validation_warnings = collect_data(
        provider,
        tickers,
        scan_start,
        scan_end.date().isoformat(),
        refresh_data,
        benchmark_symbol=benchmark_symbol,
    )
    benchmark_bars = data_by_symbol.get(benchmark_symbol, pd.DataFrame())
    bear_regime = False
    if not benchmark_bars.empty and len(benchmark_bars) >= 200:
        benchmark_close = benchmark_bars["adj_close"] if config.price_mode == "adjusted_price_mode" else benchmark_bars["close"]
        bear_regime = bool(float(benchmark_close.iloc[-1]) < float(benchmark_close.rolling(200).mean().iloc[-1]))

    rows: list[dict[str, object]] = []
    result_objects: list[object] = []
    for strategy_name in strategy_names:
        run_context = latest_saved_run_for_strategy(db, strategy_name)
        qualification_id, qualification_row = latest_qualification_result_for_strategy(db, strategy_name)
        stability = latest_parameter_stability_for_strategy(db, strategy_name)
        stability_poor = bool(stability and "narrow" in str(stability.get("conclusion", "")).lower())
        robustness_score = None
        if run_context is not None:
            robust_frame = db.read_robustness_score(str(run_context["run_id"]))
            if not robust_frame.empty:
                robustness_score = int(robust_frame.iloc[0]["score"])
        for ticker in tickers:
            bars = data_by_symbol.get(ticker, pd.DataFrame())
            symbol_validation_warnings = [warning for warning in validation_warnings if warning.startswith(f"{ticker}:")]
            corporate_actions = db.read_corporate_actions_for_period(
                [ticker],
                str((scan_end - pd.Timedelta(days=365)).date()),
                str(scan_end.date()),
            )
            corporate_warnings = summarize_corporate_action_warnings(
                corporate_actions,
                price_mode=config.price_mode,
                adjusted_available=("adj_close" in bars.columns and bars["adj_close"].notna().any()) if not bars.empty else False,
            )
            result = scan_symbol_strategy(
                ticker=ticker,
                bars=bars,
                strategy_name=strategy_name,
                strategy=build_strategy(strategy_name, default_strategy_params(strategy_name)),
                config=config,
                latest_run_id=str(run_context["run_id"]) if run_context else None,
                qualification_id=qualification_id,
                robustness_score=robustness_score,
                qualification_status=str(qualification_row["candidate_label"]) if qualification_row is not None else None,
                quality_inputs={
                    "data_quality_warnings": symbol_validation_warnings,
                    "corporate_action_warnings": corporate_warnings,
                    "trade_count": int(run_context["number_of_trades"]) if run_context else 0,
                    "parameter_stability_poor": stability_poor,
                    "bear_regime": bear_regime,
                },
            )
            result.notes_warnings.extend(symbol_validation_warnings)
            result.notes_warnings.extend(corporate_warnings)
            if result.robustness_score is not None and result.robustness_score < 40:
                result.notes_warnings.append("Saved robustness score is weak.")
            if qualification_row is not None and str(qualification_row["candidate_label"]).lower() == "not ready":
                result.notes_warnings.append("Latest qualification marked this strategy as not ready.")
            rows.append(result.to_record())
            result_objects.append(result)
    results = pd.DataFrame(rows)
    if not results.empty:
        sort_priority = {"new_buy_signal": 0, "active_long_signal": 1, "exit_signal": 2, "no_signal": 3}
        results["sort_priority"] = results["signal_type"].map(sort_priority).fillna(9)
        results = results.sort_values(["sort_priority", "signal_quality_score", "ticker", "strategy"], ascending=[True, False, True, True]).drop(columns=["sort_priority"])
    return {
        "universe_name": universe_name,
        "tickers": tickers,
        "results": results,
        "result_objects": result_objects,
        "statuses": statuses,
        "validation_warnings": validation_warnings,
    }


def summarize_scanner_counts(results: pd.DataFrame) -> dict[str, int]:
    if results.empty:
        return {"new_buy_signals": 0, "active_long_signals": 0, "exit_signals": 0, "no_signal": 0}
    return {
        "new_buy_signals": int((results["signal_type"] == "new_buy_signal").sum()),
        "active_long_signals": int((results["signal_type"] == "active_long_signal").sum()),
        "exit_signals": int((results["signal_type"] == "exit_signal").sum()),
        "no_signal": int((results["signal_type"] == "no_signal").sum()),
    }


def paper_trade_summary(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {
            "open_trades": 0,
            "planned_trades": 0,
            "closed_trades": 0,
            "realized_pnl": 0.0,
            "win_rate": 0.0,
            "average_return": 0.0,
        }
    closed = trades[trades["status"] == "closed"]
    return {
        "open_trades": int((trades["status"] == "open").sum()),
        "planned_trades": int((trades["status"] == "planned").sum()),
        "closed_trades": int((trades["status"] == "closed").sum()),
        "realized_pnl": float(closed["realized_pnl"].sum()) if not closed.empty else 0.0,
        "win_rate": float((closed["realized_pnl"] > 0).mean()) if not closed.empty else 0.0,
        "average_return": float(closed["realized_return_pct"].mean()) if not closed.empty else 0.0,
    }


def build_scanner_snapshot_result_frame(results: pd.DataFrame, snapshot_id: str) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(
            columns=[
                "scanner_result_id",
                "snapshot_id",
                "ticker",
                "strategy_name",
                "signal_type",
                "signal_date",
                "latest_close",
                "suggested_entry",
                "suggested_stop",
                "suggested_target",
                "risk_per_share",
                "reward_per_share",
                "reward_risk_ratio",
                "robustness_score",
                "qualification_status",
                "signal_quality_score",
                "signal_quality_label",
                "explanation",
                "warnings_json",
                "linked_paper_trade_id",
            ]
        )
    frame = results.copy().reset_index(drop=True)
    return pd.DataFrame(
        {
            "scanner_result_id": [str(uuid4()) for _ in range(len(frame))],
            "snapshot_id": snapshot_id,
            "ticker": frame["ticker"],
            "strategy_name": frame["strategy"],
            "signal_type": frame["signal_type"],
            "signal_date": pd.to_datetime(frame["signal_date"], errors="coerce"),
            "latest_close": frame["latest_close"],
            "suggested_entry": frame["suggested_entry_reference"],
            "suggested_stop": frame["suggested_stop"],
            "suggested_target": frame["suggested_target"],
            "risk_per_share": frame["risk_per_share"],
            "reward_per_share": frame["reward_per_share"],
            "reward_risk_ratio": frame["reward_risk_ratio"],
            "robustness_score": frame["robustness_score"],
            "qualification_status": frame["qualification_status"],
            "signal_quality_score": frame["signal_quality_score"],
            "signal_quality_label": frame["signal_quality_label"],
            "explanation": frame["explanation"],
            "warnings_json": frame["notes_warnings"].fillna("").apply(lambda value: json.dumps([item for item in str(value).split(" | ") if item])),
            "linked_paper_trade_id": None,
        }
    )


def save_scanner_snapshot(
    db: TradingLabDatabase,
    scan_state: dict[str, object],
    *,
    benchmark_symbol: str,
    price_mode: str,
    notes: str,
    tags: str,
) -> str:
    snapshot_id = str(uuid4())
    results_frame = build_scanner_snapshot_result_frame(scan_state["results"], snapshot_id)
    db.replace_scanner_snapshot(
        {
            "snapshot_id": snapshot_id,
            "created_at": datetime.now(UTC).replace(tzinfo=None),
            "universe_name": scan_state["universe_name"],
            "tickers": ",".join(scan_state["tickers"]),
            "strategies": ",".join(sorted(scan_state["results"]["strategy"].unique().tolist())) if not scan_state["results"].empty else "",
            "benchmark_symbol": benchmark_symbol,
            "price_mode": price_mode,
            "scanner_config_json": json.dumps({"validation_warning_count": len(scan_state.get("validation_warnings", []))}),
            "notes": notes,
            "tags": tags,
        },
        results_frame,
    )
    return snapshot_id


def render_signal_scanner(
    *,
    db: TradingLabDatabase,
    provider: YFinanceDataProvider,
    benchmark_symbol: str,
    refresh_data: bool,
    base_config: BacktestConfig,
    default_tickers: list[str],
) -> None:
    st.header("Signal Scanner")
    saved_tag_filter = st.text_input("Filter scanner snapshots by tag", value="", key="scanner_snapshot_tag_filter")
    saved_snapshots = db.list_scanner_snapshots(limit=100, tag=saved_tag_filter or None)
    if saved_snapshots.empty:
        st.info("No saved scanner snapshots yet.")
    else:
        st.subheader("Saved Scanner Snapshots")
        st.dataframe(saved_snapshots, use_container_width=True)
        selected_snapshot_id = st.selectbox("Select saved scanner snapshot", saved_snapshots["snapshot_id"].tolist(), key="saved_scanner_snapshot_select")
        action_filter = st.selectbox("Saved snapshot action filter", ["All", "no_action", "planned", "open", "closed", "canceled"], key="saved_scanner_action_filter")
        saved_results = db.read_scanner_snapshot_results(selected_snapshot_id, None if action_filter == "All" else action_filter)
        st.dataframe(saved_results, use_container_width=True)
        st.download_button(
            "Export saved scanner snapshot CSV",
            data=saved_results.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_snapshot_id}_scanner_snapshot.csv",
            mime="text/csv",
        )
    st.divider()
    universe_name = st.selectbox("Scanner universe", list_universe_names(), index=1, key="scanner_universe")
    default_value = ",".join(default_tickers if universe_name == "Custom" else get_universe_tickers(universe_name))
    editable_tickers = st.text_input("Scanner tickers", value=default_value, key="scanner_tickers")
    selected_tickers = normalize_ticker_list(editable_tickers)
    strategy_names = st.multiselect(
        "Strategies to scan",
        ["Moving Average Crossover", "RSI Mean Reversion", "Daily Breakout", "QQE/HMA Daily"],
        default=["Moving Average Crossover", "RSI Mean Reversion", "Daily Breakout", "QQE/HMA Daily"],
        key="scanner_strategies",
    )
    if st.button("Run Signal Scanner"):
        if not selected_tickers:
            st.error("Select at least one ticker to scan.")
        elif not strategy_names:
            st.error("Select at least one strategy to scan.")
        else:
            with st.spinner("Scanning latest daily signals..."):
                st.session_state[SESSION_SCANNER_KEY] = run_signal_scan(
                    db=db,
                    provider=provider,
                    universe_name=universe_name,
                    tickers=selected_tickers,
                    strategy_names=strategy_names,
                    benchmark_symbol=benchmark_symbol,
                    config=base_config,
                    refresh_data=refresh_data,
                )
    scan_state = st.session_state.get(SESSION_SCANNER_KEY)
    if not scan_state:
        st.info("Run the scanner to see current daily signals.")
        return

    results: pd.DataFrame = scan_state["results"]
    if results.empty:
        st.info("No scanner rows were produced for the selected universe and strategies.")
        return
    counts = summarize_scanner_counts(results)
    metric_cols = st.columns(4)
    metric_cols[0].metric("New Buy Signals", counts["new_buy_signals"])
    metric_cols[1].metric("Active Long Signals", counts["active_long_signals"])
    metric_cols[2].metric("Exit Signals", counts["exit_signals"])
    metric_cols[3].metric("No Signal", counts["no_signal"])
    snapshot_notes = st.text_area("Scanner snapshot notes", value="", key="scanner_snapshot_notes")
    snapshot_tags = st.text_input("Scanner snapshot tags", value="", key="scanner_snapshot_tags")
    if st.button("Save Scanner Snapshot", key="scanner_save_snapshot"):
        snapshot_id = save_scanner_snapshot(db, scan_state, benchmark_symbol=benchmark_symbol, price_mode=base_config.price_mode, notes=snapshot_notes, tags=snapshot_tags)
        st.session_state[SESSION_SCANNER_SNAPSHOT_KEY] = snapshot_id
        st.success(f"Scanner snapshot saved: {snapshot_id}")
    st.dataframe(results, use_container_width=True)
    st.download_button("Export scanner CSV", data=results.to_csv(index=False).encode("utf-8"), file_name="signal_scanner.csv", mime="text/csv")

    actionable = results[results["signal_type"] != "no_signal"]
    detail_source = actionable if not actionable.empty else results
    selection_labels = [f"{row.ticker} | {row.strategy} | {row.signal_type}" for row in detail_source.itertuples()]
    selected_label = st.selectbox("Signal details", selection_labels, key="scanner_signal_detail")
    selected_index = selection_labels.index(selected_label)
    selected_row = detail_source.iloc[selected_index]
    selected_signal = next(
        (
            signal
            for signal in scan_state["result_objects"]
            if signal.ticker == selected_row["ticker"] and signal.strategy == selected_row["strategy"]
        ),
        None,
    )
    st.subheader("Signal Explanation")
    st.write(selected_row["explanation"])
    bullets = [item for item in str(selected_row.get("signal_quality_bullets", "")).split(" | ") if item]
    if bullets:
        st.write("Signal Quality")
        for bullet in bullets:
            st.write(f"- {bullet}")
    if selected_row.get("notes_warnings"):
        st.write("Notes and Warnings")
        for item in str(selected_row["notes_warnings"]).split(" | "):
            if item:
                st.write(f"- {item}")

    st.subheader("Manual Trade Plan")
    portfolio_value = st.number_input("Portfolio value", min_value=1000.0, value=100000.0, key="scanner_plan_portfolio")
    sizing_method = st.selectbox(
        "Plan sizing method",
        ["fixed_dollar_allocation", "percent_of_portfolio", "fixed_dollar_risk"],
        key="scanner_plan_sizing",
    )
    if sizing_method == "fixed_dollar_allocation":
        sizing_value = st.number_input("Dollar allocation", min_value=0.0, value=5000.0, key="scanner_plan_alloc")
    elif sizing_method == "percent_of_portfolio":
        sizing_value = st.number_input("Portfolio fraction", min_value=0.0, value=0.1, step=0.01, key="scanner_plan_pct")
    else:
        sizing_value = st.number_input("Dollar risk per trade", min_value=0.0, value=500.0, key="scanner_plan_risk")
    plan_notes = st.text_area("Trade plan notes", value="", key="scanner_plan_notes")
    plan_tags = st.text_input("Trade plan tags", value="", key="scanner_plan_tags")
    if st.button("Generate Trade Plan", key="scanner_generate_plan"):
        if selected_signal is None:
            st.error("Selected signal could not be mapped back to the scanner state.")
            return
        plan = plan_trade_from_signal(
            selected_signal,
            portfolio_value=portfolio_value,
            sizing_method=sizing_method,
            sizing_value=sizing_value,
            notes=plan_notes,
            tags=plan_tags,
        )
        snapshot_id = st.session_state.get(SESSION_SCANNER_SNAPSHOT_KEY)
        if snapshot_id:
            saved_results = db.read_scanner_snapshot_results(snapshot_id)
            if not saved_results.empty:
                linked_row = saved_results[
                    (saved_results["ticker"] == selected_signal.ticker)
                    & (saved_results["strategy_name"] == selected_signal.strategy)
                    & (saved_results["signal_type"] == selected_signal.signal_type)
                ]
                if not linked_row.empty:
                    plan["scanner_snapshot_id"] = snapshot_id
                    plan["scanner_result_id"] = linked_row.iloc[0]["scanner_result_id"]
        plan["qualification_status"] = selected_signal.qualification_status
        plan["signal_explanation"] = selected_signal.explanation
        plan["signal_warnings_json"] = json.dumps(selected_signal.notes_warnings)
        plan["universe_name"] = scan_state["universe_name"]
        st.session_state[SESSION_TRADE_PLAN_KEY] = plan
    plan = st.session_state.get(SESSION_TRADE_PLAN_KEY)
    if plan:
        st.json(plan, expanded=False)
        if st.button("Create Paper Trade From Plan", key="scanner_create_paper_trade"):
            payload = create_paper_trade_payload(plan)
            db.insert_paper_trade(payload)
            if payload.get("scanner_result_id"):
                db.update_scanner_snapshot_result_link(str(payload["scanner_result_id"]), str(payload["paper_trade_id"]))
            db.insert_paper_trade_event(
                {
                    "event_id": str(uuid4()),
                    "paper_trade_id": payload["paper_trade_id"],
                    "created_at": datetime.now(UTC).replace(tzinfo=None),
                    "event_type": "planned",
                    "event_note": "Paper trade created from scanner plan.",
                    "price": float(payload["planned_entry"]),
                    "quantity": int(payload["shares"]),
                }
            )
            st.success("Paper trade created.")


def render_paper_trade_journal(db: TradingLabDatabase) -> None:
    st.header("Paper Journal")
    st.caption("Paper trades are manually tracked in this app. They are not broker-synced and do not place trades.")
    status_filter = st.selectbox("Paper trade status filter", ["All", "planned", "open", "closed", "canceled"], key="paper_trade_status_filter")
    trades = db.list_paper_trades_with_context()
    if status_filter != "All":
        trades = trades[trades["status"] == status_filter]
    if trades.empty:
        st.info("No paper trades have been created yet.")
        return
    st.dataframe(trades, use_container_width=True)
    selected_trade_id = st.selectbox("Select paper trade", trades["paper_trade_id"].tolist(), key="paper_trade_select")
    selected_trade = db.get_paper_trade(selected_trade_id)
    if selected_trade is None:
        st.warning("Selected paper trade could not be loaded.")
        return
    st.json(selected_trade, expanded=False)
    events = db.read_paper_trade_events(selected_trade_id)
    if not events.empty:
        st.write("Trade Events")
        st.dataframe(events, use_container_width=True)
    status = selected_trade.get("status")
    if status == "planned":
        actual_entry = st.number_input("Actual entry", min_value=0.0, value=float(selected_trade.get("planned_entry") or 0.0), key="paper_open_entry")
        entry_date = st.date_input("Entry date", value=pd.Timestamp.today().normalize(), key="paper_open_date")
        if st.button("Mark Trade Open", key="paper_mark_open"):
            updated = open_paper_trade_payload(selected_trade, actual_entry=actual_entry, entry_date=pd.Timestamp(entry_date))
            db.update_paper_trade(updated)
            db.insert_paper_trade_event(
                {
                    "event_id": str(uuid4()),
                    "paper_trade_id": selected_trade_id,
                    "created_at": datetime.now(UTC).replace(tzinfo=None),
                    "event_type": "opened",
                    "event_note": "Trade marked open manually.",
                    "price": float(actual_entry),
                    "quantity": int(updated.get("shares") or 0),
                }
            )
            st.success("Paper trade marked open.")
        if st.button("Cancel Planned Trade", key="paper_cancel_trade"):
            updated = dict(selected_trade)
            updated["updated_at"] = datetime.now(UTC).replace(tzinfo=None)
            updated["status"] = "canceled"
            db.update_paper_trade(updated)
            db.insert_paper_trade_event(
                {
                    "event_id": str(uuid4()),
                    "paper_trade_id": selected_trade_id,
                    "created_at": datetime.now(UTC).replace(tzinfo=None),
                    "event_type": "canceled",
                    "event_note": "Planned trade canceled manually.",
                    "price": None,
                    "quantity": int(updated.get("shares") or 0),
                }
            )
            st.success("Paper trade canceled.")
    elif status == "open":
        exit_price = st.number_input("Exit price", min_value=0.0, value=float(selected_trade.get("take_profit") or selected_trade.get("actual_entry") or 0.0), key="paper_close_price")
        exit_date = st.date_input("Exit date", value=pd.Timestamp.today().normalize(), key="paper_close_date")
        exit_reason = st.text_input("Exit reason", value="manual_close", key="paper_close_reason")
        if st.button("Close Trade", key="paper_close_trade"):
            updated = close_paper_trade_payload(selected_trade, exit_price=exit_price, exit_date=pd.Timestamp(exit_date), exit_reason=exit_reason)
            db.update_paper_trade(updated)
            db.insert_paper_trade_event(
                {
                    "event_id": str(uuid4()),
                    "paper_trade_id": selected_trade_id,
                    "created_at": datetime.now(UTC).replace(tzinfo=None),
                    "event_type": "closed",
                    "event_note": f"Trade closed manually: {exit_reason}",
                    "price": float(exit_price),
                    "quantity": int(updated.get("shares") or 0),
                }
            )
            st.success("Paper trade closed.")
    elif status == "closed":
        st.subheader("Post-Trade Review")
        review_values = {
            "thesis_review": st.text_area("Thesis review", value=str(selected_trade.get("thesis_review") or ""), key="paper_review_thesis"),
            "execution_review": st.text_area("Execution review", value=str(selected_trade.get("execution_review") or ""), key="paper_review_execution"),
            "what_went_well": st.text_area("What went well", value=str(selected_trade.get("what_went_well") or ""), key="paper_review_well"),
            "what_went_wrong": st.text_area("What went wrong", value=str(selected_trade.get("what_went_wrong") or ""), key="paper_review_wrong"),
            "lesson_learned": st.text_area("Lesson learned", value=str(selected_trade.get("lesson_learned") or ""), key="paper_review_lesson"),
            "mistake_tags": st.text_input(
                "Mistake tags",
                value=str(selected_trade.get("mistake_tags") or ""),
                key="paper_review_tags",
                help="Examples: chased entry, ignored stop, exited early, weak setup, ignored warning.",
            ),
            "followed_plan_flag": st.checkbox("Followed plan", value=bool(selected_trade.get("followed_plan_flag") or False), key="paper_review_followed"),
            "entry_quality_rating": st.slider("Entry quality rating", min_value=1, max_value=5, value=int(selected_trade.get("entry_quality_rating") or 3), key="paper_review_entry"),
            "exit_quality_rating": st.slider("Exit quality rating", min_value=1, max_value=5, value=int(selected_trade.get("exit_quality_rating") or 3), key="paper_review_exit"),
            "emotional_discipline_rating": st.slider("Emotional discipline rating", min_value=1, max_value=5, value=int(selected_trade.get("emotional_discipline_rating") or 3), key="paper_review_emotion"),
        }
        if st.button("Save Post-Trade Review", key="paper_save_review"):
            updated = update_post_trade_review(selected_trade, review_values)
            db.update_paper_trade(updated)
            db.insert_paper_trade_event(
                {
                    "event_id": str(uuid4()),
                    "paper_trade_id": selected_trade_id,
                    "created_at": datetime.now(UTC).replace(tzinfo=None),
                    "event_type": "review",
                    "event_note": "Post-trade review updated.",
                    "price": selected_trade.get("exit_price"),
                    "quantity": int(selected_trade.get("shares") or 0),
                }
            )
            st.success("Post-trade review saved.")
    event_note = st.text_input("New event note", value="", key="paper_event_note")
    event_price = st.number_input("Event price", min_value=0.0, value=0.0, key="paper_event_price")
    event_qty = st.number_input("Event quantity", min_value=0, value=int(selected_trade.get("shares") or 0), key="paper_event_qty")
    if st.button("Add Paper Trade Event", key="paper_add_event") and event_note.strip():
        db.insert_paper_trade_event(
            {
                "event_id": str(uuid4()),
                "paper_trade_id": selected_trade_id,
                "created_at": datetime.now(UTC).replace(tzinfo=None),
                "event_type": "note",
                "event_note": event_note.strip(),
                "price": float(event_price) if event_price else None,
                "quantity": int(event_qty),
            }
        )
        st.success("Paper trade event added.")
    st.download_button("Export paper trades CSV", data=trades.to_csv(index=False).encode("utf-8"), file_name="paper_trades.csv", mime="text/csv")

    analytics = closed_trade_analytics(db.list_paper_trades_with_context())
    st.subheader("Paper Trade Analytics")
    summary = analytics["summary"]
    summary_cols = st.columns(6)
    summary_cols[0].metric("Total Realized P&L", f"{float(summary['total_realized_pnl']):.2f}")
    summary_cols[1].metric("Win Rate", f"{float(summary['win_rate']):.1%}")
    summary_cols[2].metric("Profit Factor", f"{float(summary['profit_factor']):.2f}")
    summary_cols[3].metric("Expectancy", f"{float(summary['expectancy_per_trade']):.2f}")
    summary_cols[4].metric("Avg Holding Period", f"{float(summary['average_holding_period']):.1f}d")
    summary_cols[5].metric("Avg Planned RR", f"{float(summary['average_reward_risk_planned']):.2f}")
    if not analytics["planned_vs_actual"].empty:
        st.write("Planned vs Actual")
        st.dataframe(analytics["planned_vs_actual"], use_container_width=True)
    for label, frame in [
        ("By Strategy", analytics["by_strategy"]),
        ("By Ticker", analytics["by_ticker"]),
        ("By Universe", analytics["by_universe"]),
        ("By Signal Quality", analytics["by_signal_quality"]),
        ("By Qualification Status", analytics["by_qualification_status"]),
        ("By Robustness Bucket", analytics["by_robustness_bucket"]),
        ("Common Mistake Tags", analytics["mistake_tags"]),
    ]:
        if not frame.empty:
            st.write(label)
            st.dataframe(frame, use_container_width=True)


def render_watchlist(db: TradingLabDatabase, scanner_results: pd.DataFrame | None) -> None:
    st.subheader("Watchlist")
    watch_ticker = st.text_input("Watchlist ticker", value="", key="watchlist_ticker").upper().strip()
    watch_category = st.selectbox("Watchlist category", ["general watch", "high priority", "waiting for pullback", "waiting for breakout", "avoid"], key="watchlist_category")
    watch_notes = st.text_area("Watchlist notes", value="", key="watchlist_notes")
    watch_tags = st.text_input("Watchlist tags", value="", key="watchlist_tags")
    if st.button("Add To Watchlist", key="watchlist_add") and watch_ticker:
        now = datetime.now(UTC).replace(tzinfo=None)
        db.upsert_watchlist_item(
            {
                "watchlist_id": watch_ticker,
                "ticker": watch_ticker,
                "created_at": now,
                "updated_at": now,
                "category": watch_category,
                "notes": watch_notes,
                "tags": watch_tags,
            }
        )
        st.success("Watchlist item saved.")
    tag_filter = st.text_input("Watchlist tag filter", value="", key="watchlist_filter")
    watchlist = db.list_watchlist(tag_filter or None)
    if watchlist.empty:
        st.info("No watchlist items saved yet.")
        return
    if scanner_results is not None and not scanner_results.empty:
        signal_view = scanner_results.sort_values("signal_quality_score", ascending=False).drop_duplicates(subset=["ticker"])[
            ["ticker", "strategy", "signal_type", "latest_close", "signal_quality_label", "signal_quality_score"]
        ]
        watchlist = watchlist.merge(signal_view, on="ticker", how="left")
    st.dataframe(watchlist, use_container_width=True)
    st.download_button("Export watchlist CSV", data=watchlist.to_csv(index=False).encode("utf-8"), file_name="watchlist.csv", mime="text/csv")


def render_daily_trading_dashboard(db: TradingLabDatabase) -> None:
    st.header("Daily Trading Dashboard")
    scanner_state = st.session_state.get(SESSION_SCANNER_KEY)
    scanner_results = scanner_state["results"] if scanner_state else pd.DataFrame()
    counts = summarize_scanner_counts(scanner_results) if scanner_state else {"new_buy_signals": 0, "active_long_signals": 0, "exit_signals": 0, "no_signal": 0}
    paper_trades = db.list_paper_trades()
    paper_summary = paper_trade_summary(paper_trades)
    cards = st.columns(8)
    cards[0].metric("New Buy Signals", counts["new_buy_signals"])
    cards[1].metric("Active Long Signals", counts["active_long_signals"])
    cards[2].metric("Exit Signals", counts["exit_signals"])
    cards[3].metric("Open Paper Trades", paper_summary["open_trades"])
    cards[4].metric("Planned Trades", paper_summary["planned_trades"])
    cards[5].metric("Closed Trades", paper_summary["closed_trades"])
    cards[6].metric("Realized Paper P&L", f"{paper_summary['realized_pnl']:.2f}")
    cards[7].metric("Closed Win Rate", f"{paper_summary['win_rate']:.1%}")
    st.metric("Average Closed Paper Trade Return", f"{paper_summary['average_return']:.1%}")

    if scanner_state and not scanner_results.empty:
        st.subheader("Highest Quality Signals")
        high_quality = scanner_results.sort_values(["signal_quality_score", "reward_risk_ratio"], ascending=[False, False]).head(10)
        st.dataframe(high_quality, use_container_width=True)
        qualified = scanner_results[scanner_results["qualification_status"].fillna("").isin(["Strong candidate", "Possible candidate"])]
        if not qualified.empty:
            st.subheader("Signals From Qualified Strategies")
            st.dataframe(qualified, use_container_width=True)
        weak = scanner_results[(scanner_results["signal_quality_label"].isin(["Low quality", "Ignore"])) | (scanner_results["robustness_score"].fillna(0) < 40)]
        if not weak.empty:
            st.subheader("Lower-Quality Signals To Review Carefully")
            st.dataframe(weak, use_container_width=True)

    if not paper_trades.empty and scanner_state and not scanner_results.empty:
        open_trades = paper_trades[paper_trades["status"] == "open"]
        if not open_trades.empty:
            latest_prices = scanner_results.sort_values("signal_quality_score", ascending=False).drop_duplicates(subset=["ticker"])[["ticker", "latest_close"]]
            merged = open_trades.merge(latest_prices, left_on="ticker", right_on="ticker", how="left")
            approaching = merged[
                ((merged["latest_close"] - merged["stop_loss"]).abs() / merged["stop_loss"].clip(lower=0.01) < 0.03)
                | ((merged["take_profit"] - merged["latest_close"]).abs() / merged["take_profit"].clip(lower=0.01) < 0.03)
            ]
            if not approaching.empty:
                st.subheader("Open Trades Near Stop Or Target")
                st.dataframe(approaching, use_container_width=True)

    render_watchlist(db, scanner_results if scanner_state else None)
    analytics = closed_trade_analytics(db.list_paper_trades_with_context())
    if not analytics["by_strategy"].empty:
        st.subheader("Strategy-Level Paper Results")
        st.dataframe(analytics["by_strategy"], use_container_width=True)


def render_scanner_history(db: TradingLabDatabase) -> None:
    st.header("Scanner History")
    snapshots = db.list_scanner_snapshots(limit=250)
    if snapshots.empty:
        st.info("No saved scanner snapshots are available yet.")
        return
    summary = db.scanner_history_summary()
    ticker_counts = db.scanner_history_by_ticker()
    strategy_quality = db.scanner_history_by_strategy_quality()
    if not summary.empty:
        st.plotly_chart(px.line(summary, x="snapshot_date", y=["snapshot_count", "new_buy_count", "exit_count"], title="Scanner Snapshot Activity Over Time"), use_container_width=True)
        st.dataframe(summary, use_container_width=True)
    if not ticker_counts.empty:
        st.subheader("Recurring Tickers By Signal Count")
        st.dataframe(ticker_counts.head(25), use_container_width=True)
    if not strategy_quality.empty:
        st.subheader("Signal Quality Counts By Strategy")
        st.dataframe(strategy_quality, use_container_width=True)


def render_strategy_qualification(
    *,
    db: TradingLabDatabase,
    provider: YFinanceDataProvider,
    start_date: str,
    end_date: str,
    benchmark_symbol: str,
    refresh_data: bool,
    base_config: BacktestConfig,
    default_tickers: list[str],
) -> None:
    st.header("Strategy Qualification")
    render_saved_qualification_runs(db, key_prefix="qualification_saved_runs")
    st.divider()
    universe_name = st.selectbox("Universe", list_universe_names(), index=1, key="qualification_universe")
    universe_tickers = get_universe_tickers(universe_name)
    if universe_name == "Custom":
        default_value = ",".join(default_tickers)
    else:
        default_value = ",".join(universe_tickers)
    editable_tickers = st.text_input(
        "Qualification tickers",
        value=default_value,
        key="qualification_tickers",
        help="You can start from a predefined universe and then edit the comma-separated list before running the comparison.",
    )
    selected_tickers = normalize_ticker_list(editable_tickers)
    strategy_names = st.multiselect(
        "Strategies to compare",
        ["Moving Average Crossover", "RSI Mean Reversion", "Daily Breakout", "QQE/HMA Daily"],
        default=["Moving Average Crossover", "RSI Mean Reversion", "Daily Breakout", "QQE/HMA Daily"],
        key="qualification_strategies",
    )
    qualification_notes = st.text_area("Qualification notes", value="", key="qualification_notes_input")
    qualification_tags = st.text_input("Qualification tags", value="", key="qualification_tags_input", help="Comma-separated tags such as options-candidate or large-cap-tech.")
    run_qualification = st.button("Run Strategy Qualification")
    if run_qualification:
        if not selected_tickers:
            st.error("Select at least one ticker for strategy qualification.")
        elif not strategy_names:
            st.error("Select at least one strategy to compare.")
        else:
            with st.spinner("Running strategy qualification..."):
                qualification = run_strategy_qualification(
                    db=db,
                    provider=provider,
                    universe_name=universe_name,
                    tickers=selected_tickers,
                    benchmark_symbol=benchmark_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    refresh_data=refresh_data,
                    config=base_config,
                    strategy_names=strategy_names,
                    notes=qualification_notes,
                    tags=qualification_tags,
                )
            st.session_state[SESSION_QUALIFICATION_KEY] = qualification
    qualification = st.session_state.get(SESSION_QUALIFICATION_KEY)
    if not qualification:
        return

    results = qualification["results"].copy()
    if results.empty:
        st.warning("The selected strategies did not produce qualification rows.")
        return
    st.write(f"Qualification ID: `{qualification['qualification_id']}`")
    display = results.rename(
        columns={
            "total_return": "Total Return",
            "cagr": "CAGR",
            "max_drawdown": "Max Drawdown",
            "sharpe": "Sharpe",
            "sortino": "Sortino",
            "calmar": "Calmar",
            "win_rate": "Win Rate",
            "profit_factor": "Profit Factor",
            "number_of_trades": "Number of Trades",
            "exposure_pct": "Exposure %",
            "excess_cagr": "Excess CAGR",
            "robustness_score": "Robustness Score",
            "red_flag_count": "Red Flag Count",
            "options_candidate_flag": "Options Candidate Flag",
            "candidate_label": "Candidate Label",
        }
    )
    st.dataframe(display, use_container_width=True)
    st.download_button(
        "Export strategy qualification CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name=f"{qualification['qualification_id']}_strategy_qualification.csv",
        mime="text/csv",
    )

    selected_strategy = st.selectbox("Qualification strategy details", results["strategy_name"].tolist(), key="qualification_strategy_detail")
    selected_row = results.loc[results["strategy_name"] == selected_strategy].iloc[0]
    for bullet in safe_json_loads(selected_row["candidate_explanation_json"], []):
        st.write(f"- {bullet}")
    if selected_strategy == "QQE/HMA Daily":
        simple_rows = results[results["strategy_name"].isin(["Moving Average Crossover", "RSI Mean Reversion", "Daily Breakout"])]
        if not simple_rows.empty and float(selected_row["cagr"]) < float(simple_rows["cagr"].max()):
            st.warning("QQE/HMA underperformed at least one simpler strategy in this qualification run. Added complexity may not be justified yet.")
        if int(selected_row["number_of_trades"]) < 20:
            st.warning("QQE/HMA generated very few trades in this run. Evidence is weak for a higher-parameter strategy.")
        if float(selected_row["exposure_pct"]) < 0.2:
            st.warning("QQE/HMA spent long periods out of the market. Recheck whether it only worked in one favorable regime.")

    slippage_levels = [0.0, 0.0002, 0.0005, 0.0010, 0.0025]
    if st.button("Run Slippage Sensitivity", key="qualification_slippage_run"):
        builders = {name: (lambda strategy_name=name: build_strategy(strategy_name, default_strategy_params(strategy_name))) for name in strategy_names}
        engine = BacktestEngine(database=None)
        slippage_results = run_slippage_sensitivity(
            engine,
            builders,
            qualification["data_by_symbol"],
            base_config,
            benchmark_symbol,
            slippage_levels,
        )
        st.session_state[SESSION_SLIPPAGE_KEY] = slippage_results
    slippage_results = st.session_state.get(SESSION_SLIPPAGE_KEY)
    if slippage_results is not None and not slippage_results.empty:
        st.subheader("Slippage Sensitivity")
        st.dataframe(slippage_results, use_container_width=True)
        metric_name = st.selectbox("Slippage chart metric", ["CAGR", "Max Drawdown", "Profit Factor", "Number of Trades"], key="qualification_slippage_metric")
        st.plotly_chart(px.line(slippage_results, x="slippage_pct", y=metric_name, color="strategy_name", markers=True, title=f"{metric_name} by Slippage"), use_container_width=True)
        for warning in summarize_slippage_warnings(slippage_results):
            st.warning(warning)

    stability_frame = build_strategy_sweep_stability(db)
    st.subheader("Saved Sweep Stability Across Strategies")
    if stability_frame.empty:
        st.info("No saved sweeps are available yet for strategy-level stability comparison.")
    else:
        st.dataframe(stability_frame, use_container_width=True)


def render_forward_paper_trading(
    *,
    db: TradingLabDatabase,
    provider: YFinanceDataProvider,
    benchmark_symbol: str,
    refresh_data: bool,
    base_config: BacktestConfig,
    default_tickers: list[str],
    current_strategy_name: str,
    current_strategy_params: dict[str, Any],
) -> None:
    st.header("Forward Paper Trading")
    st.caption("Forward paper trading is a local daily-bar simulation. It is separate from the manual paper journal and does not place trades.")
    engine = ForwardPaperEngine()

    st.subheader("Promotion Workflow")
    promotion_source = st.selectbox(
        "Promote from",
        ["Strategy Qualification result", "Saved Backtest", "Saved Sweep result", "Manual strategy configuration"],
        key="forward_promo_source",
    )

    linked_run: dict[str, Any] | None = None
    linked_qualification_id: str | None = None
    linked_sweep_id: str | None = None
    strategy_name = current_strategy_name
    strategy_params = dict(current_strategy_params)
    universe_name = "Custom"
    selected_tickers = list(default_tickers)
    saved_defaults = {
        "initial_capital": float(base_config.initial_capital),
        "position_sizing_method": str(base_config.position_sizing_method),
        "position_size_value": float(base_config.position_size_value),
        "max_positions": int(base_config.max_positions),
        "slippage_pct": float(base_config.slippage_pct),
        "commission_per_trade": float(base_config.commission_per_trade),
        "price_mode": str(base_config.price_mode),
        "stop_loss_pct": float(base_config.stop_loss_pct or 0.0),
        "take_profit_pct": float(base_config.take_profit_pct or 0.0),
        "trailing_stop_pct": float(base_config.trailing_stop_pct or 0.0),
    }

    if promotion_source == "Strategy Qualification result":
        qualification_runs = db.list_strategy_qualification_runs(limit=100)
        if qualification_runs.empty:
            st.info("No saved qualification runs are available yet.")
        else:
            selected_qualification_id = st.selectbox("Qualification run", qualification_runs["qualification_id"].tolist(), key="forward_qual_id")
            linked_qualification_id = selected_qualification_id
            qualification_run = db.get_strategy_qualification_run(selected_qualification_id) or {}
            qualification_results = db.read_strategy_qualification_results(selected_qualification_id)
            if not qualification_results.empty:
                option_labels = [
                    f"{row.strategy_name} | CAGR {float(row.cagr):.1%} | Robustness {int(row.robustness_score)}"
                    for row in qualification_results.itertuples()
                ]
                selected_label = st.selectbox("Qualification result", option_labels, key="forward_qual_result")
                selected_result = qualification_results.iloc[option_labels.index(selected_label)]
                strategy_name = str(selected_result["strategy_name"])
                linked_run = db.get_backtest_run(str(selected_result["backtest_run_id"])) if pd.notna(selected_result.get("backtest_run_id")) else None
                strategy_params = extract_saved_run_strategy_params(linked_run) if linked_run else default_strategy_params(strategy_name)
                universe_name = str(qualification_run.get("universe_name") or "Custom")
                selected_tickers = normalize_ticker_list(str(qualification_run.get("tickers") or ""))
                saved_defaults.update(extract_saved_run_config_defaults(linked_run))
    elif promotion_source == "Saved Backtest":
        saved_runs = db.list_backtest_runs(limit=200)
        if saved_runs.empty:
            st.info("No saved backtests are available yet.")
        else:
            selected_run_id = st.selectbox("Saved backtest run", saved_runs["run_id"].tolist(), key="forward_run_id")
            linked_run = db.get_backtest_run(selected_run_id)
            if linked_run is not None:
                strategy_name = normalize_strategy_display_name(str(linked_run["strategy_name"]))
                strategy_params = extract_saved_run_strategy_params(linked_run)
                selected_tickers = normalize_ticker_list(str(linked_run.get("symbols_csv") or ""))
                saved_defaults.update(extract_saved_run_config_defaults(linked_run))
    elif promotion_source == "Saved Sweep result":
        sweep_runs = db.list_sweep_runs(limit=100)
        if sweep_runs.empty:
            st.info("No saved sweeps are available yet.")
        else:
            selected_sweep_id = st.selectbox("Sweep run", sweep_runs["sweep_id"].tolist(), key="forward_sweep_id")
            linked_sweep_id = selected_sweep_id
            sweep_run = db.get_sweep_run(selected_sweep_id) or {}
            sweep_results = db.read_sweep_results(selected_sweep_id)
            if not sweep_results.empty:
                option_labels = [
                    f"{row.sweep_result_id} | CAGR {float(row.cagr):.1%} | Trades {int(row.number_of_trades)}"
                    for row in sweep_results.itertuples()
                ]
                selected_label = st.selectbox("Sweep result", option_labels, key="forward_sweep_result")
                selected_sweep_result = sweep_results.iloc[option_labels.index(selected_label)]
                strategy_name = normalize_strategy_display_name(str(sweep_run.get("strategy_name") or current_strategy_name))
                strategy_params = safe_json_loads(selected_sweep_result.get("parameter_json"), {}) or default_strategy_params(strategy_name)
                selected_tickers = normalize_ticker_list(str(sweep_run.get("tickers") or ""))
                linked_run = db.get_backtest_run(str(selected_sweep_result["backtest_run_id"])) if pd.notna(selected_sweep_result.get("backtest_run_id")) else None
                saved_defaults["initial_capital"] = float(sweep_run.get("initial_capital", saved_defaults["initial_capital"]) or saved_defaults["initial_capital"])
                saved_defaults["price_mode"] = str(sweep_run.get("price_mode", saved_defaults["price_mode"]) or saved_defaults["price_mode"])
    else:
        universe_name = st.selectbox("Manual universe", list_universe_names(), index=list_universe_names().index("Custom"), key="forward_manual_universe")
        manual_default = ",".join(default_tickers if universe_name == "Custom" else get_universe_tickers(universe_name))
        selected_tickers = normalize_ticker_list(st.text_input("Manual tickers", value=manual_default, key="forward_manual_tickers"))

    strategy_name = normalize_strategy_display_name(strategy_name)
    if not selected_tickers:
        selected_tickers = list(default_tickers)

    robustness_score = 0
    train_test_summary = None
    walk_forward_summary = None
    parameter_stability = None
    benchmark_warning_count = 0
    if linked_run is not None:
        robustness_frame = db.read_robustness_score(str(linked_run["run_id"]))
        if not robustness_frame.empty:
            robustness_score = int(robustness_frame.iloc[0]["score"] or 0)
        train_test_summary = load_run_train_test_payload(db, str(linked_run["run_id"]))
        walk_forward_summary = load_run_walk_forward_payload(db, str(linked_run["run_id"]))
        parameter_stability = latest_parameter_stability_for_strategy(db, strategy_name)
        benchmark_diag = db.read_benchmark_diagnostics(str(linked_run["run_id"]))
        if not benchmark_diag.empty:
            benchmark_warning_count = len(safe_json_loads(benchmark_diag.iloc[0]["warnings_json"], []))

    checklist = build_promotion_checklist(
        run_record=linked_run,
        robustness_score=robustness_score,
        train_test_summary=train_test_summary,
        walk_forward_summary=walk_forward_summary,
        parameter_stability=parameter_stability,
        benchmark_warning_count=benchmark_warning_count,
    )
    st.write("Promotion Checklist")
    st.dataframe(checklist, use_container_width=True)

    config_cols = st.columns(3)
    price_mode_options = ["raw_price_mode", "adjusted_price_mode"]
    selected_price_mode_index = price_mode_options.index(saved_defaults["price_mode"]) if saved_defaults["price_mode"] in price_mode_options else 0
    active_price_mode = config_cols[0].selectbox("Forward price mode", price_mode_options, index=selected_price_mode_index, key="forward_price_mode")
    active_initial_capital = config_cols[1].number_input("Forward initial capital", min_value=1000.0, value=float(saved_defaults["initial_capital"]), key="forward_initial_capital")
    active_benchmark = config_cols[2].text_input("Forward benchmark symbol", value=benchmark_symbol, key="forward_benchmark").upper().strip()

    sizing_cols = st.columns(4)
    sizing_methods = ["fixed_dollar", "percent_of_portfolio", "fixed_dollar_risk"]
    sizing_index = sizing_methods.index(str(saved_defaults["position_sizing_method"])) if str(saved_defaults["position_sizing_method"]) in sizing_methods else 1
    active_sizing_method = sizing_cols[0].selectbox("Position sizing", sizing_methods, index=sizing_index, key="forward_sizing_method")
    active_sizing_value = sizing_cols[1].number_input("Sizing value", min_value=0.0, value=float(saved_defaults["position_size_value"]), key="forward_sizing_value")
    active_max_positions = sizing_cols[2].number_input("Max positions", min_value=1, value=int(saved_defaults["max_positions"]), key="forward_max_positions")
    active_slippage = sizing_cols[3].number_input("Slippage %", min_value=0.0, value=float(saved_defaults["slippage_pct"]), format="%.5f", key="forward_slippage")

    risk_cols = st.columns(5)
    active_commission = risk_cols[0].number_input("Commission per trade", min_value=0.0, value=float(saved_defaults["commission_per_trade"]), key="forward_commission")
    active_stop_loss = risk_cols[1].number_input("Stop loss %", min_value=0.0, value=float(saved_defaults["stop_loss_pct"]), format="%.4f", key="forward_stop_loss")
    active_take_profit = risk_cols[2].number_input("Take profit %", min_value=0.0, value=float(saved_defaults["take_profit_pct"]), format="%.4f", key="forward_take_profit")
    active_trailing_stop = risk_cols[3].number_input("Trailing stop %", min_value=0.0, value=float(saved_defaults["trailing_stop_pct"]), format="%.4f", key="forward_trailing_stop")
    active_fill_rule = risk_cols[4].selectbox("Fill rule", ["next_open", "next_close"], key="forward_fill_rule", help="Signals are generated from the close. `next_open` is the default and more realistic fill assumption.")
    active_ambiguity_rule = st.selectbox(
        "Same-bar stop/target ambiguity",
        ["conservative_stop_first", "target_first", "skip_ambiguous"],
        key="forward_ambiguity_rule",
        help="If both stop and target are hit in the same daily bar, the default assumes the stop is hit first.",
    )
    activation_reason = st.text_area("Activation reason", value="", key="forward_activation_reason")
    activation_notes = st.text_area("Active strategy notes", value="", key="forward_activation_notes")
    activation_tags = st.text_input("Active strategy tags", value="", key="forward_activation_tags")
    confirm_activation = st.checkbox("I understand this strategy will be auto-tracked in local forward paper trading.", key="forward_activate_confirm")

    if st.button("Activate Forward Paper Strategy", key="forward_activate_button", type="primary"):
        if not confirm_activation:
            st.error("Confirm the activation checkbox before promoting a strategy.")
        elif not selected_tickers:
            st.error("Select at least one ticker before activating a forward paper strategy.")
        else:
            payload = build_active_paper_strategy_payload(
                strategy_name=strategy_name,
                strategy_parameters=strategy_params,
                universe_name=universe_name,
                tickers=selected_tickers,
                benchmark_symbol=active_benchmark,
                price_mode=active_price_mode,
                initial_capital=float(active_initial_capital),
                position_sizing_method=active_sizing_method,
                position_sizing_value=float(active_sizing_value),
                max_positions=int(active_max_positions),
                risk_settings={
                    "stop_loss_pct": float(active_stop_loss),
                    "take_profit_pct": float(active_take_profit),
                    "trailing_stop_pct": float(active_trailing_stop),
                    "fill_rule": active_fill_rule,
                    "same_bar_stop_target_rule": active_ambiguity_rule,
                },
                slippage_pct=float(active_slippage),
                commission_per_trade=float(active_commission),
                linked_qualification_id=linked_qualification_id,
                linked_sweep_id=linked_sweep_id,
                linked_backtest_run_id=str(linked_run["run_id"]) if linked_run is not None else None,
                activation_reason=activation_reason,
                notes=activation_notes,
                tags=activation_tags,
                status="active",
            )
            db.insert_active_paper_strategy(payload)
            db.insert_active_paper_strategy_event(
                {
                    "event_id": str(uuid4()),
                    "active_strategy_id": payload["active_strategy_id"],
                    "created_at": datetime.now(UTC).replace(tzinfo=None),
                    "event_type": "activation",
                    "message": f"{strategy_name} promoted to active forward paper trading.",
                    "details_json": json.dumps({"source": promotion_source, "tickers": selected_tickers, "benchmark_symbol": active_benchmark}, default=str),
                }
            )
            st.success(f"Activated forward paper strategy: {payload['active_strategy_id']}")

    st.divider()
    st.subheader("Active Paper Strategies")
    active_status_filter = st.selectbox("Status filter", ["All", "active", "paused", "retired", "draft"], key="forward_status_filter")
    active_strategies = db.list_active_paper_strategies(None if active_status_filter == "All" else active_status_filter)
    if active_strategies.empty:
        st.info("No active paper strategies are saved yet.")
        return
    st.dataframe(active_strategies, use_container_width=True)

    selected_active_id = st.selectbox("Select active paper strategy", active_strategies["active_strategy_id"].tolist(), key="forward_active_select")
    selected_active = db.get_active_paper_strategy(selected_active_id)
    if selected_active is None:
        st.warning("The selected active paper strategy could not be loaded.")
        return

    action_cols = st.columns(4)
    if action_cols[0].button("Run Forward Paper Update", key="forward_run_update"):
        update_results: list[str] = []
        for strategy_row in active_strategies.itertuples():
            if str(strategy_row.status) != "active":
                continue
            strategy_payload = db.get_active_paper_strategy(str(strategy_row.active_strategy_id))
            if strategy_payload is None:
                continue
            result = engine.run_update(active_strategy=strategy_payload, provider=provider)
            db.replace_forward_engine_events(result.active_strategy_id, result.events)
            if result.skipped:
                update_results.append(f"{result.active_strategy_id}: skipped")
                continue
            db.replace_forward_paper_state(result.active_strategy_id, result.orders, result.positions, result.trades, result.equity_curve)
            strategy_payload["current_paper_equity"] = result.current_equity
            strategy_payload["updated_at"] = datetime.now(UTC).replace(tzinfo=None)
            db.update_active_paper_strategy(strategy_payload)
            update_results.append(f"{result.active_strategy_id}: updated")
        st.session_state[SESSION_FORWARD_UPDATE_KEY] = update_results
        st.success("Forward paper update completed.")

    if action_cols[1].button("Pause / Resume", key="forward_toggle_status"):
        next_status = "paused" if str(selected_active.get("status")) == "active" else "active"
        selected_active["status"] = next_status
        selected_active["updated_at"] = datetime.now(UTC).replace(tzinfo=None)
        db.update_active_paper_strategy(selected_active)
        db.insert_active_paper_strategy_event(
            {
                "event_id": str(uuid4()),
                "active_strategy_id": selected_active_id,
                "created_at": datetime.now(UTC).replace(tzinfo=None),
                "event_type": "status_change",
                "message": f"Strategy status changed to {next_status}.",
                "details_json": json.dumps({"status": next_status}),
            }
        )
        st.success(f"Strategy status updated to {next_status}.")
    if action_cols[2].button("Retire Strategy", key="forward_retire_strategy"):
        selected_active["status"] = "retired"
        selected_active["updated_at"] = datetime.now(UTC).replace(tzinfo=None)
        db.update_active_paper_strategy(selected_active)
        db.insert_active_paper_strategy_event(
            {
                "event_id": str(uuid4()),
                "active_strategy_id": selected_active_id,
                "created_at": datetime.now(UTC).replace(tzinfo=None),
                "event_type": "status_change",
                "message": "Strategy retired from forward paper trading.",
                "details_json": json.dumps({"status": "retired"}),
            }
        )
        st.success("Strategy retired.")
    if action_cols[3].button("Refresh Active Strategy Data", key="forward_refresh_data"):
        st.info("Use the normal Refresh data control before a forward update when you want a forced cache refresh.")

    orders = db.read_forward_paper_orders(selected_active_id)
    positions = db.read_forward_paper_positions(selected_active_id)
    trades = db.read_forward_paper_trades(selected_active_id)
    equity_curve = db.read_forward_paper_equity_curve(selected_active_id)
    events = db.read_active_paper_strategy_events(selected_active_id)

    st.subheader("Selected Active Strategy")
    st.json(selected_active, expanded=False)
    if not equity_curve.empty:
        metric_trades = trades.rename(columns={"realized_pnl": "pnl", "realized_return_pct": "return_pct"}).copy() if not trades.empty else pd.DataFrame(columns=["pnl", "return_pct", "entry_date", "exit_date"])
        if not metric_trades.empty:
            metric_trades["holding_days"] = (pd.to_datetime(metric_trades["exit_date"]) - pd.to_datetime(metric_trades["entry_date"])).dt.days.clip(lower=0)
        forward_metrics = compute_summary_metrics(equity_curve, metric_trades, float(selected_active.get("initial_capital", 0.0) or 0.0))
        summary_cols = st.columns(6)
        summary_cols[0].metric("Current Paper Equity", f"{float(equity_curve['equity'].iloc[-1]):,.2f}")
        summary_cols[1].metric("Open Positions", int((positions["status"] == "open").sum()) if not positions.empty else 0)
        summary_cols[2].metric("Pending Orders", int((orders["status"] == "pending").sum()) if not orders.empty else 0)
        summary_cols[3].metric("Closed Trades", int(forward_metrics.get("Number of Trades", 0)))
        summary_cols[4].metric("Win Rate", f"{float(forward_metrics.get('Win Rate', 0.0)):.1%}")
        summary_cols[5].metric("Profit Factor", f"{float(forward_metrics.get('Profit Factor', 0.0)):.2f}")
        st.metric("Expectancy", f"{float(trades['realized_pnl'].mean() if not trades.empty else 0.0):.2f}")
        st.plotly_chart(build_equity_chart(equity_curve, pd.DataFrame()), use_container_width=True)
        st.plotly_chart(build_drawdown_chart(equity_curve), use_container_width=True)

        linked_backtest_run = db.get_backtest_run(str(selected_active.get("linked_backtest_run_id"))) if selected_active.get("linked_backtest_run_id") else None
        activation_ts = pd.Timestamp(selected_active["created_at"]).normalize()
        validation_warnings = compare_forward_to_backtest(
            backtest_run=linked_backtest_run,
            forward_metrics=forward_metrics,
            days_since_activation=int((pd.Timestamp.today().normalize() - activation_ts).days),
        )
        st.subheader("Forward Validation Report")
        if linked_backtest_run is not None:
            comparison = pd.DataFrame(
                [
                    {"metric": "CAGR", "original_backtest": float(linked_backtest_run.get("cagr", 0.0) or 0.0), "forward_paper": float(forward_metrics.get("CAGR", 0.0) or 0.0)},
                    {"metric": "Max Drawdown", "original_backtest": float(linked_backtest_run.get("max_drawdown", 0.0) or 0.0), "forward_paper": float(forward_metrics.get("Max Drawdown", 0.0) or 0.0)},
                    {"metric": "Win Rate", "original_backtest": float(linked_backtest_run.get("win_rate", 0.0) or 0.0), "forward_paper": float(forward_metrics.get("Win Rate", 0.0) or 0.0)},
                    {"metric": "Profit Factor", "original_backtest": float(linked_backtest_run.get("profit_factor", 0.0) or 0.0), "forward_paper": float(forward_metrics.get("Profit Factor", 0.0) or 0.0)},
                ]
            )
            st.dataframe(comparison, use_container_width=True)
        render_warning_list(validation_warnings, "No forward validation warnings were generated yet.")
    else:
        st.info("Run a forward paper update to build orders, positions, trades, and the forward equity curve.")

    if not orders.empty:
        st.subheader("Forward Orders")
        st.dataframe(orders, use_container_width=True)
    if not positions.empty:
        st.subheader("Forward Positions")
        st.dataframe(positions, use_container_width=True)
    if not trades.empty:
        st.subheader("Forward Trades")
        st.dataframe(trades, use_container_width=True)
        st.download_button(
            "Export forward trades CSV",
            data=trades.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_active_id}_forward_trades.csv",
            mime="text/csv",
        )
    if not events.empty:
        st.subheader("Event Log")
        st.dataframe(events, use_container_width=True)


def render_spy_strategy_lab(
    *,
    db: TradingLabDatabase,
    provider: YFinanceDataProvider,
    base_config: BacktestConfig,
    start_date: str,
    end_date: str,
    refresh_data: bool,
) -> None:
    st.header("SPY Workbench")
    st.caption("Start here. Run automated SPY search, review ranked candidates, then promote only one strategy to forward paper trading.")

    st.subheader("A. Strategy Setup")
    st.info("This workbench is fixed to SPY. It is the recommended primary workflow for the app.")
    setup_cols = st.columns([1.4, 1.1])
    timeframe_label = setup_cols[0].selectbox("Timeframe", ["Daily", "15-minute", "5-minute experimental"], key="spy_workbench_timeframe")
    timeframe_map = {"Daily": "1d", "15-minute": "15m", "5-minute experimental": "5m"}
    selected_timeframe = timeframe_map[timeframe_label]
    if selected_timeframe != "1d":
        st.warning(
            "yfinance intraday history is limited to roughly the last "
            f"{INTRADAY_MAX_HISTORY_DAYS} days. Intraday results are short-history only, and 5-minute mode is more noise-sensitive."
        )
    presets = list_spy_strategy_presets(selected_timeframe)
    preset_labels = [preset.label for preset in presets]
    selected_label = setup_cols[0].selectbox("Entry strategy", preset_labels, key="spy_workbench_preset")
    preset = next(item for item in presets if item.label == selected_label)
    setup_cols[0].write(preset.description)
    if preset.experimental:
        setup_cols[0].warning("Experimental preset. It needs stronger evidence than the simpler SPY strategies.")
    setup_cols[1].write({"Ticker": "SPY", "Benchmark": "SPY buy-and-hold", "Timeframe": selected_timeframe})

    exit_structures = list_spy_exit_structures()
    exit_catalog = pd.DataFrame(
        [{"Exit structure": item.label, "Status": "Implemented" if item.enabled else "Planned", "Notes": item.description} for item in exit_structures]
    )
    st.dataframe(exit_catalog, use_container_width=True, hide_index=True)
    enabled_exit_structures = [item for item in exit_structures if item.enabled]
    exit_label = st.selectbox("Exit structure", [item.label for item in enabled_exit_structures], key="spy_workbench_exit")
    exit_structure = next(item for item in enabled_exit_structures if item.label == exit_label)

    customize = st.checkbox("Customize entry parameters", value=False, key="spy_workbench_customize")
    custom_params = dict(preset.parameters)
    if customize:
        if preset.key == "trend_filter_200":
            custom_params["sma_length"] = st.number_input("Trend SMA length", min_value=50, value=int(custom_params["sma_length"]), key="spy_workbench_sma_length")
        elif preset.key == "moving_average_50_200":
            custom_params["fast_window"] = st.number_input("Fast SMA", min_value=2, value=int(custom_params["fast_window"]), key="spy_workbench_fast")
            custom_params["slow_window"] = st.number_input("Slow SMA", min_value=3, value=int(custom_params["slow_window"]), key="spy_workbench_slow")
        elif preset.key == "rsi_pullback_uptrend":
            custom_params["rsi_length"] = st.number_input("RSI length", min_value=2, value=int(custom_params["rsi_length"]), key="spy_workbench_rsi_length")
            custom_params["buy_threshold"] = st.number_input("RSI buy threshold", min_value=1.0, max_value=50.0, value=float(custom_params["buy_threshold"]), key="spy_workbench_rsi_buy")
            custom_params["sell_threshold"] = st.number_input("RSI sell threshold", min_value=50.0, max_value=100.0, value=float(custom_params["sell_threshold"]), key="spy_workbench_rsi_sell")
            custom_params["max_holding_days"] = st.number_input("Max holding days", min_value=1, value=int(custom_params["max_holding_days"]), key="spy_workbench_max_hold")
            custom_params["trend_sma_window"] = st.number_input("Trend SMA", min_value=50, value=int(custom_params["trend_sma_window"]), key="spy_workbench_trend_sma")
        elif preset.key == "breakout_50_20":
            custom_params["lookback_window"] = st.number_input("Breakout lookback", min_value=2, value=int(custom_params["lookback_window"]), key="spy_workbench_breakout_lookback")
            custom_params["exit_lookback_window"] = st.number_input("Signal exit lookback", min_value=2, value=int(custom_params["exit_lookback_window"]), key="spy_workbench_exit_lookback")
        elif preset.key == "intraday_pullback":
            custom_params["rsi_length"] = st.number_input("Intraday RSI length", min_value=2, value=int(custom_params["rsi_length"]), key="spy_workbench_intraday_rsi_length")
            custom_params["oversold_threshold"] = st.number_input("Oversold threshold", min_value=1.0, max_value=60.0, value=float(custom_params["oversold_threshold"]), key="spy_workbench_intraday_oversold")
            custom_params["recovery_threshold"] = st.number_input("Recovery threshold", min_value=1.0, max_value=90.0, value=float(custom_params["recovery_threshold"]), key="spy_workbench_intraday_recovery")
            custom_params["moving_average_length"] = st.number_input("Intraday MA length", min_value=2, value=int(custom_params["moving_average_length"]), key="spy_workbench_intraday_ma")
            custom_params["pullback_lookback_bars"] = st.number_input("Pullback lookback bars", min_value=1, value=int(custom_params["pullback_lookback_bars"]), key="spy_workbench_intraday_pullback_lookback")
            custom_params["end_of_day_exit"] = st.checkbox("End-of-day exit", value=bool(custom_params["end_of_day_exit"]), key="spy_workbench_intraday_eod")
            custom_params["allow_overnight"] = st.checkbox("Allow overnight", value=bool(custom_params["allow_overnight"]), key="spy_workbench_intraday_overnight")
        elif preset.key == "intraday_breakout":
            custom_params["breakout_lookback_bars"] = st.number_input("Breakout lookback bars", min_value=2, value=int(custom_params["breakout_lookback_bars"]), key="spy_workbench_intraday_breakout")
            custom_params["exit_lookback_bars"] = st.number_input("Exit lookback bars", min_value=2, value=int(custom_params["exit_lookback_bars"]), key="spy_workbench_intraday_exit")
            custom_params["end_of_day_exit"] = st.checkbox("End-of-day exit", value=bool(custom_params["end_of_day_exit"]), key="spy_workbench_intraday_breakout_eod")
            custom_params["allow_overnight"] = st.checkbox("Allow overnight", value=bool(custom_params["allow_overnight"]), key="spy_workbench_intraday_breakout_overnight")
        else:
            custom_params["hma_length"] = st.number_input("HMA length", min_value=2, value=int(custom_params["hma_length"]), key="spy_workbench_hma_length")
            custom_params["rsi_length"] = st.number_input("QQE RSI length", min_value=2, value=int(custom_params["rsi_length"]), key="spy_workbench_qqe_rsi_length")
            custom_params["rsi_smoothing"] = st.number_input("QQE RSI smoothing", min_value=1, value=int(custom_params["rsi_smoothing"]), key="spy_workbench_qqe_rsi_smoothing")
            custom_params["qqe_factor"] = st.number_input("QQE factor", min_value=0.1, value=float(custom_params["qqe_factor"]), step=0.1, key="spy_workbench_qqe_factor")
            custom_params["atr_smoothing"] = st.number_input("QQE ATR smoothing", min_value=1, value=int(custom_params["atr_smoothing"]), key="spy_workbench_qqe_atr")
            custom_params["require_hma_slope"] = st.checkbox("Require HMA slope", value=bool(custom_params["require_hma_slope"]), key="spy_workbench_hma_slope")
            custom_params["exit_on_hma_break"] = st.checkbox("Exit on HMA break", value=bool(custom_params["exit_on_hma_break"]), key="spy_workbench_exit_hma")
            custom_params["exit_on_qqe_bearish"] = st.checkbox("Exit on QQE bearish turn", value=bool(custom_params["exit_on_qqe_bearish"]), key="spy_workbench_exit_qqe")

    exit_params = dict(exit_structure.default_params)
    if exit_structure.key in {"fixed_stop_loss", "oco_bracket", "stop_loss_plus_trailing_stop"}:
        exit_params["stop_loss_pct"] = st.number_input("Stop loss %", min_value=0.0, value=float(exit_params.get("stop_loss_pct", 0.08)), format="%.4f", key="spy_workbench_stop")
    if exit_structure.key in {"fixed_take_profit", "oco_bracket"}:
        exit_params["take_profit_pct"] = st.number_input("Take profit %", min_value=0.0, value=float(exit_params.get("take_profit_pct", 0.15)), format="%.4f", key="spy_workbench_take")
    if exit_structure.key in {"trailing_stop", "stop_loss_plus_trailing_stop"}:
        exit_params["trailing_stop_pct"] = st.number_input("Trailing stop %", min_value=0.0, value=float(exit_params.get("trailing_stop_pct", 0.10)), format="%.4f", key="spy_workbench_trailing")
    if exit_structure.key == "time_stop":
        exit_params["max_holding_days"] = st.number_input("Time stop holding days", min_value=1, value=int(exit_params.get("max_holding_days", 20)), key="spy_workbench_time_stop")

    config_cols = st.columns(4)
    workbench_start = str(config_cols[0].date_input("Start date", value=pd.Timestamp(start_date), key="spy_workbench_start"))
    workbench_end = str(config_cols[1].date_input("End date", value=pd.Timestamp(end_date), key="spy_workbench_end"))
    workbench_price_mode = config_cols[2].selectbox("Price mode", ["raw_price_mode", "adjusted_price_mode"], index=0 if base_config.price_mode == "raw_price_mode" else 1, key="spy_workbench_price_mode")
    workbench_initial_capital = config_cols[3].number_input("Initial capital", min_value=1000.0, value=float(base_config.initial_capital), key="spy_workbench_capital")

    risk_cols = st.columns(5)
    workbench_sizing_method = risk_cols[0].selectbox("Position sizing", ["fixed_dollar", "percent_of_portfolio"], index=0 if base_config.position_sizing_method == "fixed_dollar" else 1, key="spy_workbench_sizing_method")
    workbench_sizing_value = risk_cols[1].number_input("Sizing value", min_value=0.0, value=float(base_config.position_size_value), key="spy_workbench_sizing_value")
    workbench_max_positions = risk_cols[2].number_input("Max positions", min_value=1, value=1, key="spy_workbench_max_positions")
    workbench_slippage = risk_cols[3].number_input("Slippage %", min_value=0.0, value=float(base_config.slippage_pct), format="%.5f", key="spy_workbench_slippage")
    workbench_commission = risk_cols[4].number_input("Commission per trade", min_value=0.0, value=float(base_config.commission_per_trade), key="spy_workbench_commission")

    workbench = build_spy_workbench_config(
        preset_key=preset.key,
        entry_parameters=custom_params,
        timeframe=selected_timeframe,
        exit_structure_key=exit_structure.key,
        exit_parameters=exit_params,
        start_date=workbench_start,
        end_date=workbench_end,
        price_mode=workbench_price_mode,
        initial_capital=float(workbench_initial_capital),
        position_sizing_method=workbench_sizing_method,
        position_size_value=float(workbench_sizing_value),
        max_positions=int(workbench_max_positions),
        slippage_pct=float(workbench_slippage),
        commission_per_trade=float(workbench_commission),
    )

    st.subheader("A. Automated SPY Search")
    st.caption("Run the approved SPY strategy-and-exit grid automatically, rank the candidates, then promote one exact configuration into forward paper trading.")
    approved_entry_count = len(generate_approved_spy_entry_presets(workbench.timeframe))
    approved_exit_count = len(generate_approved_spy_exit_presets())
    combination_count = len(generate_spy_search_combinations(workbench.timeframe))
    search_cols = st.columns(4)
    search_cols[0].metric("Entry presets", approved_entry_count)
    search_cols[1].metric("Exit presets", approved_exit_count)
    search_cols[2].metric("Total combinations", combination_count)
    search_cols[3].metric("Ticker / Benchmark", "SPY / SPY")
    if combination_count > 350:
        st.warning("This search grid is larger than the preferred default. Expect slower runs.")
    search_notes = st.text_area("Search notes", value="", key="spy_search_notes", help="Optional notes saved with the automated SPY search run.")
    search_tags = st.text_input("Search tags", value="spy-only,automated-search", key="spy_search_tags")
    if st.button("Run Automated SPY Search", key="spy_search_run_button", type="secondary"):
        with st.spinner("Running automated SPY strategy search..."):
            data_by_symbol, statuses, validation_warnings = collect_data(provider, ["SPY"], workbench.start_date, workbench.end_date, refresh_data, benchmark_symbol="SPY", timeframe=workbench.timeframe)
            daily_context = None
            if is_intraday_timeframe(workbench.timeframe):
                daily_context = provider.get_stock_bars(
                    symbol="SPY",
                    start_date=str((pd.Timestamp(workbench.start_date) - pd.Timedelta(days=450)).date()),
                    end_date=workbench.end_date,
                    timeframe="1d",
                    force_refresh=refresh_data,
                )
                daily_status = provider.get_last_fetch_status("SPY")
                if daily_status is not None:
                    statuses.append(daily_status)
                    validation_warnings.extend([f"SPY daily regime: {warning}" for warning in daily_status.validation_warnings])
            progress = st.progress(0.0, text="Starting automated SPY search...")
            progress_rows: list[str] = []

            def _progress_callback(current: int, total: int, row: dict[str, Any]) -> None:
                progress.progress(
                    min(current / total, 1.0),
                    text=f"Testing {current}/{total}: {row['entry_preset_label']} + {row['exit_preset_label']}",
                )
                progress_rows.append(f"{row['entry_preset_label']} + {row['exit_preset_label']}")

            payload, results, highlights = run_automated_spy_search(
                engine=BacktestEngine(database=db),
                data_by_symbol={"SPY": data_by_symbol["SPY"]},
                timeframe=workbench.timeframe,
                start_date=workbench.start_date,
                end_date=workbench.end_date,
                price_mode=workbench.price_mode,
                initial_capital=workbench.initial_capital,
                position_sizing_method=workbench.position_sizing_method,
                position_sizing_value=workbench.position_size_value,
                slippage_pct=workbench.slippage_pct,
                commission_per_trade=workbench.commission_per_trade,
                daily_bars=daily_context,
                persist_backtest_runs=False,
                progress_callback=_progress_callback,
            )
            payload["notes"] = search_notes
            payload["tags"] = search_tags
            db.replace_spy_strategy_search_run(payload, results)
            progress.progress(1.0, text="Automated SPY search complete.")
            st.session_state[SESSION_SPY_SEARCH_KEY] = {
                "payload": payload,
                "results": results,
                "highlights": highlights,
                "statuses": statuses,
                "validation_warnings": validation_warnings,
            }

    saved_search_runs = db.list_spy_strategy_search_runs(limit=50)
    if not saved_search_runs.empty:
        saved_search_labels = {
            str(row["search_run_id"]): f"{row['created_at']} | {row['timeframe']} | {row['start_date']} to {row['end_date']} | {int(row['total_combinations_tested'])} combos"
            for _, row in saved_search_runs.iterrows()
        }
        selected_search_run_id = st.selectbox(
            "Saved search runs",
            options=list(saved_search_labels.keys()),
            format_func=lambda value: saved_search_labels[value],
            key="spy_search_saved_run",
        )
        if st.button("Load Saved Search Run", key="spy_search_load_saved"):
            saved_payload = db.get_spy_strategy_search_run(selected_search_run_id)
            saved_results = db.read_spy_strategy_search_results(selected_search_run_id)
            st.session_state[SESSION_SPY_SEARCH_KEY] = {
                "payload": saved_payload,
                "results": saved_results,
                "highlights": rank_spy_search_results(saved_results),
                "statuses": [],
                "validation_warnings": [],
            }

    if st.button("Run SPY Workbench Backtest", key="spy_workbench_run", type="primary"):
        with st.spinner("Running SPY trading workbench backtest..."):
            daily_context = None
            if is_intraday_timeframe(workbench.timeframe):
                data_by_symbol, statuses, validation_warnings = collect_data(provider, ["SPY"], workbench.start_date, workbench.end_date, refresh_data, benchmark_symbol="SPY", timeframe=workbench.timeframe)
                daily_context = provider.get_stock_bars(
                    symbol="SPY",
                    start_date=str((pd.Timestamp(workbench.start_date) - pd.Timedelta(days=450)).date()),
                    end_date=workbench.end_date,
                    timeframe="1d",
                    force_refresh=refresh_data,
                )
                daily_status = provider.get_last_fetch_status("SPY")
                if daily_status is not None:
                    statuses.append(daily_status)
                    validation_warnings.extend([f"SPY daily regime: {warning}" for warning in daily_status.validation_warnings])
            else:
                data_by_symbol, statuses, validation_warnings = collect_data(provider, ["SPY"], workbench.start_date, workbench.end_date, refresh_data, benchmark_symbol="SPY", timeframe=workbench.timeframe)
            strategy = apply_spy_exit_structure(build_spy_strategy(workbench.preset_key, workbench.entry_parameters), workbench)
            config = build_spy_backtest_config(workbench)
            engine = BacktestEngine(database=db)
            prepared_bars = prepare_spy_timeframe_bars(primary_bars=data_by_symbol["SPY"], timeframe=workbench.timeframe, daily_bars=daily_context)
            result = engine.run(data_by_symbol={"SPY": prepared_bars}, strategy=strategy, config=config, benchmark_symbol="SPY")
            result.metrics["Average R Multiple"] = average_r_multiple(result.trade_log, workbench.exit_parameters)
            research = analyze_current_result(db, {"SPY": prepared_bars}, result, config, workbench.entry_parameters, "SPY", ["SPY"], workbench.start_date, workbench.end_date)
        st.session_state[SESSION_SPY_LAB_KEY] = {
            "workbench": workbench,
            "preset_label": preset.label,
            "result": result,
            "research": research,
            "statuses": statuses,
            "validation_warnings": validation_warnings,
            "data_by_symbol": {"SPY": prepared_bars},
        }

    search_state = st.session_state.get(SESSION_SPY_SEARCH_KEY)
    if search_state and search_state.get("results") is not None:
        st.subheader("B. Ranked Candidates")
        results = search_state["results"]
        payload = search_state["payload"]
        highlights = search_state["highlights"]
        top_cols = st.columns(5)
        for idx, category in enumerate(["Best Overall", "Best Low Drawdown", "Best Risk Adjusted", "Best Simple Strategy", "Most Suspicious High Return"]):
            highlight = highlights.get(category)
            if highlight:
                top_cols[idx].metric(category, f"{highlight['entry_preset_label']} | {highlight['exit_preset_label']}")
        for category, highlight in highlights.items():
            st.write(f"- {category}: {highlight['summary_comment']}")

        filter_cols = st.columns(6)
        entry_filter = filter_cols[0].selectbox("Entry filter", ["All"] + sorted(results["entry_strategy_name"].dropna().unique().tolist()), key="spy_search_filter_entry")
        exit_filter = filter_cols[1].selectbox("Exit filter", ["All"] + sorted(results["exit_structure_name"].dropna().unique().tolist()), key="spy_search_filter_exit")
        label_filter = filter_cols[2].selectbox("Candidate label", ["All", "Strong candidate", "Possible candidate", "Not ready", "Reject"], key="spy_search_filter_label")
        min_trades_filter = int(filter_cols[3].number_input("Min trades", min_value=0, value=0, key="spy_search_filter_trades"))
        positive_excess_only = filter_cols[4].checkbox("Positive excess CAGR only", value=False, key="spy_search_filter_excess")
        max_drawdown_threshold = float(filter_cols[5].number_input("Max drawdown threshold", value=0.0, format="%.2f", key="spy_search_filter_drawdown"))
        min_pf_filter = st.number_input("Minimum profit factor", min_value=0.0, value=0.0, step=0.1, key="spy_search_filter_pf")

        filtered_results = results.copy()
        if entry_filter != "All":
            filtered_results = filtered_results[filtered_results["entry_strategy_name"] == entry_filter]
        if exit_filter != "All":
            filtered_results = filtered_results[filtered_results["exit_structure_name"] == exit_filter]
        if label_filter != "All":
            filtered_results = filtered_results[filtered_results["candidate_label"] == label_filter]
        filtered_results = filtered_results[filtered_results["number_of_trades"] >= min_trades_filter]
        if positive_excess_only:
            filtered_results = filtered_results[filtered_results["excess_cagr"] > 0]
        if max_drawdown_threshold > 0:
            filtered_results = filtered_results[filtered_results["max_drawdown"].abs() <= max_drawdown_threshold]
        filtered_results = filtered_results[filtered_results["profit_factor"] >= min_pf_filter]

        st.write(
            {
                "search_run_id": payload["search_run_id"],
                "timeframe": payload.get("timeframe", "1d"),
                "total_combinations_tested": payload["total_combinations_tested"],
                "notes": payload["notes"],
                "tags": payload["tags"],
            }
        )
        if str(payload.get("timeframe", "1d")) != "1d":
            st.warning("Intraday results have much shorter history and should not be compared directly to long-history daily results.")
        st.dataframe(
            filtered_results[
                [
                    "timeframe",
                    "entry_preset_label",
                    "exit_preset_label",
                    "candidate_label",
                    "ranking_category",
                    "cagr",
                    "spy_cagr",
                    "excess_cagr",
                    "max_drawdown",
                    "drawdown_improvement",
                    "sharpe",
                    "sortino",
                    "calmar",
                    "number_of_trades",
                    "win_rate",
                    "profit_factor",
                    "avg_r_multiple",
                    "exposure_pct",
                    "robustness_score",
                    "red_flag_count",
                    "summary_comment",
                ]
            ].sort_values(["candidate_label", "calmar", "excess_cagr"], ascending=[True, False, False]),
            use_container_width=True,
        )
        st.download_button(
            "Export automated SPY search CSV",
            data=filtered_results.to_csv(index=False).encode("utf-8"),
            file_name=f"spy_search_{payload['search_run_id']}.csv",
            mime="text/csv",
        )
        if not filtered_results.empty:
            result_choice = st.selectbox(
                "Select search result to promote",
                options=filtered_results["result_id"].tolist(),
                format_func=lambda value: (
                    f"{filtered_results.loc[filtered_results['result_id'] == value, 'entry_preset_label'].iloc[0]} | "
                    f"{filtered_results.loc[filtered_results['result_id'] == value, 'exit_preset_label'].iloc[0]} | "
                    f"{filtered_results.loc[filtered_results['result_id'] == value, 'candidate_label'].iloc[0]}"
                ),
                key="spy_search_promote_choice",
            )
            selected_result = filtered_results.loc[filtered_results["result_id"] == result_choice].iloc[0]
            st.subheader("C. Candidate Detail")
            st.json(
                {
                    "entry_strategy": selected_result["entry_preset_label"],
                    "entry_parameters": selected_result["entry_parameters_json"],
                    "exit_structure": selected_result["exit_preset_label"],
                    "exit_parameters": selected_result["exit_parameters_json"],
                    "candidate_label": selected_result["candidate_label"],
                    "robustness_score": int(selected_result["robustness_score"]),
                    "red_flag_count": int(selected_result["red_flag_count"]),
                },
                expanded=False,
            )
            st.caption(str(selected_result["summary_comment"]))
            search_promote_notes = st.text_area("Promotion notes for selected search result", value="", key="spy_search_promote_notes")
            search_promote_tags = st.text_input("Promotion tags for selected search result", value="spy-only,automated-search,promoted", key="spy_search_promote_tags")
            search_promote_confirm = st.checkbox("Freeze this exact automated SPY search result for forward paper trading.", key="spy_search_promote_confirm")
            if st.button("Promote Selected Result to Forward Paper", key="spy_search_promote_button"):
                if not search_promote_confirm:
                    st.error("Confirm the automated search promotion checkbox before continuing.")
                else:
                    promote_payload = build_active_paper_strategy_payload(
                        strategy_name=str(selected_result["entry_strategy_name"]),
                        strategy_parameters={**selected_result["entry_parameters_json"], "__workbench_exit_structure__": str(selected_result["exit_structure_key"])},
                        universe_name="SPY Workbench Automated Search",
                        tickers=["SPY"],
                        timeframe=str(selected_result.get("timeframe", workbench.timeframe)),
                        benchmark_symbol="SPY",
                        price_mode=workbench.price_mode,
                        initial_capital=workbench.initial_capital,
                        position_sizing_method=workbench.position_sizing_method,
                        position_sizing_value=workbench.position_size_value,
                        max_positions=1,
                        risk_settings={
                            **selected_result["exit_parameters_json"],
                            "fill_rule": "next_open",
                            "same_bar_stop_target_rule": "conservative_stop_first",
                            "exit_structure_key": selected_result["exit_structure_key"],
                            "exit_structure_name": selected_result["exit_structure_name"],
                            "end_of_day_exit": bool(selected_result["entry_parameters_json"].get("end_of_day_exit", False)),
                            "allow_overnight": bool(selected_result["entry_parameters_json"].get("allow_overnight", True)),
                        },
                        slippage_pct=workbench.slippage_pct,
                        commission_per_trade=workbench.commission_per_trade,
                        linked_backtest_run_id=str(selected_result["backtest_run_id"]) if pd.notna(selected_result["backtest_run_id"]) and selected_result["backtest_run_id"] else None,
                        linked_search_run_id=str(payload["search_run_id"]),
                        linked_search_result_id=str(selected_result["result_id"]),
                        activation_reason=f"Promoted from automated SPY search: {selected_result['entry_preset_label']} with {selected_result['exit_preset_label']}.",
                        notes=search_promote_notes,
                        tags=search_promote_tags,
                        status="active",
                    )
                    db.insert_active_paper_strategy(promote_payload)
                    db.insert_active_paper_strategy_event(
                        {
                            "event_id": str(uuid4()),
                            "active_strategy_id": promote_payload["active_strategy_id"],
                            "created_at": datetime.now(UTC).replace(tzinfo=None),
                            "event_type": "activation",
                            "message": f"SPY-only forward paper strategy activated from automated search result {selected_result['result_id']}.",
                            "details_json": json.dumps(
                                {
                                    "search_run_id": payload["search_run_id"],
                                    "search_result_id": selected_result["result_id"],
                                    "entry_parameters": selected_result["entry_parameters_json"],
                                    "exit_parameters": selected_result["exit_parameters_json"],
                                },
                                default=str,
                            ),
                        }
                    )
                    db.update_spy_strategy_search_result_promotion(str(selected_result["result_id"]), str(promote_payload["active_strategy_id"]))
                    st.success(f"Automated SPY search result promoted to forward paper trading: {promote_payload['active_strategy_id']}")

    state = st.session_state.get(SESSION_SPY_LAB_KEY)
    latest_bars = provider.get_stock_bars(
        symbol="SPY",
        start_date=str((pd.Timestamp(workbench.end_date) - pd.Timedelta(days=450)).date()),
        end_date=workbench.end_date,
        timeframe=workbench.timeframe,
        force_refresh=False,
    )
    latest_status = provider.get_last_fetch_status("SPY")
    latest_daily_context = None
    if is_intraday_timeframe(workbench.timeframe):
        latest_daily_context = provider.get_stock_bars(
            symbol="SPY",
            start_date=str((pd.Timestamp(workbench.end_date) - pd.Timedelta(days=450)).date()),
            end_date=workbench.end_date,
            timeframe="1d",
            force_refresh=False,
        )
    strategy_for_signal = apply_spy_exit_structure(build_spy_strategy(workbench.preset_key, workbench.entry_parameters), workbench)
    signal_bars = prepare_spy_timeframe_bars(primary_bars=latest_bars, timeframe=workbench.timeframe, daily_bars=latest_daily_context)
    active_spy = db.list_active_paper_strategies()
    active_spy = active_spy[active_spy["tickers"].fillna("").eq("SPY")] if not active_spy.empty else pd.DataFrame()
    selected_active = db.get_active_paper_strategy(str(active_spy.iloc[0]["active_strategy_id"])) if not active_spy.empty else None
    pending_orders = db.read_forward_paper_orders(str(selected_active["active_strategy_id"])) if selected_active else pd.DataFrame()
    open_positions = db.read_forward_paper_positions(str(selected_active["active_strategy_id"])) if selected_active else pd.DataFrame()
    signal_panel = spy_daily_signal_status(
        bars=signal_bars,
        strategy=strategy_for_signal,
        latest_close=float(signal_bars["close"].iloc[-1]) if not signal_bars.empty else 0.0,
        data_freshness_status=str(latest_status.cache_status if latest_status is not None else "unknown"),
        pending_orders=pending_orders,
        open_positions=open_positions,
    )

    st.subheader("SPY Signal Panel")
    signal_cols = st.columns(4)
    signal_cols[0].metric("Latest Signal", str(signal_panel["current_signal"]))
    signal_cols[1].metric("Position State", str(signal_panel["position_state"]))
    signal_cols[2].metric("Latest Close", f"{float(signal_panel['latest_close']):.2f}")
    signal_cols[3].metric("Next Expected Action", str(signal_panel["next_expected_action"]))
    st.write(
        {
            "timeframe": workbench.timeframe,
            "last_signal_date": signal_panel["last_signal_date"],
            "pending_forward_order": bool(signal_panel["pending_order"]),
            "open_forward_position": bool(signal_panel["open_position"]),
            "data_freshness_status": signal_panel["data_freshness_status"],
            "latest_data_timestamp": str(pd.to_datetime(signal_bars["timestamp"]).max()) if not signal_bars.empty else None,
            "bars_loaded": int(len(signal_bars)),
            "date_range_loaded": f"{pd.to_datetime(signal_bars['timestamp']).min()} to {pd.to_datetime(signal_bars['timestamp']).max()}" if not signal_bars.empty else None,
        }
    )
    if latest_status is not None and latest_status.validation_warnings:
        render_warning_list(latest_status.validation_warnings, "No timeframe-specific warnings.")
    if not active_spy.empty:
        st.dataframe(active_spy[["active_strategy_id", "status", "created_at", "current_paper_equity", "strategy_name", "notes", "tags"]], use_container_width=True)
        if st.button("Run Forward Paper Update", key="spy_workbench_forward_update"):
            engine = ForwardPaperEngine()
            for strategy_row in active_spy.itertuples():
                strategy_payload = db.get_active_paper_strategy(str(strategy_row.active_strategy_id))
                if strategy_payload is None or str(strategy_payload.get("status")) != "active":
                    continue
                forward_result = engine.run_update(active_strategy=strategy_payload, provider=provider)
                db.replace_forward_engine_events(forward_result.active_strategy_id, forward_result.events)
                if not forward_result.skipped:
                    db.replace_forward_paper_state(forward_result.active_strategy_id, forward_result.orders, forward_result.positions, forward_result.trades, forward_result.equity_curve)
                    strategy_payload["current_paper_equity"] = forward_result.current_equity
                    strategy_payload["updated_at"] = datetime.now(UTC).replace(tzinfo=None)
                    db.update_active_paper_strategy(strategy_payload)
            st.success("SPY forward paper update completed.")
        if selected_active is not None:
            active_id = str(selected_active["active_strategy_id"])
            forward_orders = db.read_forward_paper_orders(active_id)
            forward_positions = db.read_forward_paper_positions(active_id)
            forward_trades = db.read_forward_paper_trades(active_id)
            forward_equity = db.read_forward_paper_equity_curve(active_id)
            forward_events = db.read_active_paper_strategy_events(active_id)
            forward_cols = st.columns(5)
            forward_cols[0].metric("Pending Orders", int((forward_orders["status"] == "pending").sum()) if not forward_orders.empty else 0)
            forward_cols[1].metric("Open Positions", int((forward_positions["status"] == "open").sum()) if not forward_positions.empty else 0)
            forward_cols[2].metric("Closed Trades", len(forward_trades))
            forward_cols[3].metric("Current Paper Equity", f"{float(forward_equity['equity'].iloc[-1]):,.2f}" if not forward_equity.empty else "0.00")
            forward_cols[4].metric("Realized P&L", f"{float(forward_trades['realized_pnl'].sum() if not forward_trades.empty else 0.0):.2f}")
            if not forward_events.empty:
                st.dataframe(forward_events.tail(20), use_container_width=True)
    else:
        st.info("No active SPY forward paper strategy is running yet.")

    if not state:
        st.info("Run the workbench backtest to populate the unified results, exit comparison, robustness review, promotion flow, and trade review.")
        return

    result: BacktestResult = state["result"]
    research = state["research"]
    current_workbench = state["workbench"]
    benchmark_sharpe = 0.0
    if not result.benchmark_curve.empty:
        benchmark_equity = result.benchmark_curve.rename(columns={"benchmark_equity": "equity"})
        benchmark_sharpe = calculate_sharpe_ratio(benchmark_equity)
    summary = spy_strategy_summary(result.metrics, benchmark_sharpe=benchmark_sharpe)

    st.subheader("D. Backtest Results")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Strategy CAGR", f"{float(summary['Strategy CAGR']):.1%}")
    summary_cols[1].metric("SPY CAGR", f"{float(summary['Buy-and-Hold SPY CAGR']):.1%}")
    summary_cols[2].metric("Excess CAGR", f"{float(summary['Excess CAGR vs SPY']):.1%}")
    summary_cols[3].metric("Drawdown Improvement", f"{float(summary['Drawdown Improvement vs SPY']):.1%}")
    st.write(spy_summary_commentary(summary))
    st.dataframe(pd.DataFrame([{"metric": key, "value": value} for key, value in summary.items()]), use_container_width=True)
    st.plotly_chart(build_equity_chart(result.equity_curve, result.benchmark_curve), use_container_width=True)
    st.plotly_chart(build_drawdown_chart(result.equity_curve), use_container_width=True)
    st.dataframe(result.trade_log, use_container_width=True)

    st.subheader("E. Exit Comparison")
    exit_compare_choices = st.multiselect(
        "Exit structures to compare",
        [item.label for item in exit_structures],
        default=[exit_structure.label, "Signal exit only", "OCO bracket", "Trailing stop"],
        key="spy_workbench_exit_compare",
    )
    if st.button("Run Exit Comparison", key="spy_workbench_exit_run"):
        labels_to_keys = {item.label: item.key for item in exit_structures}
        comparison = run_spy_exit_comparison(
            engine=BacktestEngine(database=None),
            data_by_symbol=state["data_by_symbol"],
            workbench=current_workbench,
            exit_structure_keys=[labels_to_keys[label] for label in exit_compare_choices],
            benchmark_symbol="SPY",
        )
        st.session_state[SESSION_SPY_LAB_EXIT_KEY] = comparison
    exit_results = st.session_state.get(SESSION_SPY_LAB_EXIT_KEY, pd.DataFrame())
    if isinstance(exit_results, pd.DataFrame) and not exit_results.empty:
        st.dataframe(exit_results, use_container_width=True)
        for line in summarize_exit_comparison_results(exit_results):
            st.write(f"- {line}")

    st.subheader("F. Robustness Summary")
    if st.button("Run Robustness Review", key="spy_workbench_robustness"):
        sweep_engine = BacktestEngine(database=None)
        _, sweep_results, stability_summary = run_spy_parameter_stability(
            engine=sweep_engine,
            config=build_spy_backtest_config(current_workbench),
            data_by_symbol=state["data_by_symbol"],
            preset_key=current_workbench.preset_key,
            benchmark_symbol="SPY",
        )
        st.session_state[SESSION_SPY_LAB_STABILITY_KEY] = {"results": sweep_results, "summary": stability_summary}
        robustness_engine = BacktestEngine(database=None)
        robustness_payload = run_spy_robustness_checks(
            engine=robustness_engine,
            strategy=apply_spy_exit_structure(build_spy_strategy(current_workbench.preset_key, current_workbench.entry_parameters), current_workbench),
            config=build_spy_backtest_config(current_workbench),
            data_by_symbol=state["data_by_symbol"],
            benchmark_symbol="SPY",
            parameter_stability_summary_payload=stability_summary,
        )
        st.session_state[SESSION_SPY_LAB_ROBUSTNESS_KEY] = robustness_payload

    stability_state = st.session_state.get(SESSION_SPY_LAB_STABILITY_KEY)
    robustness_payload = st.session_state.get(SESSION_SPY_LAB_ROBUSTNESS_KEY)
    if robustness_payload:
        concentration = summarize_profit_concentration(result.trade_log)
        checklist, final_label = build_spy_robustness_checklist(metrics=result.metrics, concentration=concentration, robustness_payload=robustness_payload)
        st.dataframe(checklist, use_container_width=True)
        st.metric("SPY Candidate Label", final_label)
        if stability_state and stability_state.get("summary"):
            stability_summary = stability_state["summary"]
            st.write(
                {
                    "best_cagr": float(stability_summary.get("best_cagr", 0.0) or 0.0),
                    "median_cagr": float(stability_summary.get("median_cagr", 0.0) or 0.0),
                    "worst_cagr": float(stability_summary.get("worst_cagr", 0.0) or 0.0),
                    "percent_beating_spy": float(stability_summary.get("percent_beating_spy", 0.0) or 0.0),
                }
            )
        st.write("Train/Test", robustness_payload["train_test"]["degradation"])
        st.write("Walk-Forward", robustness_payload["walk_summary"])
        if not robustness_payload["slippage_results"].empty:
            st.dataframe(robustness_payload["slippage_results"], use_container_width=True)
        render_warning_list(robustness_payload["slippage_warnings"] + state["validation_warnings"] + state["research"]["corporate_action_warnings"], "No robustness warnings were generated.")

    st.subheader("G. Promote Candidate")
    st.caption("Promote only one strategy at a time. Freeze the current SPY entry-plus-exit configuration only after reviewing the comparison, drawdown, trade count, and robustness checks.")
    promote_notes = st.text_area("Promotion notes", value="", key="spy_workbench_promote_notes")
    promote_tags = st.text_input("Promotion tags", value="spy-only,spy-trading-workbench", key="spy_workbench_promote_tags")
    promotion_checks = pd.DataFrame(
        [
            {"check": "Compared to SPY", "passed": True},
            {"check": "Drawdown acceptable", "passed": float(summary["Drawdown Improvement vs SPY"]) >= -0.02},
            {"check": "Trade count sufficient", "passed": int(summary["Number of Trades"]) >= 10},
            {"check": "Robustness acceptable", "passed": bool(robustness_payload)},
            {"check": "Exit structure selected", "passed": True},
            {"check": "No major data warnings", "passed": not state["validation_warnings"]},
        ]
    )
    st.dataframe(promotion_checks, use_container_width=True, hide_index=True)
    promote_confirm = st.checkbox("Freeze this SPY Workbench configuration for forward paper trading.", key="spy_workbench_promote_confirm")
    if st.button("Promote To SPY Forward Paper Trading", key="spy_workbench_promote_button"):
        if not promote_confirm:
            st.error("Confirm the promotion checkbox before activating forward paper trading.")
        else:
            payload = build_active_paper_strategy_payload(
                strategy_name=current_workbench.entry_label,
                strategy_parameters={**current_workbench.entry_parameters, "__workbench_exit_structure__": current_workbench.exit_structure_key},
                universe_name="SPY Workbench",
                tickers=["SPY"],
                timeframe=current_workbench.timeframe,
                benchmark_symbol="SPY",
                price_mode=current_workbench.price_mode,
                initial_capital=current_workbench.initial_capital,
                position_sizing_method=current_workbench.position_sizing_method,
                position_sizing_value=current_workbench.position_size_value,
                max_positions=1,
                risk_settings={
                    **current_workbench.exit_parameters,
                    "fill_rule": "next_open",
                    "same_bar_stop_target_rule": "conservative_stop_first",
                    "exit_structure_key": current_workbench.exit_structure_key,
                    "end_of_day_exit": bool(current_workbench.entry_parameters.get("end_of_day_exit", False)),
                    "allow_overnight": bool(current_workbench.entry_parameters.get("allow_overnight", True)),
                },
                slippage_pct=current_workbench.slippage_pct,
                commission_per_trade=current_workbench.commission_per_trade,
                linked_backtest_run_id=result.run_id,
                activation_reason=f"Promoted from SPY Workbench: {preset.label} with {current_workbench.exit_structure_label}.",
                notes=promote_notes,
                tags=promote_tags,
                status="active",
            )
            db.insert_active_paper_strategy(payload)
            db.insert_active_paper_strategy_event(
                {
                    "event_id": str(uuid4()),
                    "active_strategy_id": payload["active_strategy_id"],
                    "created_at": datetime.now(UTC).replace(tzinfo=None),
                    "event_type": "activation",
                    "message": f"SPY-only forward paper strategy activated from {preset.label} using {current_workbench.exit_structure_label}.",
                    "details_json": json.dumps({"preset_key": preset.key, "entry_parameters": current_workbench.entry_parameters, "exit_structure": current_workbench.exit_structure_key, "exit_parameters": current_workbench.exit_parameters}, default=str),
                }
            )
            st.success(f"SPY workbench configuration promoted to forward paper trading: {payload['active_strategy_id']}")

    st.subheader("G. Trade Review")
    if not active_spy.empty and selected_active is not None:
        forward_trades = db.read_forward_paper_trades(str(selected_active["active_strategy_id"]))
        if forward_trades.empty:
            st.info("No closed SPY forward paper trades yet.")
        else:
            review_cols = st.columns(6)
            review_cols[0].metric("Closed Trades", len(forward_trades))
            review_cols[1].metric("Average R Multiple", f"{float(forward_trades['realized_r_multiple'].mean()):.2f}")
            review_cols[2].metric("Win Rate", f"{float((forward_trades['realized_pnl'] > 0).mean()):.1%}")
            gross_profit = float(forward_trades.loc[forward_trades["realized_pnl"] > 0, "realized_pnl"].sum())
            gross_loss = abs(float(forward_trades.loc[forward_trades["realized_pnl"] < 0, "realized_pnl"].sum()))
            review_cols[3].metric("Profit Factor", f"{(gross_profit / gross_loss) if gross_loss else 0.0:.2f}")
            review_cols[4].metric("Expectancy", f"{float(forward_trades['realized_pnl'].mean()):.2f}")
            review_cols[5].metric("Best Trade", f"{float(forward_trades['realized_pnl'].max()):.2f}")
            st.metric("Worst Trade", f"{float(forward_trades['realized_pnl'].min()):.2f}")
            st.dataframe(forward_trades, use_container_width=True)
            st.caption("Forward paper notes and lessons remain lightweight. Use the separate Paper Journal tab if you need richer manual journaling.")


def render_parameter_sweep(current_meta: dict[str, Any], current_data: dict[str, pd.DataFrame]) -> None:
    st.header("Parameter Sweep")
    if current_meta:
        render_saved_sweeps(current_meta["db"])
    else:
        render_saved_sweeps(TradingLabDatabase())
    st.divider()
    if not current_meta or not current_data:
        st.info("Run a backtest first so the sweep uses the current dataset and settings.")
        return
    strategy_name = current_meta["strategy_name"]
    engine = BacktestEngine(database=current_meta["db"])
    config: BacktestConfig = current_meta["config"]
    benchmark_symbol = current_meta["benchmark_symbol"]
    st.caption("Use sweeps to explore stability, not to cherry-pick the single best CAGR.")
    if strategy_name == "Moving Average Crossover":
        param_grid = {
            "fast_window": parse_range_input(st.text_input("Fast windows", "10,20,30", key="sweep_fast"), int),
            "slow_window": parse_range_input(st.text_input("Slow windows", "50,100,150", key="sweep_slow"), int),
        }
    elif strategy_name == "RSI Mean Reversion":
        param_grid = {
            "rsi_length": parse_range_input(st.text_input("RSI lengths", "10,14,20", key="sweep_rsi_len"), int),
            "buy_threshold": parse_range_input(st.text_input("Buy thresholds", "25,30,35", key="sweep_buy"), float),
        }
    elif strategy_name == "QQE/HMA Daily":
        st.warning("QQE/HMA has more tunable parameters than the starter strategies. Treat sweep winners as fragile until stability and out-of-sample behavior are confirmed.")
        param_grid = {
            "hma_length": parse_range_input(st.text_input("HMA lengths", "14,21,28", key="sweep_qqe_hma_len"), int),
            "rsi_length": parse_range_input(st.text_input("RSI lengths", "10,14,18", key="sweep_qqe_rsi_len"), int),
            "qqe_factor": parse_range_input(st.text_input("QQE factors", "3.0,4.236,5.0", key="sweep_qqe_factor"), float),
        }
    else:
        param_grid = {"lookback_window": parse_range_input(st.text_input("Lookback windows", "10,20,50", key="sweep_breakout"), int)}
    sweep_notes = st.text_area("Sweep notes", value=current_meta.get("notes", ""), key="sweep_notes", help="Use notes to record what hypothesis this sweep is testing.")
    sweep_tags = st.text_input("Sweep tags", value=current_meta.get("tags", ""), key="sweep_tags", help="Comma-separated tags for filtering saved sweeps.")
    drawdown_threshold = st.number_input("Max drawdown threshold", value=-0.25, format="%.4f", key="sweep_dd_threshold", help="Used in the stability summary.")
    sort_metric = st.selectbox("Sort sweep by", ["CAGR", "Sharpe Ratio", "Max Drawdown", "Calmar Ratio", "Total Return", "Profit Factor"], key="sweep_sort")
    if st.button("Run Parameter Sweep"):
        sweep_id, results = run_parameter_sweep(
            engine,
            lambda params: build_strategy(strategy_name, params),
            current_data,
            config,
            param_grid,
            benchmark_symbol,
            sort_metric=sort_metric,
            strategy_name=strategy_name,
            notes=sweep_notes,
            tags=sweep_tags,
            sweep_context={"sort_metric": sort_metric, "drawdown_threshold": drawdown_threshold},
        )
        stability = summarize_parameter_stability(results, drawdown_threshold=drawdown_threshold)
        st.session_state[SESSION_SWEEP_KEY] = {"sweep_id": sweep_id, "results": results, "stability": stability}
    sweep_state = st.session_state.get(SESSION_SWEEP_KEY)
    if not sweep_state:
        return
    results = sweep_state["results"]
    stability = sweep_state["stability"]
    st.write(f"Sweep ID: `{sweep_state['sweep_id']}`")
    st.warning("Do not trust the single top row. Prefer robust neighborhoods that still hold up out of sample.")
    st.dataframe(results, use_container_width=True)
    st.subheader("Parameter Stability Report")
    st.json(stability)
    if results.shape[0] > 1 and not results.empty and results["parameters_json"].apply(len).max() == 2:
        param_names = list(results["parameters_json"].iloc[0].keys())
        x_col, y_col = param_names[0], param_names[1]
        heatmap_data = results.copy()
        heatmap_data[x_col] = heatmap_data["parameters_json"].apply(lambda x: x.get(x_col))
        heatmap_data[y_col] = heatmap_data["parameters_json"].apply(lambda x: x.get(y_col))
        pivot = heatmap_data.pivot(index=y_col, columns=x_col, values=sort_metric)
        st.plotly_chart(px.imshow(pivot, aspect="auto", title=f"{sort_metric} Heatmap"), use_container_width=True)


def render_train_test_section(current_meta: dict[str, Any], current_data: dict[str, pd.DataFrame]) -> None:
    st.header("Train/Test")
    if not current_meta or not current_data:
        st.info("Run a backtest first so train/test uses the current dataset and settings.")
        return
    split_method = st.selectbox("Split method", ["split_date", "split_percentage"], key="train_test_method")
    if split_method == "split_date":
        split_date = st.date_input("Split date", value=pd.Timestamp("2022-01-01"), key="train_test_date", help="All data before this date is train, on or after it is test.")
        train_data, test_data = split_data_by_date(current_data, str(split_date))
        split_value = str(split_date)
    else:
        split_pct = st.slider("Train percentage", min_value=0.5, max_value=0.9, value=0.7, step=0.05, key="train_test_pct")
        train_data, test_data = split_data_by_percentage(current_data, split_pct)
        split_value = str(split_pct)
    if st.button("Run Train/Test Analysis"):
        engine = BacktestEngine(database=None)
        analysis = run_train_test_analysis(engine, current_meta["strategy"], current_meta["config"], train_data, test_data, current_meta["benchmark_symbol"])
        st.session_state[SESSION_TRAIN_TEST_KEY] = analysis
        current_meta["db"].replace_train_test_summary(
            current_meta["result"].run_id,
            {
                "run_id": current_meta["result"].run_id,
                "split_method": split_method,
                "split_value": split_value,
                "train_metrics_json": json.dumps(analysis["train_metrics"]),
                "test_metrics_json": json.dumps(analysis["test_metrics"]),
                "degradation_json": json.dumps(analysis["degradation"]),
                "created_at": datetime.now(UTC).replace(tzinfo=None),
            },
        )
    analysis = st.session_state.get(SESSION_TRAIN_TEST_KEY)
    if not analysis:
        return
    cols = st.columns(2)
    cols[0].subheader("Train Metrics")
    cols[0].json(analysis["train_metrics"])
    cols[1].subheader("Test Metrics")
    cols[1].json(analysis["test_metrics"])
    st.subheader("Degradation")
    st.json(analysis["degradation"])
    if analysis["test_metrics"]["CAGR"] < analysis["train_metrics"]["CAGR"] * 0.5:
        st.warning("Test-period performance degraded materially versus train. This raises overfit risk.")
    st.subheader("Benchmark Diagnostics")
    render_benchmark_diagnostics(current_meta["db"].read_benchmark_diagnostics(current_meta["result"].run_id))


def render_walk_forward_section(current_meta: dict[str, Any], current_data: dict[str, pd.DataFrame]) -> None:
    st.header("Walk-Forward")
    if not current_meta or not current_data:
        st.info("Run a backtest first so walk-forward uses the current dataset and settings.")
        return
    train_months = st.number_input("Train window months", min_value=3, value=24, step=1, help="Months used for each train window.")
    test_months = st.number_input("Test window months", min_value=1, value=6, step=1, help="Months used for each test window.")
    step_months = st.number_input("Step size months", min_value=1, value=3, step=1, help="How far each fold advances.")
    min_train_trades = st.number_input("Minimum train trades", min_value=0, value=0, step=1)
    min_test_trades = st.number_input("Minimum test trades", min_value=0, value=0, step=1)
    if st.button("Run Walk-Forward"):
        engine = BacktestEngine(database=None)
        walk_id, folds, summary = run_walk_forward_analysis(
            engine,
            current_meta["strategy"],
            current_meta["config"],
            current_data,
            current_meta["benchmark_symbol"],
            int(train_months),
            int(test_months),
            int(step_months),
            int(min_train_trades),
            int(min_test_trades),
        )
        st.session_state[SESSION_WALK_FORWARD_KEY] = {"walk_forward_id": walk_id, "folds": folds, "summary": summary}
        current_meta["db"].replace_walk_forward_run(
            {
                "walk_forward_id": walk_id,
                "run_id": current_meta["result"].run_id,
                "benchmark_symbol": current_meta["benchmark_symbol"],
                "train_window_months": int(train_months),
                "test_window_months": int(test_months),
                "step_months": int(step_months),
                "min_train_trades": int(min_train_trades),
                "min_test_trades": int(min_test_trades),
                "summary_json": json.dumps(summary),
                "created_at": datetime.now(UTC).replace(tzinfo=None),
            },
            folds,
        )
    walk_state = st.session_state.get(SESSION_WALK_FORWARD_KEY)
    if not walk_state:
        return
    folds = walk_state["folds"]
    summary = walk_state["summary"]
    if folds.empty:
        st.warning("No valid walk-forward folds were produced for the selected settings.")
        return
    st.subheader("Walk-Forward Summary")
    st.json(summary)
    st.subheader("Fold Results")
    st.dataframe(folds, use_container_width=True)
    st.plotly_chart(px.bar(folds, x="fold_number", y="test_cagr", title="Test Fold CAGR Over Time"), use_container_width=True)
    cagr_long = folds.melt(id_vars=["fold_number"], value_vars=["train_cagr", "test_cagr"], var_name="series", value_name="cagr")
    dd_long = folds.melt(id_vars=["fold_number"], value_vars=["train_max_drawdown", "test_max_drawdown"], var_name="series", value_name="drawdown")
    st.plotly_chart(px.line(cagr_long, x="fold_number", y="cagr", color="series", markers=True, title="Train vs Test CAGR by Fold"), use_container_width=True)
    st.plotly_chart(px.line(dd_long, x="fold_number", y="drawdown", color="series", markers=True, title="Train vs Test Drawdown by Fold"), use_container_width=True)
    st.subheader("Benchmark Diagnostics")
    render_benchmark_diagnostics(current_meta["db"].read_benchmark_diagnostics(current_meta["result"].run_id))
    st.download_button("Export walk-forward folds", data=folds.to_csv(index=False).encode("utf-8"), file_name="walk_forward_folds.csv", mime="text/csv")


def render_regime_section(current_research: dict[str, Any] | None, current_meta: dict[str, Any]) -> None:
    st.header("Regime Analysis")
    if not current_research:
        st.info("Run a backtest first to generate regime analysis.")
        return
    regime_metrics = current_research["regime_metrics"]
    if regime_metrics.empty:
        st.info("Regime analysis is not available for the current run.")
        return
    st.dataframe(regime_metrics, use_container_width=True)
    st.write("Regime Comments")
    for comment in current_research["regime_comments"]:
        st.write(f"- {comment}")
    st.subheader("Benchmark Diagnostics")
    render_benchmark_diagnostics(current_meta["db"].read_benchmark_diagnostics(current_meta["result"].run_id))
    st.download_button("Export regime metrics", data=regime_metrics.to_csv(index=False).encode("utf-8"), file_name="regime_metrics.csv", mime="text/csv")


def render_data_health(statuses: list[CacheStatus], warnings: list[str], current_research: dict[str, Any] | None) -> None:
    st.header("Data Health")
    st.subheader("Freshness")
    render_freshness_status(statuses)
    st.subheader("Validation Warnings")
    render_warning_list(warnings, "No validation warnings were captured for the current load.")
    st.subheader("Benchmark Diagnostics")
    if current_research:
        render_benchmark_diagnostics(current_research["benchmark_diagnostics"])
    else:
        st.info("Run a backtest to evaluate benchmark diagnostics.")
    st.subheader("Corporate-Action Warnings")
    render_warning_list(current_research["corporate_action_warnings"] if current_research else [], "No corporate-action warnings were generated for the current run.")


def render_indicator_preview(provider: YFinanceDataProvider) -> None:
    st.header("Indicators")
    st.caption("Research-only indicator preview. This does not place trades and does not imply a production-ready strategy.")
    symbol = st.text_input("Indicator ticker", value="AAPL", key="indicator_symbol").upper().strip()
    indicator_start = st.date_input("Indicator start date", value=pd.Timestamp("2022-01-01"), key="indicator_start")
    indicator_end = st.date_input("Indicator end date", value=pd.Timestamp.today().normalize(), key="indicator_end")
    indicator_price_mode = st.selectbox("Indicator price mode", ["raw_price_mode", "adjusted_price_mode"], key="indicator_price_mode")
    indicator_name = st.selectbox("Indicator", ["HMA", "RSI", "QQE"], key="indicator_name")

    indicator_params: dict[str, Any]
    if indicator_name == "HMA":
        indicator_params = {"length": st.number_input("HMA length", min_value=2, value=21, key="indicator_hma_length")}
    elif indicator_name == "RSI":
        indicator_params = {"length": st.number_input("RSI length", min_value=2, value=14, key="indicator_rsi_length")}
    else:
        indicator_params = {
            "rsi_length": st.number_input("QQE RSI length", min_value=2, value=14, key="indicator_qqe_rsi_length"),
            "rsi_smoothing": st.number_input("QQE RSI smoothing", min_value=1, value=5, key="indicator_qqe_rsi_smoothing"),
            "qqe_factor": st.number_input("QQE factor", min_value=0.1, value=4.236, step=0.1, key="indicator_qqe_factor"),
            "atr_smoothing": st.number_input("QQE ATR smoothing", min_value=1, value=5, key="indicator_qqe_atr_smoothing"),
        }

    if not st.button("Preview Indicator", key="indicator_preview_button"):
        st.info("Choose a ticker and indicator to preview the calculation.")
        return

    with st.spinner("Loading data for indicator preview..."):
        bars = provider.get_stock_bars(symbol, str(indicator_start), str(indicator_end), timeframe="1d", force_refresh=False)
    preview = build_indicator_preview_frame(bars, indicator_name, price_mode=indicator_price_mode, indicator_params=indicator_params)

    if preview.empty:
        st.warning("No bars were available for the selected preview range.")
        return

    st.subheader("Price")
    st.plotly_chart(px.line(preview, x="timestamp", y="display_price", title=f"{symbol} Price"), use_container_width=True)

    if indicator_name == "HMA":
        if preview["hma"].notna().sum() == 0:
            st.warning("There is not enough data to compute HMA for the selected length.")
            return
        st.plotly_chart(px.line(preview, x="timestamp", y=["display_price", "hma"], title=f"{symbol} with HMA"), use_container_width=True)
        st.dataframe(preview[["timestamp", "display_price", "hma"]].tail(20), use_container_width=True)
    elif indicator_name == "RSI":
        if preview["rsi"].notna().sum() == 0:
            st.warning("There is not enough data to compute RSI for the selected length.")
            return
        st.plotly_chart(px.line(preview, x="timestamp", y="rsi", title=f"{symbol} RSI"), use_container_width=True)
        st.dataframe(preview[["timestamp", "display_price", "rsi"]].tail(20), use_container_width=True)
    else:
        qqe_columns = ["rsi", "rsi_smoothed", "qqe_slow", "trend", "signal"]
        valid_rows = preview[qqe_columns].dropna(how="all")
        if valid_rows.empty:
            st.warning("There is not enough data to compute QQE for the selected parameters.")
            return
        st.plotly_chart(px.line(preview, x="timestamp", y=["qqe_fast", "qqe_slow"], title=f"{symbol} QQE"), use_container_width=True)
        st.dataframe(preview[["timestamp", "display_price", *qqe_columns]].tail(20), use_container_width=True)


def filter_research_dashboard_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    strategy_filter = st.selectbox("Strategy filter", ["All"] + sorted(frame["strategy_name"].dropna().unique().tolist()), key="dashboard_strategy")
    ticker_filter = st.text_input("Ticker/universe contains", value="", key="dashboard_ticker")
    benchmark_filter = st.selectbox("Benchmark filter", ["All"] + sorted(frame["benchmark_symbol"].dropna().unique().tolist()), key="dashboard_benchmark")
    min_trades = st.number_input("Minimum trades", min_value=0, value=0, step=1, key="dashboard_min_trades")
    min_robustness = st.number_input("Minimum robustness score", min_value=0, max_value=100, value=0, step=5, key="dashboard_min_robustness")
    tag_filter = st.text_input("Tag contains", value="", key="dashboard_tag")

    filtered = frame.copy()
    if strategy_filter != "All":
        filtered = filtered[filtered["strategy_name"] == strategy_filter]
    if ticker_filter:
        filtered = filtered[filtered["symbols_csv"].fillna("").str.contains(ticker_filter, case=False)]
    if benchmark_filter != "All":
        filtered = filtered[filtered["benchmark_symbol"] == benchmark_filter]
    if tag_filter:
        filtered = filtered[filtered["tags"].fillna("").str.contains(tag_filter, case=False)]
    filtered = filtered[(filtered["number_of_trades"].fillna(0) >= min_trades) & (filtered["robustness_score"].fillna(0) >= min_robustness)]
    return filtered


def render_research_dashboard(db: TradingLabDatabase) -> None:
    st.header("Research Dashboard")
    dashboard = db.get_research_dashboard_rows()
    if dashboard.empty:
        st.info("No saved research results are available yet.")
        return
    filtered = filter_research_dashboard_rows(dashboard)
    if filtered.empty:
        st.info("No saved runs match the current dashboard filters.")
        return

    def best_row(column: str, ascending: bool = False) -> pd.Series | None:
        subset = filtered.dropna(subset=[column])
        if subset.empty:
            return None
        return subset.sort_values(column, ascending=ascending).iloc[0]

    most_robust = best_row("robustness_score")
    best_relative = best_row("excess_cagr")
    suspicious = filtered.sort_values(["total_return", "robustness_score"], ascending=[False, True]).iloc[0]
    low_drawdown = filtered.sort_values("max_drawdown", ascending=False).iloc[0]
    options_candidate_pool = filtered[filtered["options_candidate_flag"] == 1]
    options_candidate = options_candidate_pool.sort_values(["robustness_score", "excess_cagr"], ascending=[False, False]).iloc[0] if not options_candidate_pool.empty else most_robust

    cards = st.columns(5)
    cards[0].metric("Most Robust Saved Strategy", f"{most_robust['strategy_name']} ({most_robust['run_id']})" if most_robust is not None else "N/A")
    cards[1].metric("Best Benchmark-Relative", f"{best_relative['strategy_name']} ({best_relative['run_id']})" if best_relative is not None else "N/A")
    cards[2].metric("Most Suspicious High Return", f"{suspicious['strategy_name']} ({suspicious['run_id']})")
    cards[3].metric("Best Low-Drawdown Strategy", f"{low_drawdown['strategy_name']} ({low_drawdown['run_id']})")
    cards[4].metric("Best Options-Overlay Candidate", f"{options_candidate['strategy_name']} ({options_candidate['run_id']})" if options_candidate is not None else "N/A")

    flags = st.columns(6)
    flags[0].metric("High Profit Concentration", int(filtered["high_profit_concentration"].sum()))
    flags[1].metric("Poor Train/Test", int(filtered["poor_train_test_flag"].sum()))
    flags[2].metric("Poor Walk-Forward", int(filtered["poor_walk_forward_flag"].sum()))
    flags[3].metric("Regime Dependence", int(filtered["regime_dependence_flag"].sum()))
    flags[4].metric("Too Few Trades", int(filtered["too_few_trades_flag"].sum()))
    flags[5].metric("Underperformed Benchmark", int(filtered["underperformed_benchmark_flag"].sum()))

    qualification_runs = db.list_strategy_qualification_runs(limit=100)
    if not qualification_runs.empty:
        latest_qualification = qualification_runs.iloc[0]
        latest_results = db.read_strategy_qualification_results(str(latest_qualification["qualification_id"]))
        if not latest_results.empty:
            best_by_universe = latest_results.sort_values(["robustness_score", "excess_cagr"], ascending=[False, False]).iloc[0]
            consistency_candidate = filtered.sort_values(["walk_forward_consistency", "robustness_score"], ascending=[False, False]).iloc[0]
            qqe_rows = latest_results[latest_results["strategy_name"] == "QQE/HMA Daily"]
            simple_rows = latest_results[latest_results["strategy_name"].isin(["Moving Average Crossover", "RSI Mean Reversion", "Daily Breakout"])]
            st.subheader("Qualification Highlights")
            detail_cols = st.columns(4)
            detail_cols[0].metric("Best Strategy By Universe", f"{best_by_universe['strategy_name']} ({latest_qualification['universe_name']})")
            detail_cols[1].metric("Best Walk-Forward Consistency", f"{consistency_candidate['strategy_name']} ({consistency_candidate['run_id']})")
            detail_cols[2].metric("Candidate Flags", int(latest_results["options_candidate_flag"].sum()))
            detail_cols[3].metric("Latest Qualification Universe", str(latest_qualification["universe_name"]))
            if not qqe_rows.empty and not simple_rows.empty:
                qqe_row = qqe_rows.iloc[0]
                best_simple = simple_rows.sort_values("cagr", ascending=False).iloc[0]
                if float(qqe_row["cagr"]) > float(best_simple["cagr"]):
                    st.success(f"QQE/HMA beat the strongest simple strategy in the latest qualification run: {qqe_row['cagr']:.1%} vs {best_simple['cagr']:.1%}.")
                else:
                    st.warning(f"QQE/HMA did not beat the strongest simple strategy in the latest qualification run: {qqe_row['cagr']:.1%} vs {best_simple['cagr']:.1%}.")

    st.dataframe(filtered.sort_values(["robustness_score", "cagr"], ascending=[False, False]), use_container_width=True)
    st.download_button("Export research dashboard CSV", data=filtered.to_csv(index=False).encode("utf-8"), file_name="research_dashboard.csv", mime="text/csv")


def render_saved_spy_searches(
    db: TradingLabDatabase,
    *,
    strategy_filter: str = "All",
    candidate_label_filter: str = "All",
    promoted_only: bool = False,
    tag_filter: str = "",
) -> None:
    """Render saved automated SPY search runs and their stored result rows."""
    st.subheader("A. Saved SPY Searches")
    search_runs = db.list_spy_strategy_search_runs(limit=100, tag=tag_filter or None)
    if search_runs.empty:
        st.info("No automated SPY search runs are saved yet.")
        return
    st.dataframe(search_runs, use_container_width=True)
    selected_run_id = st.selectbox("Select saved SPY search run", search_runs["search_run_id"].tolist(), key="history_spy_search_run")
    results = db.read_spy_strategy_search_results(str(selected_run_id))
    if results.empty:
        st.info("The selected SPY search run has no stored results.")
        return
    filtered = results.copy()
    if strategy_filter != "All":
        filtered = filtered[filtered["entry_strategy_name"] == strategy_filter]
    if candidate_label_filter != "All":
        filtered = filtered[filtered["candidate_label"] == candidate_label_filter]
    if promoted_only:
        filtered = filtered[filtered["promoted_active_strategy_id"].notna()]
    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "Export saved SPY search CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"saved_spy_search_{selected_run_id}.csv",
        mime="text/csv",
        key="history_export_spy_search",
    )


def render_forward_paper_workspace(
    *,
    db: TradingLabDatabase,
    provider: YFinanceDataProvider,
    benchmark_symbol: str,
    refresh_data: bool,
    base_config: BacktestConfig,
    default_tickers: list[str],
    current_strategy_name: str,
    current_strategy_params: dict[str, Any],
    show_advanced_tools: bool,
) -> None:
    """Render the simplified forward-paper workspace, with legacy controls hidden behind expanders."""
    st.header("Forward Paper")
    st.caption("Forward paper trading uses future data only as it becomes available. Promote only one SPY strategy at a time.")
    active_strategies = db.list_active_paper_strategies()
    active_spy = active_strategies[active_strategies["tickers"].fillna("").eq("SPY")] if not active_strategies.empty else pd.DataFrame()
    if active_spy.empty:
        st.info("No promoted SPY forward-paper strategy is active yet. Start in SPY Workbench, review candidates, and promote one strategy.")
    else:
        selected_active_id = st.selectbox("Active SPY strategy", active_spy["active_strategy_id"].tolist(), key="workspace_forward_active_id")
        selected_active = db.get_active_paper_strategy(str(selected_active_id))
        if selected_active is not None:
            summary_cols = st.columns(5)
            summary_cols[0].metric("Status", str(selected_active.get("status", "")))
            summary_cols[1].metric("Activation Date", str(pd.Timestamp(selected_active["created_at"]).date()))
            summary_cols[2].metric("Current Paper Equity", f"{float(selected_active.get('current_paper_equity', 0.0) or 0.0):,.2f}")
            summary_cols[3].metric("Ticker", str(selected_active.get("tickers", "")))
            summary_cols[4].metric("Benchmark", str(selected_active.get("benchmark_symbol", "")))
            if st.button("Run Forward Paper Update", key="workspace_forward_update", type="primary"):
                engine = ForwardPaperEngine()
                updated: list[str] = []
                for strategy_row in active_spy.itertuples():
                    if str(strategy_row.status) != "active":
                        continue
                    strategy_payload = db.get_active_paper_strategy(str(strategy_row.active_strategy_id))
                    if strategy_payload is None:
                        continue
                    result = engine.run_update(active_strategy=strategy_payload, provider=provider)
                    db.replace_forward_engine_events(result.active_strategy_id, result.events)
                    if not result.skipped:
                        db.replace_forward_paper_state(result.active_strategy_id, result.orders, result.positions, result.trades, result.equity_curve)
                        strategy_payload["current_paper_equity"] = result.current_equity
                        strategy_payload["updated_at"] = datetime.now(UTC).replace(tzinfo=None)
                        db.update_active_paper_strategy(strategy_payload)
                    updated.append(result.active_strategy_id)
                st.success(f"Forward paper update completed for {len(updated)} SPY strategy record(s).")

            orders = db.read_forward_paper_orders(str(selected_active_id))
            positions = db.read_forward_paper_positions(str(selected_active_id))
            trades = db.read_forward_paper_trades(str(selected_active_id))
            equity_curve = db.read_forward_paper_equity_curve(str(selected_active_id))
            events = db.read_active_paper_strategy_events(str(selected_active_id))

            st.subheader("A. Active SPY Strategy")
            st.json(selected_active, expanded=False)
            st.subheader("B. Pending Orders")
            if orders.empty:
                st.info("No pending forward-paper orders.")
            else:
                st.dataframe(orders[orders["status"] == "pending"], use_container_width=True)
            st.subheader("C. Open Positions")
            if positions.empty:
                st.info("No open forward-paper positions.")
            else:
                st.dataframe(positions[positions["status"] == "open"], use_container_width=True)
            st.subheader("D. Closed Trades")
            if trades.empty:
                st.info("No closed forward-paper trades yet.")
            else:
                st.dataframe(trades, use_container_width=True)
            st.subheader("E. Forward Equity Curve")
            if equity_curve.empty:
                st.info("Run a forward update after new daily data exists to build the forward equity curve.")
            else:
                st.plotly_chart(build_equity_chart(equity_curve, pd.DataFrame()), use_container_width=True)
                st.plotly_chart(build_drawdown_chart(equity_curve), use_container_width=True)
            st.subheader("F. Event Log")
            if events.empty:
                st.info("No forward-paper events were recorded yet.")
            else:
                st.dataframe(events.tail(50), use_container_width=True)
            st.subheader("G. Forward vs Backtest Validation")
            if selected_active.get("linked_backtest_run_id") and not equity_curve.empty:
                metric_trades = trades.rename(columns={"realized_pnl": "pnl", "realized_return_pct": "return_pct"}).copy() if not trades.empty else pd.DataFrame(columns=["pnl", "return_pct", "entry_date", "exit_date"])
                if not metric_trades.empty:
                    metric_trades["holding_days"] = (pd.to_datetime(metric_trades["exit_date"]) - pd.to_datetime(metric_trades["entry_date"])).dt.days.clip(lower=0)
                forward_metrics = compute_summary_metrics(equity_curve, metric_trades, float(selected_active.get("initial_capital", 0.0) or 0.0))
                linked_backtest_run = db.get_backtest_run(str(selected_active["linked_backtest_run_id"]))
                warnings = compare_forward_to_backtest(
                    backtest_run=linked_backtest_run,
                    forward_metrics=forward_metrics,
                    days_since_activation=int((pd.Timestamp.today().normalize() - pd.Timestamp(selected_active["created_at"]).normalize()).days),
                )
                render_warning_list(warnings, "No forward-versus-backtest warnings were generated.")
            else:
                st.info("Forward validation will appear once the promoted strategy has linked backtest context and forward equity history.")
    if show_advanced_tools:
        with st.expander("Advanced Daily Dashboard", expanded=False):
            render_daily_trading_dashboard(db)
        with st.expander("H. Manual Paper Journal", expanded=False):
            render_paper_trade_journal(db)
        with st.expander("Advanced Forward Paper Tools", expanded=False):
            render_forward_paper_trading(
                db=db,
                provider=provider,
                benchmark_symbol=benchmark_symbol,
                refresh_data=refresh_data,
                base_config=base_config,
                default_tickers=default_tickers,
                current_strategy_name=current_strategy_name,
                current_strategy_params=current_strategy_params,
            )


def render_research_history_workspace(
    *,
    db: TradingLabDatabase,
    current_meta: dict[str, Any],
    current_data: dict[str, pd.DataFrame],
    current_research: dict[str, Any] | None,
    show_advanced_tools: bool,
) -> None:
    """Render saved research and historical review tools under one tab."""
    st.header("Research History")
    st.caption("Use this tab to review prior work. Start new SPY experiments in SPY Workbench.")
    filter_cols = st.columns(5)
    strategy_filter = filter_cols[0].selectbox("Strategy", ["All", "SPY 200-Day Trend Filter", "Moving Average Crossover", "RSI Mean Reversion", "Daily Breakout", "QQE/HMA Daily"], key="history_strategy_filter")
    candidate_label_filter = filter_cols[1].selectbox("Candidate label", ["All", "Strong candidate", "Possible candidate", "Not ready", "Reject"], key="history_candidate_filter")
    promoted_only = filter_cols[2].checkbox("Promoted only", value=False, key="history_promoted_only")
    tag_filter = filter_cols[3].text_input("Tag contains", value="", key="history_tag_filter")
    filter_cols[4].write("Review candidates and saved runs here. New experiments should start in SPY Workbench.")

    render_saved_spy_searches(
        db,
        strategy_filter=strategy_filter,
        candidate_label_filter=candidate_label_filter,
        promoted_only=promoted_only,
        tag_filter=tag_filter,
    )
    st.subheader("B. Saved SPY Sessions")
    st.info("Dedicated SPY session objects are not persisted yet. Use saved SPY searches, saved backtests, and active forward-paper strategies as the current session trail.")
    with st.expander("C. Saved Backtests", expanded=False):
        render_saved_backtests(db, key_prefix="history_saved_backtests")
    with st.expander("D. Saved Sweeps", expanded=False):
        render_saved_sweeps(db, key_prefix="history_saved_sweeps")
    with st.expander("E. Strategy Qualification Results", expanded=False):
        render_saved_qualification_runs(db, key_prefix="history_saved_qualifications")
    with st.expander("F. Walk-Forward Results", expanded=False):
        st.info("Walk-forward summaries are attached to saved runs. Open a saved backtest to inspect linked walk-forward results.")
    with st.expander("G. Train/Test Results", expanded=False):
        st.info("Train/test summaries are attached to saved runs. Open a saved backtest to inspect linked train/test degradation.")
    with st.expander("H. Exports", expanded=False):
        runs = db.list_backtest_runs(limit=500)
        if runs.empty:
            st.info("No saved backtests are available to export yet.")
        else:
            st.download_button("Export saved backtests CSV", data=runs.to_csv(index=False).encode("utf-8"), file_name="saved_backtests.csv", mime="text/csv", key="history_export_backtests")
    if show_advanced_tools:
        with st.expander("Advanced Research Dashboard", expanded=False):
            render_research_dashboard(db)
        with st.expander("Advanced Compare Backtests", expanded=False):
            render_compare_backtests(db, key_prefix="history_compare_backtests")
        with st.expander("Advanced Scanner History", expanded=False):
            render_scanner_history(db)
        with st.expander("Legacy Multi-Ticker Research", expanded=False):
            render_parameter_sweep(current_meta, current_data)
            render_train_test_section(current_meta, current_data)
            render_walk_forward_section(current_meta, current_data)
            render_regime_section(current_research, current_meta)


def render_data_settings_workspace(
    *,
    db: TradingLabDatabase,
    provider: YFinanceDataProvider,
    statuses: list[CacheStatus],
    warnings: list[str],
    current_research: dict[str, Any] | None,
    start_date: str,
    end_date: str,
    refresh_data: bool,
    settings_snapshot: dict[str, Any],
    show_advanced_tools: bool,
) -> None:
    """Render data refresh, diagnostics, and app-level settings."""
    st.header("Data & Settings")
    st.caption("Refresh data, inspect cache health, and review diagnostics here.")
    st.subheader("A. Data Refresh")
    st.write("Start here if SPY data looks stale before rerunning search or forward paper updates.")
    if st.button("Refresh SPY Daily Data", key="data_settings_refresh_spy"):
        provider.get_stock_bars(symbol="SPY", start_date=start_date, end_date=end_date, timeframe="1d", force_refresh=True)
        st.success("Requested a forced refresh for SPY daily bars.")
    latest_status = provider.get_last_fetch_status("SPY")
    freshness_rows = statuses[:] if statuses else ([latest_status] if latest_status is not None else [])
    st.subheader("B. Cache Status")
    render_freshness_status([status for status in freshness_rows if status is not None])
    st.subheader("C. Data Quality Warnings")
    render_warning_list(warnings, "No data-quality warnings are currently loaded in session.")
    st.subheader("D. Benchmark Diagnostics")
    render_benchmark_diagnostics(current_research["benchmark_diagnostics"] if current_research else None)
    st.subheader("E. Corporate Action Warnings")
    render_warning_list(current_research["corporate_action_warnings"] if current_research else [], "No corporate-action warnings are currently loaded in session.")
    st.subheader("F. App Settings")
    st.json(settings_snapshot, expanded=False)
    st.subheader("G. Database Backup / Export")
    st.write({"db_path": db.db_path})
    db_path = Path(db.db_path)
    if db_path.exists():
        try:
            db_bytes = db_path.read_bytes()
        except OSError as exc:
            st.warning(f"Database backup is temporarily unavailable because Windows is locking `{db_path.name}`: {exc}")
        else:
            st.download_button("Backup Database", data=db_bytes, file_name=db_path.name, mime="application/octet-stream", key="data_settings_backup_db")
    else:
        st.info("The DuckDB database file does not exist yet.")
    if show_advanced_tools:
        with st.expander("H. Developer Diagnostics", expanded=False):
            st.write({"session_keys": sorted(st.session_state.keys())})
            render_indicator_preview(provider)
            render_data_health(statuses, warnings, current_research)


def _format_report_percent(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Not available"
    return f"{float(value):.2%}"


def _format_report_number(value: Any, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Not available"
    return f"{float(value):,.{decimals}f}"


def _caption_source(label: str, detail: str) -> str:
    return f"Source: {label} | {detail}"


def render_market_regime_workspace(
    *,
    provider: YFinanceDataProvider,
    start_date: str,
    end_date: str,
    refresh_data: bool,
) -> None:
    st.header("Market Regime Report")
    st.caption("Research-only market context dashboard. This tab does not place trades or change the backtesting workflow.")
    refresh_report = st.button("Refresh Market Regime Report", key="market_regime_refresh")
    report = build_market_regime_report(
        provider,
        start_date=start_date,
        end_date=end_date,
        refresh_data=refresh_data or refresh_report,
        as_of_date=pd.Timestamp(end_date).date(),
    )

    if report.warnings:
        for warning in report.warnings:
            st.warning(warning)
    if any(
        section.source.is_demo
        for section in [
            report.index_momentum,
            report.breadth,
            report.options_positioning,
            report.sector_leadership,
            report.seasonality,
            report.macro_calendar,
            report.earnings_watch,
        ]
    ):
        st.info("One or more sections are using demo placeholder data. The labels below show which sections are live, cached, or demo-backed.")

    summary_cols = st.columns(4)
    summary_cols[0].metric("Regime", report.summary.regime_label)
    summary_cols[1].metric("Trend", report.summary.trend_direction)
    summary_cols[2].metric("Posture", report.summary.recommended_posture)
    summary_cols[3].metric("Score", report.summary.total_score)
    st.caption(f"Last updated: {report.generated_at}")
    st.write(report.summary.short_summary)

    with st.container(border=True):
        st.subheader("Analyst Summary")
        st.write(report.analyst_summary)

    score_frame = pd.DataFrame(
        [
            {
                "component": component.name,
                "score": component.score,
                "active": component.active,
                "rationale": component.rationale,
            }
            for component in report.summary.components
        ]
    )
    with st.container(border=True):
        st.subheader("Scoring Model")
        st.caption("Auditable first-pass score from -2 to +2 by component. Options positioning only contributes when real data is connected.")
        st.dataframe(score_frame, use_container_width=True, hide_index=True)

    index_frame = pd.DataFrame(report.index_momentum.rows)
    with st.container(border=True):
        st.subheader("Index Momentum")
        st.caption(_caption_source(report.index_momentum.source.label, report.index_momentum.source.detail))
        if index_frame.empty:
            st.info("Index momentum data is not available.")
        else:
            display = index_frame.copy()
            for column in ["return_5d", "return_20d", "distance_50dma", "distance_200dma", "realized_vol_20d"]:
                display[column] = display[column].map(_format_report_percent)
            display["rsi_14"] = display["rsi_14"].map(lambda value: _format_report_number(value, 1))
            display["overextended_flag"] = display["overextended_flag"].map(lambda value: "Yes" if value else "")
            st.dataframe(display, use_container_width=True, hide_index=True)
            normalized = []
            for row in report.index_momentum.rows:
                symbol = str(row["symbol"])
                frame = provider.database.read_stock_bars(symbol=symbol, start_date=start_date, end_date=end_date, timeframe="1d")
                if frame.empty:
                    continue
                close_series = indicator_price_series(frame.sort_values("timestamp"), "adjusted_price_mode")
                if close_series.empty or float(close_series.iloc[0]) == 0.0:
                    continue
                normalized.append(pd.DataFrame({"timestamp": pd.to_datetime(frame["timestamp"]), "symbol": symbol, "normalized": close_series / float(close_series.iloc[0])}))
            if normalized:
                chart_frame = pd.concat(normalized, ignore_index=True)
                st.plotly_chart(px.line(chart_frame, x="timestamp", y="normalized", color="symbol", title="Normalized Performance"), use_container_width=True)

    breadth_frame = pd.DataFrame(report.breadth.rows)
    with st.container(border=True):
        st.subheader("Breadth")
        st.caption(_caption_source(report.breadth.source.label, report.breadth.source.detail))
        if not breadth_frame.empty:
            display = breadth_frame.copy()
            for column in ["pct_above_20dma", "pct_above_50dma", "pct_above_200dma"]:
                display[column] = display[column].map(lambda value: "Not available" if value is None else f"{float(value):.1f}%")
            st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.info("Breadth data is not available yet.")
        for note in report.breadth.notes:
            st.write(f"- {note}")

    options_frame = pd.DataFrame(report.options_positioning.rows)
    with st.container(border=True):
        st.subheader("Options / Dealer Positioning")
        st.caption(_caption_source(report.options_positioning.source.label, report.options_positioning.source.detail))
        if not options_frame.empty:
            display = options_frame.copy()
            if "historical_percentile" in display.columns:
                display["historical_percentile"] = display["historical_percentile"].map(lambda value: f"{float(value):.1f}%" if value is not None else "Not available")
            st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.info("Options positioning data is not available yet.")
        for note in report.options_positioning.notes:
            st.write(f"- {note}")

    sector_frame = pd.DataFrame(report.sector_leadership.rows)
    with st.container(border=True):
        st.subheader("Sector / Leadership")
        st.caption(_caption_source(report.sector_leadership.source.label, report.sector_leadership.source.detail))
        if sector_frame.empty:
            st.info("Sector leadership data is not available.")
        else:
            display = sector_frame.copy()
            for column in ["return_1w", "return_1m", "distance_50dma", "relative_strength_vs_spy"]:
                display[column] = display[column].map(_format_report_percent)
            st.dataframe(display, use_container_width=True, hide_index=True)

    seasonality_frame = pd.DataFrame(report.seasonality.rows)
    with st.container(border=True):
        st.subheader("Seasonality")
        st.caption(_caption_source(report.seasonality.source.label, report.seasonality.source.detail))
        if seasonality_frame.empty:
            st.info("Seasonality calculations are not available.")
        else:
            display = seasonality_frame.copy()
            display["value"] = display["value"].map(_format_report_percent)
            st.dataframe(display, use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.subheader("Macro Calendar")
        st.caption(_caption_source(report.macro_calendar.source.label, report.macro_calendar.source.detail))
        st.dataframe(pd.DataFrame(report.macro_calendar.rows), use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.subheader("Earnings Watch")
        st.caption(_caption_source(report.earnings_watch.source.label, report.earnings_watch.source.detail))
        st.dataframe(pd.DataFrame(report.earnings_watch.rows), use_container_width=True, hide_index=True)


def main() -> None:
    load_dotenv()
    settings = load_settings()
    app_settings = settings.get("app", {})
    bt_settings = settings.get("backtest", {})
    data_settings = settings.get("data", {})

    st.set_page_config(page_title="Personal Trading Lab", layout="wide")
    st.title("Personal Trading Lab")
    st.caption("Research-only stock backtesting platform with benchmark-aware robustness analysis.")

    db = TradingLabDatabase(data_settings.get("db_path", "data/trading_lab.duckdb"))
    provider = YFinanceDataProvider(
        database=db,
        cache_max_age_hours=int(data_settings.get("cache_max_age_hours", 24)),
        force_refresh_default=bool(data_settings.get("force_refresh_default", False)),
        allow_stale_cache=bool(data_settings.get("allow_stale_cache", False)),
    )

    with st.sidebar:
        st.header("Workflow")
        st.caption("Start here in SPY Workbench. Review candidates, then promote only one strategy at a time.")
        show_advanced_tools = st.checkbox("Show Advanced Tools", value=default_show_advanced_tools(), help="When off, the app stays focused on the simplified SPY workflow. When on, legacy research and diagnostics appear inside expanders.")
        st.header("Shared Settings")
        benchmark_symbol = "SPY"
        tickers = "SPY"
        start_date = st.date_input("Start date", value=pd.Timestamp("2020-01-01"))
        end_date = st.date_input("End date", value=pd.Timestamp.today().normalize())
        initial_capital = st.number_input("Initial capital", min_value=1000.0, value=float(bt_settings.get("default_initial_capital", 100000)))
        slippage_pct = st.number_input("Slippage %", min_value=0.0, value=float(bt_settings.get("default_slippage_pct", 0.0005)), format="%.5f", help="Shared default used by SPY Workbench and forward paper promotion.")
        commission_per_trade = st.number_input("Commission per trade", min_value=0.0, value=float(bt_settings.get("default_commission_per_trade", 1.0)))
        price_mode = st.selectbox("Price mode", ["raw_price_mode", "adjusted_price_mode"], help="Adjusted mode uses adjusted closes where available. Raw mode is more exposed to split and dividend distortions.")
        refresh_data = st.checkbox("Refresh data", value=bool(data_settings.get("force_refresh_default", False)))
        sizing_method = "percent_of_portfolio"
        position_size_value = float(bt_settings.get("default_position_size_value", 0.1))
        max_positions = 1
        stop_loss_pct = 0.08
        take_profit_pct = 0.15
        trailing_stop_pct = 0.0
        return_mode = "price_return_only"
        research_notes = ""
        research_tags = ""
        strategy_name = "Moving Average Crossover"
        strategy_params: dict[str, int | float] = default_strategy_params(strategy_name)
        run_backtest = False
        if show_advanced_tools:
            with st.expander("Advanced Research Controls", expanded=False):
                tickers = st.text_input("Tickers", value="AAPL,MSFT", help="Legacy multi-ticker backtest control. SPY Workbench remains fixed to SPY.")
                benchmark_symbol = st.text_input("Benchmark Symbol", value=app_settings.get("benchmark_symbol", "SPY"), help="Used by advanced research tools.").upper().strip()
                strategy_name = st.selectbox("Strategy", ["Moving Average Crossover", "RSI Mean Reversion", "Daily Breakout", "QQE/HMA Daily"])
                strategy_params = {}
                if strategy_name == "Moving Average Crossover":
                    strategy_params["fast_window"] = st.number_input("Fast window", min_value=2, value=20)
                    strategy_params["slow_window"] = st.number_input("Slow window", min_value=3, value=50)
                elif strategy_name == "RSI Mean Reversion":
                    strategy_params["rsi_length"] = st.number_input("RSI length", min_value=2, value=14)
                    strategy_params["buy_threshold"] = st.number_input("Buy threshold", min_value=1.0, max_value=50.0, value=30.0)
                    strategy_params["sell_threshold"] = st.number_input("Sell threshold", min_value=50.0, max_value=100.0, value=55.0)
                    strategy_params["max_holding_days"] = st.number_input("Max holding days", min_value=1, value=10)
                elif strategy_name == "Daily Breakout":
                    strategy_params["lookback_window"] = st.number_input("Lookback window", min_value=2, value=20)
                else:
                    strategy_params["hma_length"] = st.number_input("HMA length", min_value=2, value=21)
                    strategy_params["rsi_length"] = st.number_input("QQE RSI length", min_value=2, value=14)
                    strategy_params["rsi_smoothing"] = st.number_input("QQE RSI smoothing", min_value=1, value=5)
                    strategy_params["qqe_factor"] = st.number_input("QQE factor", min_value=0.1, value=4.236, step=0.1)
                    strategy_params["atr_smoothing"] = st.number_input("QQE ATR smoothing", min_value=1, value=5)
                    strategy_params["require_hma_slope"] = st.checkbox("Require HMA slope to be positive", value=True)
                    strategy_params["exit_on_hma_break"] = st.checkbox("Exit on HMA break", value=True)
                    strategy_params["exit_on_qqe_bearish"] = st.checkbox("Exit on QQE bearish turn", value=True)
                sizing_method = st.selectbox("Position sizing method", ["fixed_dollar", "percent_of_portfolio"])
                position_size_value = st.number_input("Position size value", min_value=0.0, value=float(bt_settings.get("default_position_size_value", 0.1)))
                max_positions = st.number_input("Max positions", min_value=1, value=int(bt_settings.get("default_max_positions", 5)))
                stop_loss_pct = st.number_input("Stop loss %", min_value=0.0, value=0.08, format="%.4f")
                take_profit_pct = st.number_input("Take profit %", min_value=0.0, value=0.15, format="%.4f")
                trailing_stop_pct = st.number_input("Trailing stop %", min_value=0.0, value=0.0, format="%.4f")
                return_mode = st.selectbox("Dividend mode", ["price_return_only", "total_return_with_dividends"], help="Dividend cash is only added in raw-price mode to avoid double-counting adjusted series.")
                research_notes = st.text_area("Run notes", value="", help="Optional notes to save with this backtest run.")
                research_tags = st.text_input("Run tags", value="", help="Comma-separated tags such as momentum, overfit-risk, options-candidate.")
                run_backtest = st.button("Run advanced backtest", type="primary")

    base_config = BacktestConfig(
        initial_capital=initial_capital,
        slippage_pct=slippage_pct,
        commission_per_trade=commission_per_trade,
        position_sizing_method=sizing_method,
        position_size_value=position_size_value,
        max_positions=int(max_positions),
        stop_loss_pct=stop_loss_pct or None,
        take_profit_pct=take_profit_pct or None,
        trailing_stop_pct=trailing_stop_pct or None,
        return_mode=return_mode,
        price_mode=price_mode,
    )

    if run_backtest:
        symbols = [ticker.strip().upper() for ticker in tickers.split(",") if ticker.strip()]
        if not symbols:
            st.error("At least one ticker is required.")
        else:
            strategy = build_strategy(strategy_name, strategy_params)
            with st.spinner("Loading market data and running backtest..."):
                data_by_symbol, statuses, validation_warnings = collect_data(provider, symbols, str(start_date), str(end_date), refresh_data, benchmark_symbol=benchmark_symbol)
                engine = BacktestEngine(database=db)
                result = engine.run(data_by_symbol=data_by_symbol, strategy=strategy, config=base_config, benchmark_symbol=benchmark_symbol)
            db.update_backtest_run_annotations(result.run_id, research_notes, research_tags)
            research = analyze_current_result(db, data_by_symbol, result, base_config, strategy_params, benchmark_symbol, symbols, str(start_date), str(end_date))
            st.session_state[SESSION_RESULT_KEY] = result
            st.session_state[SESSION_DATA_KEY] = data_by_symbol
            st.session_state[SESSION_STATUSES_KEY] = statuses
            st.session_state[SESSION_WARNINGS_KEY] = validation_warnings
            st.session_state[SESSION_RESEARCH_KEY] = research
            st.session_state[SESSION_META_KEY] = {
                "db": db,
                "config": base_config,
                "strategy": strategy,
                "strategy_name": strategy_name,
                "strategy_params": strategy_params,
                "symbols": symbols,
                "benchmark_symbol": benchmark_symbol,
                "result": result,
                "notes": research_notes,
                "tags": research_tags,
                "start_date": str(start_date),
                "end_date": str(end_date),
            }

    current_result: BacktestResult | None = st.session_state.get(SESSION_RESULT_KEY)
    current_data = st.session_state.get(SESSION_DATA_KEY, {})
    current_statuses = st.session_state.get(SESSION_STATUSES_KEY, [])
    current_warnings = st.session_state.get(SESSION_WARNINGS_KEY, [])
    current_research = st.session_state.get(SESSION_RESEARCH_KEY)
    current_meta = st.session_state.get(SESSION_META_KEY, {})

    tabs = st.tabs(get_primary_tab_labels())
    tab_spy_workbench, tab_pybroker, tab_forward, tab_history, tab_market_regime, tab_data = tabs

    with tab_spy_workbench:
        render_spy_strategy_lab(
            db=db,
            provider=provider,
            base_config=base_config,
            start_date=str(start_date),
            end_date=str(end_date),
            refresh_data=refresh_data,
        )
        if show_advanced_tools:
            with st.expander("Advanced Tools", expanded=False):
                render_strategy_qualification(
                    db=db,
                    provider=provider,
                    start_date=str(start_date),
                    end_date=str(end_date),
                    benchmark_symbol=benchmark_symbol,
                    refresh_data=refresh_data,
                    base_config=base_config,
                    default_tickers=normalize_ticker_list(tickers),
                )
                render_signal_scanner(
                    db=db,
                    provider=provider,
                    benchmark_symbol=benchmark_symbol,
                    refresh_data=refresh_data,
                    base_config=base_config,
                    default_tickers=normalize_ticker_list(tickers),
                )

    with tab_pybroker:
        render_pybroker_lab_workspace(
            provider=provider,
            start_date=str(start_date),
            end_date=str(end_date),
            refresh_data=refresh_data,
            base_config=base_config,
        )

    with tab_forward:
        render_forward_paper_workspace(
            db=db,
            provider=provider,
            benchmark_symbol=benchmark_symbol,
            refresh_data=refresh_data,
            base_config=base_config,
            default_tickers=normalize_ticker_list(tickers),
            current_strategy_name=strategy_name,
            current_strategy_params=strategy_params,
            show_advanced_tools=show_advanced_tools,
        )

    with tab_history:
        render_research_history_workspace(
            db=db,
            current_meta=current_meta,
            current_data=current_data,
            current_research=current_research,
            show_advanced_tools=show_advanced_tools,
        )

    with tab_market_regime:
        render_market_regime_workspace(
            provider=provider,
            start_date=str(start_date),
            end_date=str(end_date),
            refresh_data=refresh_data,
        )

    with tab_data:
        render_data_settings_workspace(
            db=db,
            provider=provider,
            statuses=current_statuses,
            warnings=current_warnings,
            current_research=current_research,
            start_date=str(start_date),
            end_date=str(end_date),
            refresh_data=refresh_data,
            settings_snapshot={
                "benchmark_symbol": benchmark_symbol,
                "cache_max_age_hours": data_settings.get("cache_max_age_hours", 24),
                "force_refresh_default": data_settings.get("force_refresh_default", False),
                "allow_stale_cache": data_settings.get("allow_stale_cache", False),
                "show_advanced_tools": show_advanced_tools,
            },
            show_advanced_tools=show_advanced_tools,
        )
