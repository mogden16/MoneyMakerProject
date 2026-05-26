from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from trading_lab.pybroker_lab.config import PyBrokerLabConfig
from trading_lab.pybroker_lab.fixed_strategy_utils import bars_to_frame, safe_json_dumps, sizing_method_label
from trading_lab.pybroker_lab.strategy_registry import fixed_strategy_library


AUDIT_COLUMNS = [
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
]

INDICATOR_DEBUG_BASE_COLUMNS = [
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
]

DATA_AUDIT_COLUMNS = ["check_name", "status", "detail"]
HIGHER_TIMEFRAME_CHECK_COLUMNS = [
    "lower_timeframe_timestamp",
    "higher_timeframe_source_timestamp",
    "higher_timeframe_close_timestamp",
    "higher_timeframe_value_used",
    "higher_timeframe_bar_complete",
    "lookahead_check_result",
]


def describe_data_origin(status) -> str:
    if status is None:
        return "unknown"
    if bool(status.performed_refresh) and bool(status.used_cached_data):
        return "cache_plus_refresh"
    if bool(status.performed_refresh):
        return "fresh_fetch"
    if bool(status.used_cached_data):
        return "cache"
    return str(status.cache_status)


def build_actual_data_used_frame(
    *,
    market_data: pd.DataFrame,
    config: PyBrokerLabConfig,
    statuses: list[Any],
    symbols: tuple[str, ...],
    provider_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    status_by_symbol = {str(status.symbol).upper(): status for status in statuses}
    normalized = market_data.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    for symbol in symbols:
        symbol_bars = normalized[normalized["symbol"] == symbol].sort_values("date").reset_index(drop=True)
        if symbol_bars.empty:
            continue
        status = status_by_symbol.get(symbol.upper())
        first_timestamp = pd.to_datetime(symbol_bars["date"].iloc[0])
        adjusted_series = pd.to_numeric(symbol_bars.get("adj_close"), errors="coerce") if "adj_close" in symbol_bars.columns else None
        close_series = pd.to_numeric(symbol_bars.get("close"), errors="coerce") if "close" in symbol_bars.columns else None
        adjusted_used = None
        if adjusted_series is not None and close_series is not None:
            adjusted_used = bool(adjusted_series.round(8).ne(close_series.round(8)).any())
        timestamps = pd.to_datetime(symbol_bars["date"])
        minutes = timestamps.dt.hour * 60 + timestamps.dt.minute
        out_of_hours = pd.Series(False, index=symbol_bars.index) if str(config.timeframe) == "1d" else ((minutes < 570) | (minutes >= 960))
        rows.append(
            {
                "symbol": symbol,
                "timeframe": config.timeframe,
                "requested_start_date": config.start_date,
                "requested_end_date": config.end_date,
                "actual_first_bar_timestamp": first_timestamp,
                "actual_last_bar_timestamp": pd.to_datetime(symbol_bars["date"].iloc[-1]),
                "number_of_bars": int(len(symbol_bars)),
                "data_provider": provider_name,
                "data_origin": describe_data_origin(status),
                "cache_status": getattr(status, "cache_status", "unknown"),
                "timezone": "America/New_York",
                "extended_hours_included": bool(out_of_hours.any()),
                "adjusted_data_used": adjusted_used,
                "timestamp_basis": "bar_start",
                "intraday_clamp_warning": _find_intraday_clamp_warning(status),
            }
        )
    return pd.DataFrame(rows)


def _signal_frame_for_strategy(strategy_id: str, bars: pd.DataFrame, config: PyBrokerLabConfig) -> pd.DataFrame:
    template = fixed_strategy_library()[strategy_id]
    if strategy_id == "legacy_mtf_qqe_rsi_momentum":
        signals = template.signal_frame_builder(bars, timeframe=config.timeframe)
    else:
        signals = template.signal_frame_builder(bars)
    frame = bars_to_frame(bars)
    signal_payload = signals.reset_index(drop=True).drop(columns=["timestamp"], errors="ignore")
    signal_payload = signal_payload.loc[:, [column for column in signal_payload.columns if column not in frame.columns]]
    payload = pd.concat([frame.reset_index(drop=True), signal_payload], axis=1)
    return payload


def build_strategy_debug_frame(strategy_id: str, *, bars: pd.DataFrame, config: PyBrokerLabConfig) -> pd.DataFrame:
    return _signal_frame_for_strategy(strategy_id, bars, config)


def build_trade_audit_frame(
    *,
    trades: pd.DataFrame,
    strategy_id: str,
    bars: pd.DataFrame,
    config: PyBrokerLabConfig,
) -> pd.DataFrame:
    template = fixed_strategy_library()[strategy_id]
    signal_frame = _signal_frame_for_strategy(strategy_id, bars, config).sort_values("timestamp").reset_index(drop=True)
    if trades.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)

    rows: list[dict[str, Any]] = []
    for _, trade in trades.iterrows():
        side = str(trade.get("type", "long")).lower()
        symbol = str(trade.get("symbol", ""))
        entry_ts = pd.to_datetime(trade.get("entry_date"))
        exit_ts = pd.to_datetime(trade.get("exit_date"))
        entry_signal_col = "long_entry_signal" if side == "long" else "short_entry_signal"
        exit_signal_col = "long_exit_signal" if side == "long" else "short_exit_signal"
        entry_reason_col = "long_entry_reason" if side == "long" else "short_entry_reason"
        exit_reason_col = "long_exit_reason" if side == "long" else "short_exit_reason"
        prior_entry = signal_frame[(signal_frame["symbol"] == symbol) & signal_frame["timestamp"].lt(entry_ts) & signal_frame[entry_signal_col].fillna(False)]
        prior_exit = signal_frame[(signal_frame["symbol"] == symbol) & signal_frame["timestamp"].lt(exit_ts) & signal_frame[exit_signal_col].fillna(False)]
        entry_row = prior_entry.iloc[-1] if not prior_entry.empty else pd.Series(dtype=object)
        exit_row = prior_exit.iloc[-1] if not prior_exit.empty else pd.Series(dtype=object)
        stop_reason = trade.get("stop")
        if stop_reason is not None and str(stop_reason).lower() not in {"", "none", "nan"}:
            exit_reason = f"stop:{stop_reason}"
        else:
            exit_reason = exit_row.get(exit_reason_col)
        entry_indicators = {column: entry_row.get(column) for column in template.indicator_snapshot_columns if column in entry_row.index}
        exit_indicators = {column: exit_row.get(column) for column in template.indicator_snapshot_columns if column in exit_row.index}
        shares = float(trade.get("shares", 0.0) or 0.0)
        entry_price = float(trade.get("entry", 0.0) or 0.0)
        exit_price = float(trade.get("exit", 0.0) or 0.0)
        pnl = float(trade.get("pnl", 0.0) or 0.0)
        holding_period = exit_ts - entry_ts if pd.notna(entry_ts) and pd.notna(exit_ts) else pd.NaT
        rows.append(
            {
                "strategy_id": strategy_id,
                "symbol": symbol,
                "timeframe": config.timeframe,
                "signal_timestamp": entry_row.get("timestamp"),
                "entry_timestamp": entry_ts,
                "entry_price": entry_price,
                "entry_reason": entry_row.get(entry_reason_col),
                "exit_timestamp": exit_ts,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "shares_contracts": shares,
                "position_value": shares * entry_price,
                "percent_return": float(trade.get("return_pct", 0.0) or 0.0),
                "dollar_pnl": pnl,
                "holding_period": str(holding_period) if holding_period is not pd.NaT else None,
                "holding_period_bars": int(trade.get("bars", 0) or 0),
                "entry_indicator_values": safe_json_dumps(entry_indicators),
                "exit_indicator_values": safe_json_dumps(exit_indicators),
            }
        )
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def build_indicator_debug_table(trade_audit: pd.DataFrame) -> pd.DataFrame:
    if trade_audit.empty:
        return pd.DataFrame(columns=INDICATOR_DEBUG_BASE_COLUMNS)
    rows: list[dict[str, Any]] = []
    for _, row in trade_audit.iterrows():
        entry_payload = _load_indicator_payload(row.get("entry_indicator_values"))
        exit_payload = _load_indicator_payload(row.get("exit_indicator_values"))
        record = {column: row.get(column) for column in INDICATOR_DEBUG_BASE_COLUMNS}
        for key, value in entry_payload.items():
            record[f"entry_{key}"] = value
        for key, value in exit_payload.items():
            record[f"exit_{key}"] = value
        rows.append(record)
    debug_table = pd.DataFrame(rows)
    fixed_columns = [column for column in INDICATOR_DEBUG_BASE_COLUMNS if column in debug_table.columns]
    other_columns = [column for column in debug_table.columns if column not in fixed_columns]
    return debug_table.loc[:, [*fixed_columns, *other_columns]]


def build_data_quality_audit(
    *,
    bars: pd.DataFrame,
    strategy_id: str,
    config: PyBrokerLabConfig,
    actual_data_row: dict[str, Any] | None,
) -> pd.DataFrame:
    frame = bars_to_frame(bars)
    template = fixed_strategy_library()[strategy_id]
    rows: list[dict[str, Any]] = []
    required_columns = {"open", "high", "low", "close", "volume", "timestamp"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    rows.append(_audit_row("required_ohlcv_columns", "PASS" if not missing_columns else "FAIL", "All required OHLCV columns exist." if not missing_columns else f"Missing columns: {', '.join(missing_columns)}"))
    duplicate_count = int(frame["timestamp"].duplicated().sum())
    rows.append(_audit_row("duplicate_timestamps", "PASS" if duplicate_count == 0 else "FAIL", "No duplicate timestamps found." if duplicate_count == 0 else f"Found {duplicate_count} duplicate timestamps."))
    is_sorted = bool(frame["timestamp"].is_monotonic_increasing)
    rows.append(_audit_row("sorted_timestamps", "PASS" if is_sorted else "FAIL", "Timestamps are sorted ascending." if is_sorted else "Timestamps are not sorted ascending."))
    null_ohlc = int(frame[["open", "high", "low", "close"]].isna().sum().sum())
    rows.append(_audit_row("null_ohlc_values", "PASS" if null_ohlc == 0 else "FAIL", "No null OHLC values found." if null_ohlc == 0 else f"Found {null_ohlc} null OHLC values."))
    non_positive_prices = int((frame[["open", "high", "low", "close"]] <= 0).sum().sum())
    rows.append(_audit_row("non_positive_prices", "PASS" if non_positive_prices == 0 else "FAIL", "All OHLC prices are positive." if non_positive_prices == 0 else f"Found {non_positive_prices} non-positive OHLC prices."))
    missing_volume = int(frame["volume"].isna().sum()) if "volume" in frame.columns else len(frame)
    rows.append(_audit_row("missing_volume_values", "PASS" if missing_volume == 0 else "WARNING", "No missing volume values found." if missing_volume == 0 else f"Found {missing_volume} bars with missing volume."))
    enough_bars = len(frame) >= int(template.minimum_required_bars) and len(frame) >= int(config.warmup_bars)
    required_bars = max(int(template.minimum_required_bars), int(config.warmup_bars))
    rows.append(_audit_row("sufficient_bars_for_strategy", "PASS" if enough_bars else "FAIL", f"Bar count {len(frame)} meets the required minimum of {required_bars}." if enough_bars else f"Bar count {len(frame)} is below the required minimum of {required_bars}."))
    clamp_warning = None if actual_data_row is None else actual_data_row.get("intraday_clamp_warning")
    rows.append(_audit_row("intraday_range_clamp", "WARNING" if clamp_warning else "PASS", "No intraday range clamp was detected." if not clamp_warning else str(clamp_warning)))
    return pd.DataFrame(rows, columns=DATA_AUDIT_COLUMNS)


def build_higher_timeframe_check(strategy_id: str, debug_frame: pd.DataFrame) -> pd.DataFrame:
    frame = debug_frame.copy().sort_values("timestamp").reset_index(drop=True)
    if strategy_id == "blackflag_fts_hma":
        return _higher_timeframe_table(
            frame,
            source_timestamp_column="higher_hma_source_timestamp",
            close_timestamp_column="higher_hma_higher_close_timestamp",
            value_column="higher_hma",
            complete_column="higher_hma_bar_complete",
            result_column="higher_hma_lookahead_result",
        )
    if strategy_id == "legacy_mtf_qqe_rsi_momentum":
        return _higher_timeframe_table(
            frame,
            source_timestamp_column="higher_qqe_source_timestamp",
            close_timestamp_column="higher_qqe_higher_close_timestamp",
            value_column="higher_rsi_ma",
            complete_column="higher_qqe_bar_complete",
            result_column="higher_qqe_lookahead_result",
        )
    return pd.DataFrame()


def build_chart_metadata(
    *,
    actual_data_row: dict[str, Any] | None,
    chart_frame: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    chart_start = None if chart_frame.empty else pd.to_datetime(chart_frame["timestamp"].min())
    chart_end = None if chart_frame.empty else pd.to_datetime(chart_frame["timestamp"].max())
    data = {
        "requested_start": None if actual_data_row is None else actual_data_row.get("requested_start_date"),
        "requested_end": None if actual_data_row is None else actual_data_row.get("requested_end_date"),
        "actual_first_bar": None if actual_data_row is None else actual_data_row.get("actual_first_bar_timestamp"),
        "actual_last_bar": None if actual_data_row is None else actual_data_row.get("actual_last_bar_timestamp"),
        "chart_display_range": None if chart_start is None or chart_end is None else f"{chart_start} to {chart_end}",
        "symbol": symbol,
        "timeframe": None if actual_data_row is None else actual_data_row.get("timeframe"),
        "data_provider": None if actual_data_row is None else actual_data_row.get("data_provider"),
        "number_of_bars": int(len(chart_frame)),
        "timezone": None if actual_data_row is None else actual_data_row.get("timezone"),
        "session_scope": None if actual_data_row is None else ("Extended hours" if actual_data_row.get("extended_hours_included") else "Regular hours"),
    }
    return pd.DataFrame([data])


def build_raw_bars_export_name(symbol: str, timeframe: str, bars: pd.DataFrame) -> str:
    frame = bars_to_frame(bars)
    actual_start = pd.to_datetime(frame["timestamp"].min()).strftime("%Y%m%d_%H%M")
    actual_end = pd.to_datetime(frame["timestamp"].max()).strftime("%Y%m%d_%H%M")
    return f"{symbol}_{timeframe}_{actual_start}_{actual_end}_yf_bars.csv"


def _load_indicator_payload(payload: Any) -> dict[str, Any]:
    if payload is None or (isinstance(payload, float) and math.isnan(payload)):
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}
    return {}


def build_benchmark_summary_frame(
    *,
    strategy_metrics: dict[str, Any],
    benchmark_metrics: dict[str, Any],
    actual_data_row: dict[str, Any] | None,
    sizing_method: str,
    sizing_value: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy_id": strategy_metrics.get("strategy_name"),
                "actual_first_bar_timestamp": None if actual_data_row is None else actual_data_row.get("actual_first_bar_timestamp"),
                "actual_last_bar_timestamp": None if actual_data_row is None else actual_data_row.get("actual_last_bar_timestamp"),
                "strategy_total_return": strategy_metrics.get("total_return", 0.0),
                "buy_and_hold_total_return": benchmark_metrics.get("total_return", 0.0),
                "strategy_max_drawdown": strategy_metrics.get("max_drawdown", 0.0),
                "buy_and_hold_max_drawdown": benchmark_metrics.get("max_drawdown", 0.0),
                "number_of_trades": strategy_metrics.get("trade_count", 0),
                "exposure_time": strategy_metrics.get("exposure", 0.0),
                "sizing_method": sizing_method,
                "sizing_label": sizing_method_label(sizing_method, sizing_value),
            }
        ]
    )


def filter_actual_data_row(actual_data_used: pd.DataFrame, symbol: str) -> dict[str, Any] | None:
    if actual_data_used.empty:
        return None
    match = actual_data_used[actual_data_used["symbol"] == symbol]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _find_intraday_clamp_warning(status) -> str | None:
    if status is None:
        return None
    for warning in getattr(status, "validation_warnings", []):
        if "clamped" in str(warning).lower():
            return str(warning)
    return None


def _audit_row(check_name: str, status: str, detail: str) -> dict[str, str]:
    return {"check_name": check_name, "status": status, "detail": detail}


def _higher_timeframe_table(
    frame: pd.DataFrame,
    *,
    source_timestamp_column: str,
    close_timestamp_column: str,
    value_column: str,
    complete_column: str,
    result_column: str,
) -> pd.DataFrame:
    if source_timestamp_column not in frame.columns:
        return pd.DataFrame(columns=HIGHER_TIMEFRAME_CHECK_COLUMNS)
    return pd.DataFrame(
        {
            "lower_timeframe_timestamp": pd.to_datetime(frame["timestamp"]),
            "higher_timeframe_source_timestamp": pd.to_datetime(frame[source_timestamp_column]),
            "higher_timeframe_close_timestamp": pd.to_datetime(frame[close_timestamp_column]),
            "higher_timeframe_value_used": frame.get(value_column),
            "higher_timeframe_bar_complete": frame.get(complete_column),
            "lookahead_check_result": frame.get(result_column),
        }
    )
