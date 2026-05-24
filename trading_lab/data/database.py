from __future__ import annotations

import json
from contextlib import contextmanager, suppress
from threading import RLock
from typing import Any

import duckdb
import pandas as pd

from trading_lab.data.cache import ensure_parent_dir


class TradingLabDatabase:
    """DuckDB persistence layer for market data, backtests, and research outputs."""

    _shared_connections: dict[str, duckdb.DuckDBPyConnection] = {}
    _shared_connection_locks: dict[str, RLock] = {}
    _registry_lock = RLock()

    def __init__(self, db_path: str = "data/trading_lab.duckdb") -> None:
        self.db_path = str(ensure_parent_dir(db_path))
        self._initialize()

    @classmethod
    def _get_shared_connection(cls, db_path: str) -> tuple[duckdb.DuckDBPyConnection, RLock]:
        with cls._registry_lock:
            lock = cls._shared_connection_locks.setdefault(db_path, RLock())
            connection = cls._shared_connections.get(db_path)
            if connection is None:
                connection = duckdb.connect(db_path)
                cls._shared_connections[db_path] = connection
            return connection, lock

    @contextmanager
    def connect(self):
        conn, lock = self._get_shared_connection(self.db_path)
        lock.acquire()
        try:
            yield conn
        finally:
            lock.release()

    @classmethod
    def close_shared_connection(cls, db_path: str) -> None:
        with cls._registry_lock:
            connection = cls._shared_connections.pop(db_path, None)
            cls._shared_connection_locks.pop(db_path, None)
        if connection is not None:
            with suppress(Exception):
                connection.close()

    def _initialize(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stock_bars (
                    source_vendor TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    timestamp TIMESTAMP,
                    session_date DATE,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    adj_close DOUBLE,
                    volume DOUBLE,
                    dividends DOUBLE,
                    stock_splits DOUBLE,
                    adjusted_flag BOOLEAN,
                    retrieved_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS corporate_actions (
                    source_vendor TEXT,
                    symbol TEXT,
                    action_type TEXT,
                    effective_date DATE,
                    cash_amount DOUBLE,
                    split_ratio DOUBLE,
                    split_from DOUBLE,
                    split_to DOUBLE,
                    retrieved_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    run_id TEXT,
                    strategy_name TEXT,
                    parameters_json TEXT,
                    symbols_csv TEXT,
                    start_date DATE,
                    end_date DATE,
                    created_at TIMESTAMP,
                    initial_capital DOUBLE,
                    timeframe TEXT,
                    total_return DOUBLE,
                    cagr DOUBLE,
                    max_drawdown DOUBLE,
                    sharpe_ratio DOUBLE,
                    sortino_ratio DOUBLE,
                    calmar_ratio DOUBLE,
                    win_rate DOUBLE,
                    profit_factor DOUBLE,
                    exposure_pct DOUBLE,
                    number_of_trades INTEGER,
                    benchmark_symbol TEXT,
                    benchmark_total_return DOUBLE,
                    benchmark_cagr DOUBLE,
                    benchmark_max_drawdown DOUBLE,
                    excess_cagr DOUBLE,
                    beta DOUBLE,
                    correlation DOUBLE,
                    return_mode TEXT,
                    price_mode TEXT,
                    sweep_id TEXT,
                    notes TEXT,
                    tags TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS backtest_trades (
                    run_id TEXT,
                    symbol TEXT,
                    entry_timestamp TIMESTAMP,
                    exit_timestamp TIMESTAMP,
                    entry_price DOUBLE,
                    exit_price DOUBLE,
                    shares DOUBLE,
                    pnl DOUBLE,
                    return_pct DOUBLE,
                    holding_days INTEGER,
                    exit_reason TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS backtest_equity_curve (
                    run_id TEXT,
                    timestamp TIMESTAMP,
                    equity DOUBLE,
                    cash DOUBLE,
                    positions_value DOUBLE,
                    drawdown DOUBLE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS backtest_benchmark_curve (
                    run_id TEXT,
                    benchmark_symbol TEXT,
                    timestamp TIMESTAMP,
                    benchmark_equity DOUBLE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_ingestion_log (
                    symbol TEXT,
                    timeframe TEXT,
                    requested_start DATE,
                    requested_end DATE,
                    latest_cached_session DATE,
                    expected_latest_session DATE,
                    cache_status TEXT,
                    validation_warnings_json TEXT,
                    used_cached_data BOOLEAN,
                    performed_refresh BOOLEAN,
                    retrieved_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS backtest_audit_results (
                    run_id TEXT,
                    severity TEXT,
                    message TEXT,
                    created_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS train_test_summaries (
                    run_id TEXT,
                    split_method TEXT,
                    split_value TEXT,
                    train_metrics_json TEXT,
                    test_metrics_json TEXT,
                    degradation_json TEXT,
                    created_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS walk_forward_runs (
                    walk_forward_id TEXT,
                    run_id TEXT,
                    benchmark_symbol TEXT,
                    train_window_months INTEGER,
                    test_window_months INTEGER,
                    step_months INTEGER,
                    min_train_trades INTEGER,
                    min_test_trades INTEGER,
                    summary_json TEXT,
                    created_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS walk_forward_folds (
                    walk_forward_id TEXT,
                    fold_number INTEGER,
                    train_start DATE,
                    train_end DATE,
                    test_start DATE,
                    test_end DATE,
                    train_cagr DOUBLE,
                    test_cagr DOUBLE,
                    train_max_drawdown DOUBLE,
                    test_max_drawdown DOUBLE,
                    train_sharpe DOUBLE,
                    test_sharpe DOUBLE,
                    train_trades INTEGER,
                    test_trades INTEGER,
                    degradation_score DOUBLE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS regime_metrics (
                    run_id TEXT,
                    benchmark_symbol TEXT,
                    regime_type TEXT,
                    regime_name TEXT,
                    total_return DOUBLE,
                    cagr DOUBLE,
                    max_drawdown DOUBLE,
                    sharpe_ratio DOUBLE,
                    sortino_ratio DOUBLE,
                    calmar_ratio DOUBLE,
                    number_of_trades INTEGER,
                    win_rate DOUBLE,
                    profit_factor DOUBLE,
                    average_trade_return DOUBLE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS robustness_scores (
                    run_id TEXT,
                    score INTEGER,
                    label TEXT,
                    strengths_json TEXT,
                    red_flags_json TEXT,
                    explanation_bullets_json TEXT,
                    created_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_diagnostics (
                    run_id TEXT,
                    benchmark_symbol TEXT,
                    coverage_ratio DOUBLE,
                    missing_session_count INTEGER,
                    dropped_strategy_dates INTEGER,
                    zero_return_days INTEGER,
                    status TEXT,
                    warnings_json TEXT,
                    created_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sweep_runs (
                    sweep_id TEXT,
                    created_at TIMESTAMP,
                    strategy_name TEXT,
                    tickers TEXT,
                    start_date DATE,
                    end_date DATE,
                    benchmark_symbol TEXT,
                    initial_capital DOUBLE,
                    price_mode TEXT,
                    position_sizing_method TEXT,
                    risk_settings_json TEXT,
                    sweep_config_json TEXT,
                    notes TEXT,
                    tags TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sweep_results (
                    sweep_result_id TEXT,
                    sweep_id TEXT,
                    backtest_run_id TEXT,
                    parameter_json TEXT,
                    total_return DOUBLE,
                    cagr DOUBLE,
                    max_drawdown DOUBLE,
                    sharpe DOUBLE,
                    sortino DOUBLE,
                    calmar DOUBLE,
                    win_rate DOUBLE,
                    profit_factor DOUBLE,
                    number_of_trades INTEGER,
                    exposure_pct DOUBLE,
                    robustness_score INTEGER,
                    beats_benchmark_flag BOOLEAN,
                    created_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sweep_parameters (
                    sweep_id TEXT,
                    parameter_name TEXT,
                    parameter_values_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_qualification_runs (
                    qualification_id TEXT,
                    created_at TIMESTAMP,
                    universe_name TEXT,
                    tickers TEXT,
                    benchmark_symbol TEXT,
                    start_date DATE,
                    end_date DATE,
                    price_mode TEXT,
                    initial_capital DOUBLE,
                    risk_settings_json TEXT,
                    notes TEXT,
                    tags TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_qualification_results (
                    qualification_result_id TEXT,
                    qualification_id TEXT,
                    strategy_name TEXT,
                    backtest_run_id TEXT,
                    total_return DOUBLE,
                    cagr DOUBLE,
                    max_drawdown DOUBLE,
                    sharpe DOUBLE,
                    sortino DOUBLE,
                    calmar DOUBLE,
                    win_rate DOUBLE,
                    profit_factor DOUBLE,
                    number_of_trades INTEGER,
                    exposure_pct DOUBLE,
                    excess_cagr DOUBLE,
                    robustness_score INTEGER,
                    red_flag_count INTEGER,
                    options_candidate_flag BOOLEAN,
                    candidate_label TEXT,
                    candidate_explanation_json TEXT,
                    created_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_trades (
                    paper_trade_id TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    ticker TEXT,
                    strategy_name TEXT,
                    signal_date TIMESTAMP,
                    planned_entry DOUBLE,
                    actual_entry DOUBLE,
                    stop_loss DOUBLE,
                    take_profit DOUBLE,
                    shares INTEGER,
                    status TEXT,
                    entry_date TIMESTAMP,
                    exit_date TIMESTAMP,
                    exit_price DOUBLE,
                    exit_reason TEXT,
                    realized_pnl DOUBLE,
                    realized_return_pct DOUBLE,
                    notes TEXT,
                    tags TEXT,
                    linked_backtest_run_id TEXT,
                    linked_qualification_id TEXT,
                    scanner_snapshot_id TEXT,
                    scanner_result_id TEXT,
                    signal_quality_score INTEGER,
                    qualification_status TEXT,
                    signal_explanation TEXT,
                    signal_warnings_json TEXT,
                    thesis_review TEXT,
                    execution_review TEXT,
                    what_went_well TEXT,
                    what_went_wrong TEXT,
                    lesson_learned TEXT,
                    mistake_tags TEXT,
                    followed_plan_flag BOOLEAN,
                    entry_quality_rating INTEGER,
                    exit_quality_rating INTEGER,
                    emotional_discipline_rating INTEGER,
                    universe_name TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_trade_events (
                    event_id TEXT,
                    paper_trade_id TEXT,
                    created_at TIMESTAMP,
                    event_type TEXT,
                    event_note TEXT,
                    price DOUBLE,
                    quantity INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS watchlist (
                    watchlist_id TEXT,
                    ticker TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    category TEXT,
                    notes TEXT,
                    tags TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scanner_snapshots (
                    snapshot_id TEXT,
                    created_at TIMESTAMP,
                    universe_name TEXT,
                    tickers TEXT,
                    strategies TEXT,
                    benchmark_symbol TEXT,
                    price_mode TEXT,
                    scanner_config_json TEXT,
                    notes TEXT,
                    tags TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scanner_snapshot_results (
                    scanner_result_id TEXT,
                    snapshot_id TEXT,
                    ticker TEXT,
                    strategy_name TEXT,
                    signal_type TEXT,
                    signal_date TIMESTAMP,
                    latest_close DOUBLE,
                    suggested_entry DOUBLE,
                    suggested_stop DOUBLE,
                    suggested_target DOUBLE,
                    risk_per_share DOUBLE,
                    reward_per_share DOUBLE,
                    reward_risk_ratio DOUBLE,
                    robustness_score INTEGER,
                    qualification_status TEXT,
                    signal_quality_score INTEGER,
                    signal_quality_label TEXT,
                    explanation TEXT,
                    warnings_json TEXT,
                    linked_paper_trade_id TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS active_paper_strategies (
                    active_strategy_id TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    status TEXT,
                    strategy_name TEXT,
                    strategy_parameters_json TEXT,
                    universe_name TEXT,
                    tickers TEXT,
                    timeframe TEXT,
                    benchmark_symbol TEXT,
                    price_mode TEXT,
                    initial_capital DOUBLE,
                    current_paper_equity DOUBLE,
                    position_sizing_method TEXT,
                    position_sizing_value DOUBLE,
                    max_positions INTEGER,
                    risk_settings_json TEXT,
                    slippage_pct DOUBLE,
                    commission_per_trade DOUBLE,
                    linked_qualification_id TEXT,
                    linked_sweep_id TEXT,
                    linked_backtest_run_id TEXT,
                    linked_search_run_id TEXT,
                    linked_search_result_id TEXT,
                    activation_reason TEXT,
                    notes TEXT,
                    tags TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS active_paper_strategy_events (
                    event_id TEXT,
                    active_strategy_id TEXT,
                    created_at TIMESTAMP,
                    event_type TEXT,
                    message TEXT,
                    details_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forward_paper_positions (
                    position_id TEXT,
                    active_strategy_id TEXT,
                    ticker TEXT,
                    timeframe TEXT,
                    strategy_name TEXT,
                    entry_signal_date TIMESTAMP,
                    entry_date TIMESTAMP,
                    entry_price DOUBLE,
                    shares INTEGER,
                    stop_loss DOUBLE,
                    take_profit DOUBLE,
                    trailing_stop DOUBLE,
                    current_stop DOUBLE,
                    status TEXT,
                    exit_signal_date TIMESTAMP,
                    exit_date TIMESTAMP,
                    exit_price DOUBLE,
                    exit_reason TEXT,
                    realized_pnl DOUBLE,
                    realized_return_pct DOUBLE,
                    realized_r_multiple DOUBLE,
                    notes TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forward_paper_orders (
                    order_id TEXT,
                    active_strategy_id TEXT,
                    created_at TIMESTAMP,
                    ticker TEXT,
                    timeframe TEXT,
                    order_type TEXT,
                    side TEXT,
                    status TEXT,
                    signal_date TIMESTAMP,
                    planned_fill_date TIMESTAMP,
                    planned_fill_rule TEXT,
                    planned_entry_reference DOUBLE,
                    stop_loss DOUBLE,
                    take_profit DOUBLE,
                    trailing_stop DOUBLE,
                    shares INTEGER,
                    estimated_price DOUBLE,
                    actual_fill_date TIMESTAMP,
                    actual_fill_price DOUBLE,
                    cancel_reason TEXT,
                    notes TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forward_paper_trades (
                    trade_id TEXT,
                    active_strategy_id TEXT,
                    ticker TEXT,
                    timeframe TEXT,
                    strategy_name TEXT,
                    entry_signal_date TIMESTAMP,
                    entry_date TIMESTAMP,
                    entry_price DOUBLE,
                    exit_signal_date TIMESTAMP,
                    exit_date TIMESTAMP,
                    exit_price DOUBLE,
                    shares INTEGER,
                    exit_reason TEXT,
                    realized_pnl DOUBLE,
                    realized_return_pct DOUBLE,
                    realized_r_multiple DOUBLE,
                    notes TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forward_paper_equity_curve (
                    active_strategy_id TEXT,
                    timestamp TIMESTAMP,
                    equity DOUBLE,
                    cash DOUBLE,
                    positions_value DOUBLE,
                    drawdown DOUBLE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spy_strategy_search_runs (
                    search_run_id TEXT,
                    created_at TIMESTAMP,
                    start_date DATE,
                    end_date DATE,
                    timeframe TEXT,
                    price_mode TEXT,
                    initial_capital DOUBLE,
                    slippage_pct DOUBLE,
                    commission_per_trade DOUBLE,
                    position_sizing_method TEXT,
                    position_sizing_value DOUBLE,
                    benchmark_symbol TEXT,
                    total_combinations_tested INTEGER,
                    notes TEXT,
                    tags TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spy_strategy_search_results (
                    result_id TEXT,
                    search_run_id TEXT,
                    timeframe TEXT,
                    entry_strategy_name TEXT,
                    entry_parameters_json TEXT,
                    exit_structure_key TEXT,
                    exit_structure_name TEXT,
                    exit_parameters_json TEXT,
                    backtest_run_id TEXT,
                    total_return DOUBLE,
                    cagr DOUBLE,
                    spy_cagr DOUBLE,
                    excess_cagr DOUBLE,
                    max_drawdown DOUBLE,
                    spy_max_drawdown DOUBLE,
                    drawdown_improvement DOUBLE,
                    sharpe DOUBLE,
                    sortino DOUBLE,
                    calmar DOUBLE,
                    number_of_trades INTEGER,
                    win_rate DOUBLE,
                    profit_factor DOUBLE,
                    avg_trade_return DOUBLE,
                    avg_r_multiple DOUBLE,
                    exposure_pct DOUBLE,
                    robustness_score INTEGER,
                    candidate_label TEXT,
                    ranking_category TEXT,
                    red_flag_count INTEGER,
                    summary_comment TEXT,
                    promoted_active_strategy_id TEXT,
                    created_at TIMESTAMP
                )
                """
            )
            for ddl in [
                "ALTER TABLE active_paper_strategies ADD COLUMN IF NOT EXISTS linked_search_run_id TEXT",
                "ALTER TABLE active_paper_strategies ADD COLUMN IF NOT EXISTS linked_search_result_id TEXT",
                "ALTER TABLE active_paper_strategies ADD COLUMN IF NOT EXISTS timeframe TEXT",
                "ALTER TABLE forward_paper_positions ADD COLUMN IF NOT EXISTS timeframe TEXT",
                "ALTER TABLE forward_paper_orders ADD COLUMN IF NOT EXISTS timeframe TEXT",
                "ALTER TABLE forward_paper_trades ADD COLUMN IF NOT EXISTS timeframe TEXT",
                "ALTER TABLE spy_strategy_search_runs ADD COLUMN IF NOT EXISTS timeframe TEXT",
                "ALTER TABLE spy_strategy_search_results ADD COLUMN IF NOT EXISTS timeframe TEXT",
                "ALTER TABLE spy_strategy_search_results ADD COLUMN IF NOT EXISTS entry_preset_id TEXT",
                "ALTER TABLE spy_strategy_search_results ADD COLUMN IF NOT EXISTS entry_preset_label TEXT",
                "ALTER TABLE spy_strategy_search_results ADD COLUMN IF NOT EXISTS exit_structure_key TEXT",
                "ALTER TABLE spy_strategy_search_results ADD COLUMN IF NOT EXISTS exit_preset_id TEXT",
                "ALTER TABLE spy_strategy_search_results ADD COLUMN IF NOT EXISTS exit_preset_label TEXT",
                "ALTER TABLE spy_strategy_search_results ADD COLUMN IF NOT EXISTS experimental BOOLEAN",
                "ALTER TABLE spy_strategy_search_results ADD COLUMN IF NOT EXISTS complexity_score INTEGER",
            ]:
                try:
                    conn.execute(ddl)
                except duckdb.Error:
                    pass
            self._ensure_backtest_run_columns(conn)
            self._ensure_paper_trade_columns(conn)
            self._ensure_watchlist_columns(conn)

    def _ensure_backtest_run_columns(self, conn: duckdb.DuckDBPyConnection) -> None:
        for statement in [
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS symbols_csv TEXT",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS start_date DATE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS end_date DATE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS created_at TIMESTAMP",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS timeframe TEXT",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS total_return DOUBLE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS cagr DOUBLE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS max_drawdown DOUBLE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS sharpe_ratio DOUBLE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS sortino_ratio DOUBLE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS calmar_ratio DOUBLE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS win_rate DOUBLE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS profit_factor DOUBLE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS exposure_pct DOUBLE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS number_of_trades INTEGER",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS benchmark_symbol TEXT",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS benchmark_total_return DOUBLE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS benchmark_cagr DOUBLE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS benchmark_max_drawdown DOUBLE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS excess_cagr DOUBLE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS beta DOUBLE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS correlation DOUBLE",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS return_mode TEXT",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS price_mode TEXT",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS sweep_id TEXT",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS notes TEXT",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS tags TEXT",
        ]:
            conn.execute(statement)

    def _ensure_paper_trade_columns(self, conn: duckdb.DuckDBPyConnection) -> None:
        for statement in [
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS scanner_snapshot_id TEXT",
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS scanner_result_id TEXT",
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS signal_quality_score INTEGER",
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS qualification_status TEXT",
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS signal_explanation TEXT",
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS signal_warnings_json TEXT",
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS thesis_review TEXT",
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS execution_review TEXT",
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS what_went_well TEXT",
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS what_went_wrong TEXT",
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS lesson_learned TEXT",
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS mistake_tags TEXT",
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS followed_plan_flag BOOLEAN",
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS entry_quality_rating INTEGER",
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS exit_quality_rating INTEGER",
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS emotional_discipline_rating INTEGER",
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS universe_name TEXT",
        ]:
            conn.execute(statement)

    def _ensure_watchlist_columns(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute("ALTER TABLE watchlist ADD COLUMN IF NOT EXISTS category TEXT")

    def read_stock_bars(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1d") -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT *
                FROM stock_bars
                WHERE symbol = ?
                  AND timeframe = ?
                  AND session_date BETWEEN ? AND ?
                ORDER BY timestamp
                """,
                [symbol, timeframe, start_date, end_date],
            ).df()

    def get_latest_stock_bar_session(self, symbol: str, timeframe: str = "1d") -> pd.Timestamp | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT MAX(session_date) AS latest_session
                FROM stock_bars
                WHERE symbol = ?
                  AND timeframe = ?
                """,
                [symbol, timeframe],
            ).fetchone()
        if row is None or row[0] is None:
            return None
        return pd.Timestamp(row[0])

    def upsert_stock_bars(self, bars: pd.DataFrame, symbol: str, timeframe: str = "1d") -> None:
        if bars.empty:
            return
        min_date = pd.to_datetime(bars["session_date"]).min().date()
        max_date = pd.to_datetime(bars["session_date"]).max().date()
        with self.connect() as conn:
            conn.execute("DELETE FROM stock_bars WHERE symbol = ? AND timeframe = ? AND session_date BETWEEN ? AND ?", [symbol, timeframe, min_date, max_date])
            conn.register("bars_df", bars)
            conn.execute("INSERT INTO stock_bars SELECT * FROM bars_df")
            conn.unregister("bars_df")

    def replace_stock_bars(self, bars: pd.DataFrame, symbol: str, timeframe: str = "1d") -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM stock_bars WHERE symbol = ? AND timeframe = ?", [symbol, timeframe])
            conn.register("bars_df", bars)
            conn.execute("INSERT INTO stock_bars SELECT * FROM bars_df")
            conn.unregister("bars_df")

    def upsert_corporate_actions(self, actions: pd.DataFrame, symbol: str) -> None:
        if actions.empty:
            return
        min_date = pd.to_datetime(actions["effective_date"]).min().date()
        max_date = pd.to_datetime(actions["effective_date"]).max().date()
        with self.connect() as conn:
            conn.execute("DELETE FROM corporate_actions WHERE symbol = ? AND effective_date BETWEEN ? AND ?", [symbol, min_date, max_date])
            conn.register("actions_df", actions)
            conn.execute("INSERT INTO corporate_actions SELECT * FROM actions_df")
            conn.unregister("actions_df")

    def read_corporate_actions(self, symbols: list[str], limit_per_symbol: int = 10) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        placeholders = ", ".join(["?"] * len(symbols))
        with self.connect() as conn:
            return conn.execute(
                f"""
                SELECT source_vendor, symbol, action_type, effective_date, cash_amount, split_ratio, split_from, split_to, retrieved_at
                FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY effective_date DESC) AS row_num
                    FROM corporate_actions
                    WHERE symbol IN ({placeholders})
                )
                WHERE row_num <= ?
                ORDER BY effective_date DESC, symbol
                """,
                [*symbols, limit_per_symbol],
            ).df()

    def read_corporate_actions_for_period(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        placeholders = ", ".join(["?"] * len(symbols))
        with self.connect() as conn:
            return conn.execute(
                f"""
                SELECT *
                FROM corporate_actions
                WHERE symbol IN ({placeholders})
                  AND effective_date BETWEEN ? AND ?
                ORDER BY effective_date, symbol
                """,
                [*symbols, start_date, end_date],
            ).df()

    def insert_ingestion_log(self, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.register("ingestion_df", pd.DataFrame([payload]))
            conn.execute("INSERT INTO data_ingestion_log SELECT * FROM ingestion_df")
            conn.unregister("ingestion_df")

    def insert_backtest_run(self, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            frame = pd.DataFrame([payload])
            columns = ", ".join(frame.columns)
            conn.register("run_df", frame)
            conn.execute(f"INSERT INTO backtest_runs ({columns}) SELECT {columns} FROM run_df")
            conn.unregister("run_df")

    def update_backtest_run_annotations(self, run_id: str, notes: str = "", tags: str = "") -> None:
        with self.connect() as conn:
            conn.execute("UPDATE backtest_runs SET notes = ?, tags = ? WHERE run_id = ?", [notes, tags, run_id])

    def insert_backtest_trades(self, trades: pd.DataFrame) -> None:
        if trades.empty:
            return
        with self.connect() as conn:
            conn.register("trades_df", trades)
            conn.execute("INSERT INTO backtest_trades SELECT * FROM trades_df")
            conn.unregister("trades_df")

    def insert_backtest_equity_curve(self, equity_curve: pd.DataFrame) -> None:
        if equity_curve.empty:
            return
        with self.connect() as conn:
            conn.register("equity_df", equity_curve)
            conn.execute("INSERT INTO backtest_equity_curve SELECT * FROM equity_df")
            conn.unregister("equity_df")

    def insert_backtest_benchmark_curve(self, benchmark_curve: pd.DataFrame) -> None:
        if benchmark_curve.empty:
            return
        with self.connect() as conn:
            conn.register("benchmark_df", benchmark_curve)
            conn.execute("INSERT INTO backtest_benchmark_curve SELECT * FROM benchmark_df")
            conn.unregister("benchmark_df")

    def replace_audit_results(self, run_id: str, findings: pd.DataFrame) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM backtest_audit_results WHERE run_id = ?", [run_id])
            if not findings.empty:
                conn.register("audit_df", findings)
                conn.execute("INSERT INTO backtest_audit_results SELECT * FROM audit_df")
                conn.unregister("audit_df")

    def read_audit_results(self, run_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM backtest_audit_results WHERE run_id = ? ORDER BY created_at, severity", [run_id]).df()

    def replace_train_test_summary(self, run_id: str, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM train_test_summaries WHERE run_id = ?", [run_id])
            conn.register("train_test_df", pd.DataFrame([payload]))
            conn.execute("INSERT INTO train_test_summaries SELECT * FROM train_test_df")
            conn.unregister("train_test_df")

    def read_train_test_summary(self, run_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM train_test_summaries WHERE run_id = ? ORDER BY created_at DESC", [run_id]).df()

    def replace_walk_forward_run(self, payload: dict[str, Any], folds: pd.DataFrame) -> None:
        walk_forward_id = payload["walk_forward_id"]
        with self.connect() as conn:
            conn.execute("DELETE FROM walk_forward_runs WHERE walk_forward_id = ?", [walk_forward_id])
            conn.execute("DELETE FROM walk_forward_folds WHERE walk_forward_id = ?", [walk_forward_id])
            conn.register("walk_df", pd.DataFrame([payload]))
            conn.execute("INSERT INTO walk_forward_runs SELECT * FROM walk_df")
            conn.unregister("walk_df")
            if not folds.empty:
                conn.register("folds_df", folds)
                conn.execute("INSERT INTO walk_forward_folds SELECT * FROM folds_df")
                conn.unregister("folds_df")

    def read_walk_forward_runs(self, run_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM walk_forward_runs WHERE run_id = ? ORDER BY created_at DESC", [run_id]).df()

    def read_walk_forward_folds(self, walk_forward_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM walk_forward_folds WHERE walk_forward_id = ? ORDER BY fold_number", [walk_forward_id]).df()

    def replace_regime_metrics(self, run_id: str, regime_metrics: pd.DataFrame) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM regime_metrics WHERE run_id = ?", [run_id])
            if not regime_metrics.empty:
                conn.register("regime_df", regime_metrics)
                conn.execute("INSERT INTO regime_metrics SELECT * FROM regime_df")
                conn.unregister("regime_df")

    def read_regime_metrics(self, run_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM regime_metrics WHERE run_id = ? ORDER BY regime_type, regime_name", [run_id]).df()

    def replace_robustness_score(self, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM robustness_scores WHERE run_id = ?", [payload["run_id"]])
            conn.register("robust_df", pd.DataFrame([payload]))
            conn.execute("INSERT INTO robustness_scores SELECT * FROM robust_df")
            conn.unregister("robust_df")

    def read_robustness_score(self, run_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM robustness_scores WHERE run_id = ? ORDER BY created_at DESC", [run_id]).df()

    def replace_benchmark_diagnostics(self, run_id: str, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM benchmark_diagnostics WHERE run_id = ?", [run_id])
            conn.register("benchmark_diag_df", pd.DataFrame([payload]))
            conn.execute("INSERT INTO benchmark_diagnostics SELECT * FROM benchmark_diag_df")
            conn.unregister("benchmark_diag_df")

    def read_benchmark_diagnostics(self, run_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM benchmark_diagnostics WHERE run_id = ? ORDER BY created_at DESC", [run_id]).df()

    def replace_sweep_run(self, payload: dict[str, Any], results: pd.DataFrame, parameters: pd.DataFrame) -> None:
        sweep_id = payload["sweep_id"]
        with self.connect() as conn:
            conn.execute("DELETE FROM sweep_runs WHERE sweep_id = ?", [sweep_id])
            conn.execute("DELETE FROM sweep_results WHERE sweep_id = ?", [sweep_id])
            conn.execute("DELETE FROM sweep_parameters WHERE sweep_id = ?", [sweep_id])
            conn.register("sweep_run_df", pd.DataFrame([payload]))
            conn.execute("INSERT INTO sweep_runs SELECT * FROM sweep_run_df")
            conn.unregister("sweep_run_df")
            if not results.empty:
                conn.register("sweep_results_df", results)
                conn.execute("INSERT INTO sweep_results SELECT * FROM sweep_results_df")
                conn.unregister("sweep_results_df")
            if not parameters.empty:
                conn.register("sweep_parameters_df", parameters)
                conn.execute("INSERT INTO sweep_parameters SELECT * FROM sweep_parameters_df")
                conn.unregister("sweep_parameters_df")

    def update_sweep_annotations(self, sweep_id: str, notes: str = "", tags: str = "") -> None:
        with self.connect() as conn:
            conn.execute("UPDATE sweep_runs SET notes = ?, tags = ? WHERE sweep_id = ?", [notes, tags, sweep_id])

    def replace_strategy_qualification_run(self, payload: dict[str, Any], results: pd.DataFrame) -> None:
        qualification_id = payload["qualification_id"]
        with self.connect() as conn:
            conn.execute("DELETE FROM strategy_qualification_runs WHERE qualification_id = ?", [qualification_id])
            conn.execute("DELETE FROM strategy_qualification_results WHERE qualification_id = ?", [qualification_id])
            conn.register("qualification_run_df", pd.DataFrame([payload]))
            conn.execute("INSERT INTO strategy_qualification_runs SELECT * FROM qualification_run_df")
            conn.unregister("qualification_run_df")
            if not results.empty:
                conn.register("qualification_results_df", results)
                conn.execute("INSERT INTO strategy_qualification_results SELECT * FROM qualification_results_df")
                conn.unregister("qualification_results_df")

    def update_strategy_qualification_annotations(self, qualification_id: str, notes: str = "", tags: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE strategy_qualification_runs SET notes = ?, tags = ? WHERE qualification_id = ?",
                [notes, tags, qualification_id],
            )

    def list_strategy_qualification_runs(self, limit: int = 50, tag: str | None = None) -> pd.DataFrame:
        where_clause = "WHERE lower(coalesce(q.tags, '')) LIKE lower(?)" if tag else ""
        parameters: list[Any] = [f"%{tag}%"] if tag else []
        parameters.append(limit)
        with self.connect() as conn:
            return conn.execute(
                f"""
                SELECT
                    q.qualification_id,
                    q.created_at,
                    q.universe_name,
                    q.tickers,
                    q.benchmark_symbol,
                    q.start_date,
                    q.end_date,
                    q.price_mode,
                    q.notes,
                    q.tags,
                    COUNT(r.qualification_result_id) AS strategy_count,
                    MAX(r.cagr) AS best_cagr,
                    MEDIAN(r.cagr) AS median_cagr,
                    MAX(r.robustness_score) AS best_robustness,
                    AVG(CASE WHEN r.options_candidate_flag THEN 1.0 ELSE 0.0 END) AS candidate_ratio
                FROM strategy_qualification_runs q
                LEFT JOIN strategy_qualification_results r ON q.qualification_id = r.qualification_id
                {where_clause}
                GROUP BY 1,2,3,4,5,6,7,8,9,10
                ORDER BY q.created_at DESC
                LIMIT ?
                """,
                parameters,
            ).df()

    def get_strategy_qualification_run(self, qualification_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM strategy_qualification_runs WHERE qualification_id = ? ORDER BY created_at DESC LIMIT 1",
                [qualification_id],
            ).fetchone()
            if row is None:
                return None
            columns = [item[0] for item in conn.description]
        return dict(zip(columns, row, strict=False))

    def read_strategy_qualification_results(self, qualification_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT *
                FROM strategy_qualification_results
                WHERE qualification_id = ?
                ORDER BY robustness_score DESC, cagr DESC
                """,
                [qualification_id],
            ).df()

    def replace_spy_strategy_search_run(self, payload: dict[str, Any], results: pd.DataFrame) -> None:
        search_run_id = payload["search_run_id"]
        stored_results = results.copy()
        if not stored_results.empty:
            for column in ["entry_parameters_json", "exit_parameters_json"]:
                if column in stored_results.columns:
                    stored_results[column] = stored_results[column].apply(lambda value: json.dumps(value, default=str) if isinstance(value, dict) else value)
        with self.connect() as conn:
            conn.execute("DELETE FROM spy_strategy_search_runs WHERE search_run_id = ?", [search_run_id])
            conn.execute("DELETE FROM spy_strategy_search_results WHERE search_run_id = ?", [search_run_id])
            run_frame = pd.DataFrame([payload])
            run_columns = ", ".join(run_frame.columns)
            conn.register("spy_search_run_df", run_frame)
            conn.execute(f"INSERT INTO spy_strategy_search_runs ({run_columns}) SELECT {run_columns} FROM spy_search_run_df")
            conn.unregister("spy_search_run_df")
            if not stored_results.empty:
                conn.register("spy_search_results_df", stored_results)
                columns = ", ".join(stored_results.columns)
                conn.execute(f"INSERT INTO spy_strategy_search_results ({columns}) SELECT {columns} FROM spy_search_results_df")
                conn.unregister("spy_search_results_df")

    def list_spy_strategy_search_runs(self, limit: int = 50, tag: str | None = None) -> pd.DataFrame:
        where_clause = "WHERE lower(coalesce(r.tags, '')) LIKE lower(?)" if tag else ""
        parameters: list[Any] = [f"%{tag}%"] if tag else []
        parameters.append(limit)
        with self.connect() as conn:
            return conn.execute(
                f"""
                SELECT
                    r.search_run_id,
                    r.created_at,
                    r.start_date,
                    r.end_date,
                    r.timeframe,
                    r.price_mode,
                    r.initial_capital,
                    r.slippage_pct,
                    r.commission_per_trade,
                    r.position_sizing_method,
                    r.position_sizing_value,
                    r.benchmark_symbol,
                    r.total_combinations_tested,
                    r.notes,
                    r.tags,
                    MAX(s.cagr) AS best_cagr,
                    MEDIAN(s.cagr) AS median_cagr,
                    MIN(s.cagr) AS worst_cagr,
                    AVG(CASE WHEN s.total_return > 0 THEN 1.0 ELSE 0.0 END) AS percent_profitable,
                    AVG(CASE WHEN s.excess_cagr > 0 THEN 1.0 ELSE 0.0 END) AS percent_beating_spy
                FROM spy_strategy_search_runs r
                LEFT JOIN spy_strategy_search_results s ON r.search_run_id = s.search_run_id
                {where_clause}
                GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
                ORDER BY r.created_at DESC
                LIMIT ?
                """,
                parameters,
            ).df()

    def get_spy_strategy_search_run(self, search_run_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM spy_strategy_search_runs WHERE search_run_id = ? ORDER BY created_at DESC LIMIT 1",
                [search_run_id],
            ).fetchone()
            if row is None:
                return None
            columns = [item[0] for item in conn.description]
        return dict(zip(columns, row, strict=False))

    def read_spy_strategy_search_results(self, search_run_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            frame = conn.execute(
                """
                SELECT *
                FROM spy_strategy_search_results
                WHERE search_run_id = ?
                ORDER BY ranking_category DESC, candidate_label DESC, cagr DESC
                """,
                [search_run_id],
            ).df()
        for column in ["entry_parameters_json", "exit_parameters_json"]:
            if column in frame.columns and not frame.empty:
                frame[column] = frame[column].apply(lambda value: json.loads(value) if isinstance(value, str) and value else {})
        return frame

    def update_spy_strategy_search_result_promotion(self, result_id: str, active_strategy_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE spy_strategy_search_results SET promoted_active_strategy_id = ? WHERE result_id = ?",
                [active_strategy_id, result_id],
            )

    def insert_paper_trade(self, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            frame = pd.DataFrame([payload])
            columns = ", ".join(frame.columns)
            conn.register("paper_trade_df", frame)
            conn.execute(f"INSERT INTO paper_trades ({columns}) SELECT {columns} FROM paper_trade_df")
            conn.unregister("paper_trade_df")

    def update_paper_trade(self, payload: dict[str, Any]) -> None:
        paper_trade_id = payload["paper_trade_id"]
        with self.connect() as conn:
            conn.execute("DELETE FROM paper_trades WHERE paper_trade_id = ?", [paper_trade_id])
            frame = pd.DataFrame([payload])
            columns = ", ".join(frame.columns)
            conn.register("paper_trade_df", frame)
            conn.execute(f"INSERT INTO paper_trades ({columns}) SELECT {columns} FROM paper_trade_df")
            conn.unregister("paper_trade_df")

    def get_paper_trade(self, paper_trade_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM paper_trades WHERE paper_trade_id = ? ORDER BY created_at DESC LIMIT 1",
                [paper_trade_id],
            ).fetchone()
            if row is None:
                return None
            columns = [item[0] for item in conn.description]
        return dict(zip(columns, row, strict=False))

    def list_paper_trades(self, status: str | None = None) -> pd.DataFrame:
        where_clause = "WHERE status = ?" if status else ""
        parameters: list[Any] = [status] if status else []
        with self.connect() as conn:
            return conn.execute(
                f"""
                SELECT *
                FROM paper_trades
                {where_clause}
                ORDER BY created_at DESC
                """,
                parameters,
            ).df()

    def list_paper_trades_with_context(self) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT
                    pt.*,
                    br.cagr AS linked_strategy_cagr,
                    br.excess_cagr AS linked_strategy_excess_cagr,
                    br.benchmark_symbol AS linked_benchmark_symbol,
                    rs.score AS linked_robustness_score,
                    sqr.candidate_label AS linked_candidate_label
                FROM paper_trades pt
                LEFT JOIN backtest_runs br ON pt.linked_backtest_run_id = br.run_id
                LEFT JOIN robustness_scores rs ON pt.linked_backtest_run_id = rs.run_id
                LEFT JOIN strategy_qualification_results sqr ON pt.linked_qualification_id = sqr.qualification_id AND pt.strategy_name = sqr.strategy_name
                ORDER BY pt.created_at DESC
                """
            ).df()

    def insert_paper_trade_event(self, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.register("paper_event_df", pd.DataFrame([payload]))
            conn.execute("INSERT INTO paper_trade_events SELECT * FROM paper_event_df")
            conn.unregister("paper_event_df")

    def read_paper_trade_events(self, paper_trade_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM paper_trade_events WHERE paper_trade_id = ? ORDER BY created_at",
                [paper_trade_id],
            ).df()

    def upsert_watchlist_item(self, payload: dict[str, Any]) -> None:
        watchlist_id = payload["watchlist_id"]
        with self.connect() as conn:
            conn.execute("DELETE FROM watchlist WHERE watchlist_id = ?", [watchlist_id])
            frame = pd.DataFrame([payload])
            columns = ", ".join(frame.columns)
            conn.register("watchlist_df", frame)
            conn.execute(f"INSERT INTO watchlist ({columns}) SELECT {columns} FROM watchlist_df")
            conn.unregister("watchlist_df")

    def list_watchlist(self, tag: str | None = None) -> pd.DataFrame:
        where_clause = "WHERE lower(coalesce(tags, '')) LIKE lower(?)" if tag else ""
        parameters: list[Any] = [f"%{tag}%"] if tag else []
        with self.connect() as conn:
            return conn.execute(
                f"""
                SELECT *
                FROM watchlist
                {where_clause}
                ORDER BY updated_at DESC, ticker
                """,
                parameters,
            ).df()

    def replace_scanner_snapshot(self, payload: dict[str, Any], results: pd.DataFrame) -> None:
        snapshot_id = payload["snapshot_id"]
        with self.connect() as conn:
            conn.execute("DELETE FROM scanner_snapshots WHERE snapshot_id = ?", [snapshot_id])
            conn.execute("DELETE FROM scanner_snapshot_results WHERE snapshot_id = ?", [snapshot_id])
            conn.register("scanner_snapshot_df", pd.DataFrame([payload]))
            conn.execute("INSERT INTO scanner_snapshots SELECT * FROM scanner_snapshot_df")
            conn.unregister("scanner_snapshot_df")
            if not results.empty:
                conn.register("scanner_results_df", results)
                conn.execute("INSERT INTO scanner_snapshot_results SELECT * FROM scanner_results_df")
                conn.unregister("scanner_results_df")

    def list_scanner_snapshots(self, limit: int = 100, tag: str | None = None) -> pd.DataFrame:
        where_clause = "WHERE lower(coalesce(s.tags, '')) LIKE lower(?)" if tag else ""
        parameters: list[Any] = [f"%{tag}%"] if tag else []
        parameters.append(limit)
        with self.connect() as conn:
            return conn.execute(
                f"""
                SELECT
                    s.snapshot_id,
                    s.created_at,
                    s.universe_name,
                    s.tickers,
                    s.strategies,
                    s.benchmark_symbol,
                    s.price_mode,
                    s.notes,
                    s.tags,
                    COUNT(r.scanner_result_id) AS signal_count,
                    SUM(CASE WHEN r.signal_type = 'new_buy_signal' THEN 1 ELSE 0 END) AS new_buy_count,
                    SUM(CASE WHEN r.signal_type = 'exit_signal' THEN 1 ELSE 0 END) AS exit_count,
                    AVG(r.signal_quality_score) AS average_signal_quality
                FROM scanner_snapshots s
                LEFT JOIN scanner_snapshot_results r ON s.snapshot_id = r.snapshot_id
                {where_clause}
                GROUP BY 1,2,3,4,5,6,7,8,9
                ORDER BY s.created_at DESC
                LIMIT ?
                """,
                parameters,
            ).df()

    def get_scanner_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM scanner_snapshots WHERE snapshot_id = ? ORDER BY created_at DESC LIMIT 1",
                [snapshot_id],
            ).fetchone()
            if row is None:
                return None
            columns = [item[0] for item in conn.description]
        return dict(zip(columns, row, strict=False))

    def read_scanner_snapshot_results(self, snapshot_id: str, action_status: str | None = None) -> pd.DataFrame:
        if action_status:
            status_filter = """
                WHERE r.snapshot_id = ?
                  AND (
                    (? = 'no_action' AND coalesce(pt.status, '') = '')
                    OR (? = 'planned' AND pt.status = 'planned')
                    OR (? = 'open' AND pt.status = 'open')
                    OR (? = 'closed' AND pt.status = 'closed')
                    OR (? = 'canceled' AND pt.status = 'canceled')
                  )
            """
            parameters: list[Any] = [snapshot_id, action_status, action_status, action_status, action_status, action_status]
        else:
            status_filter = "WHERE r.snapshot_id = ?"
            parameters = [snapshot_id]
        with self.connect() as conn:
            return conn.execute(
                f"""
                SELECT
                    r.*,
                    COALESCE(pt.status, 'no_action') AS action_status
                FROM scanner_snapshot_results r
                LEFT JOIN paper_trades pt ON r.linked_paper_trade_id = pt.paper_trade_id
                {status_filter}
                ORDER BY r.signal_quality_score DESC, r.ticker, r.strategy_name
                """,
                parameters,
            ).df()

    def update_scanner_snapshot_result_link(self, scanner_result_id: str, paper_trade_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE scanner_snapshot_results SET linked_paper_trade_id = ? WHERE scanner_result_id = ?",
                [paper_trade_id, scanner_result_id],
            )

    def insert_active_paper_strategy(self, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            frame = pd.DataFrame([payload])
            columns = ", ".join(frame.columns)
            conn.register("active_strategy_df", frame)
            conn.execute(f"INSERT INTO active_paper_strategies ({columns}) SELECT {columns} FROM active_strategy_df")
            conn.unregister("active_strategy_df")

    def update_active_paper_strategy(self, payload: dict[str, Any]) -> None:
        strategy_id = payload["active_strategy_id"]
        with self.connect() as conn:
            conn.execute("DELETE FROM active_paper_strategies WHERE active_strategy_id = ?", [strategy_id])
            frame = pd.DataFrame([payload])
            columns = ", ".join(frame.columns)
            conn.register("active_strategy_df", frame)
            conn.execute(f"INSERT INTO active_paper_strategies ({columns}) SELECT {columns} FROM active_strategy_df")
            conn.unregister("active_strategy_df")

    def get_active_paper_strategy(self, active_strategy_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM active_paper_strategies WHERE active_strategy_id = ? ORDER BY created_at DESC LIMIT 1",
                [active_strategy_id],
            ).fetchone()
            if row is None:
                return None
            columns = [item[0] for item in conn.description]
        return dict(zip(columns, row, strict=False))

    def list_active_paper_strategies(self, status: str | None = None) -> pd.DataFrame:
        where_clause = "WHERE status = ?" if status else ""
        parameters: list[Any] = [status] if status else []
        with self.connect() as conn:
            return conn.execute(
                f"""
                SELECT *
                FROM active_paper_strategies
                {where_clause}
                ORDER BY created_at DESC
                """,
                parameters,
            ).df()

    def insert_active_paper_strategy_event(self, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            frame = pd.DataFrame([payload])
            columns = ", ".join(frame.columns)
            conn.register("active_strategy_event_df", frame)
            conn.execute(f"INSERT INTO active_paper_strategy_events ({columns}) SELECT {columns} FROM active_strategy_event_df")
            conn.unregister("active_strategy_event_df")

    def replace_forward_engine_events(self, active_strategy_id: str, events: pd.DataFrame) -> None:
        engine_event_types = ["engine_info", "order_created", "order_filled", "position_closed", "data_skip", "update_summary"]
        placeholders = ", ".join(["?"] * len(engine_event_types))
        with self.connect() as conn:
            conn.execute(
                f"DELETE FROM active_paper_strategy_events WHERE active_strategy_id = ? AND event_type IN ({placeholders})",
                [active_strategy_id, *engine_event_types],
            )
            if not events.empty:
                conn.register("active_strategy_engine_events_df", events)
                conn.execute("INSERT INTO active_paper_strategy_events SELECT * FROM active_strategy_engine_events_df")
                conn.unregister("active_strategy_engine_events_df")

    def read_active_paper_strategy_events(self, active_strategy_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM active_paper_strategy_events WHERE active_strategy_id = ? ORDER BY created_at",
                [active_strategy_id],
            ).df()

    def replace_forward_paper_state(
        self,
        active_strategy_id: str,
        orders: pd.DataFrame,
        positions: pd.DataFrame,
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
    ) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM forward_paper_orders WHERE active_strategy_id = ?", [active_strategy_id])
            conn.execute("DELETE FROM forward_paper_positions WHERE active_strategy_id = ?", [active_strategy_id])
            conn.execute("DELETE FROM forward_paper_trades WHERE active_strategy_id = ?", [active_strategy_id])
            conn.execute("DELETE FROM forward_paper_equity_curve WHERE active_strategy_id = ?", [active_strategy_id])
            for table_name, frame_name, frame in [
                ("forward_paper_orders", "forward_orders_df", orders),
                ("forward_paper_positions", "forward_positions_df", positions),
                ("forward_paper_trades", "forward_trades_df", trades),
                ("forward_paper_equity_curve", "forward_equity_df", equity_curve),
            ]:
                if frame.empty:
                    continue
                conn.register(frame_name, frame)
                conn.execute(f"INSERT INTO {table_name} SELECT * FROM {frame_name}")
                conn.unregister(frame_name)

    def read_forward_paper_orders(self, active_strategy_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM forward_paper_orders WHERE active_strategy_id = ? ORDER BY created_at, planned_fill_date",
                [active_strategy_id],
            ).df()

    def read_forward_paper_positions(self, active_strategy_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM forward_paper_positions WHERE active_strategy_id = ? ORDER BY entry_date, ticker",
                [active_strategy_id],
            ).df()

    def read_forward_paper_trades(self, active_strategy_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM forward_paper_trades WHERE active_strategy_id = ? ORDER BY exit_date, ticker",
                [active_strategy_id],
            ).df()

    def read_forward_paper_equity_curve(self, active_strategy_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM forward_paper_equity_curve WHERE active_strategy_id = ? ORDER BY timestamp",
                [active_strategy_id],
            ).df()

    def scanner_history_summary(self) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT
                    DATE(s.created_at) AS snapshot_date,
                    COUNT(DISTINCT s.snapshot_id) AS snapshot_count,
                    SUM(CASE WHEN r.signal_type = 'new_buy_signal' THEN 1 ELSE 0 END) AS new_buy_count,
                    SUM(CASE WHEN r.signal_type = 'exit_signal' THEN 1 ELSE 0 END) AS exit_count
                FROM scanner_snapshots s
                LEFT JOIN scanner_snapshot_results r ON s.snapshot_id = r.snapshot_id
                GROUP BY 1
                ORDER BY 1
                """
            ).df()

    def scanner_history_by_ticker(self) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT ticker, COUNT(*) AS signal_count
                FROM scanner_snapshot_results
                GROUP BY 1
                ORDER BY signal_count DESC, ticker
                """
            ).df()

    def scanner_history_by_strategy_quality(self) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT
                    strategy_name,
                    SUM(CASE WHEN signal_quality_label = 'High quality' THEN 1 ELSE 0 END) AS high_quality_count,
                    SUM(CASE WHEN signal_quality_label IN ('Low quality', 'Ignore') THEN 1 ELSE 0 END) AS low_quality_count
                FROM scanner_snapshot_results
                GROUP BY 1
                ORDER BY high_quality_count DESC, low_quality_count DESC, strategy_name
                """
            ).df()

    def list_sweep_runs(self, limit: int = 50, tag: str | None = None) -> pd.DataFrame:
        tag_filter = f"WHERE lower(coalesce(sr.tags, '')) LIKE lower(?)" if tag else ""
        parameters: list[Any] = [f"%{tag}%"] if tag else []
        parameters.append(limit)
        with self.connect() as conn:
            return conn.execute(
                f"""
                SELECT
                    sr.sweep_id,
                    sr.created_at,
                    sr.strategy_name,
                    sr.tickers,
                    sr.start_date,
                    sr.end_date,
                    sr.benchmark_symbol,
                    sr.price_mode,
                    sr.notes,
                    sr.tags,
                    COUNT(res.sweep_result_id) AS parameter_combinations,
                    MAX(res.cagr) AS best_cagr,
                    MEDIAN(res.cagr) AS median_cagr,
                    MIN(res.cagr) AS worst_cagr,
                    AVG(CASE WHEN res.total_return > 0 THEN 1.0 ELSE 0.0 END) AS percent_profitable,
                    AVG(CASE WHEN res.beats_benchmark_flag THEN 1.0 ELSE 0.0 END) AS percent_beating_benchmark
                FROM sweep_runs sr
                LEFT JOIN sweep_results res ON sr.sweep_id = res.sweep_id
                {tag_filter}
                GROUP BY 1,2,3,4,5,6,7,8,9,10
                ORDER BY sr.created_at DESC
                LIMIT ?
                """,
                parameters,
            ).df()

    def get_sweep_run(self, sweep_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM sweep_runs WHERE sweep_id = ? ORDER BY created_at DESC LIMIT 1", [sweep_id]).fetchone()
            if row is None:
                return None
            columns = [item[0] for item in conn.description]
        return dict(zip(columns, row, strict=False))

    def read_sweep_results(self, sweep_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT *
                FROM sweep_results
                WHERE sweep_id = ?
                ORDER BY cagr DESC, total_return DESC
                """,
                [sweep_id],
            ).df()

    def read_sweep_parameters(self, sweep_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM sweep_parameters WHERE sweep_id = ? ORDER BY parameter_name", [sweep_id]).df()

    def list_backtest_runs(self, limit: int = 25, tag: str | None = None) -> pd.DataFrame:
        where_clause = "WHERE lower(coalesce(tags, '')) LIKE lower(?)" if tag else ""
        parameters: list[Any] = [f"%{tag}%"] if tag else []
        parameters.append(limit)
        with self.connect() as conn:
            return conn.execute(
                f"""
                SELECT run_id, strategy_name, symbols_csv, start_date, end_date, created_at, total_return, cagr,
                       max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, win_rate, profit_factor,
                       exposure_pct, number_of_trades, benchmark_symbol, price_mode, excess_cagr, notes, tags
                FROM backtest_runs
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                parameters,
            ).df()

    def get_backtest_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM backtest_runs WHERE run_id = ? ORDER BY created_at DESC LIMIT 1", [run_id]).fetchone()
            if row is None:
                return None
            columns = [item[0] for item in conn.description]
        return dict(zip(columns, row, strict=False))

    def read_backtest_trades(self, run_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM backtest_trades WHERE run_id = ? ORDER BY entry_timestamp", [run_id]).df()

    def read_backtest_equity_curve(self, run_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM backtest_equity_curve WHERE run_id = ? ORDER BY timestamp", [run_id]).df()

    def read_backtest_benchmark_curve(self, run_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM backtest_benchmark_curve WHERE run_id = ? ORDER BY timestamp", [run_id]).df()

    def compare_backtest_runs(self, run_ids: list[str]) -> pd.DataFrame:
        if not run_ids:
            return pd.DataFrame()
        placeholders = ", ".join(["?"] * len(run_ids))
        with self.connect() as conn:
            return conn.execute(
                f"""
                SELECT run_id, strategy_name, symbols_csv, start_date, end_date, created_at, total_return, cagr,
                       max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, win_rate, profit_factor,
                       number_of_trades, exposure_pct, benchmark_symbol, excess_cagr, price_mode, notes, tags
                FROM backtest_runs
                WHERE run_id IN ({placeholders})
                ORDER BY created_at DESC
                """,
                run_ids,
            ).df()

    def replace_backtest_benchmark_curve(self, benchmark_curve: pd.DataFrame, run_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM backtest_benchmark_curve WHERE run_id = ?", [run_id])
            if not benchmark_curve.empty:
                conn.register("benchmark_df", benchmark_curve)
                conn.execute("INSERT INTO backtest_benchmark_curve SELECT * FROM benchmark_df")
                conn.unregister("benchmark_df")

    def get_research_dashboard_rows(self) -> pd.DataFrame:
        from trading_lab.backtest.qualification import evaluate_options_overlay_candidate
        from trading_lab.backtest.robustness import parameter_stability_summary

        runs = self.list_backtest_runs(limit=1000)
        if runs.empty:
            return runs

        robustness_rows: list[dict[str, Any]] = []
        diagnostics_rows: list[dict[str, Any]] = []
        train_test_rows: list[dict[str, Any]] = []
        walk_rows: list[dict[str, Any]] = []
        audit_flags: list[dict[str, Any]] = []
        regime_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        saved_sweeps = self.list_sweep_runs(limit=1000)
        sweep_results_cache: dict[str, pd.DataFrame] = {}

        for run_id in runs["run_id"].tolist():
            saved_run = self.get_backtest_run(run_id) or {}
            robust = self.read_robustness_score(run_id)
            robustness_score = 0
            if not robust.empty:
                row = robust.iloc[0]
                robustness_score = int(row["score"])
                robustness_rows.append({"run_id": run_id, "robustness_score": robustness_score, "robustness_label": row["label"]})
            diag = self.read_benchmark_diagnostics(run_id)
            if not diag.empty:
                row = diag.iloc[0]
                diagnostics_rows.append(
                    {
                        "run_id": run_id,
                        "benchmark_diag_status": row["status"],
                        "benchmark_warning_count": len(json.loads(row["warnings_json"] or "[]")),
                    }
                )
            train_test = self.read_train_test_summary(run_id)
            if not train_test.empty:
                degradation = json.loads(train_test.iloc[0]["degradation_json"] or "{}")
                train_test_rows.append({"run_id": run_id, "train_test_cagr_degradation": float(degradation.get("CAGR", 0.0) or 0.0)})
            walk_runs = self.read_walk_forward_runs(run_id)
            if not walk_runs.empty:
                summary = json.loads(walk_runs.iloc[0]["summary_json"] or "{}")
                walk_rows.append(
                    {
                        "run_id": run_id,
                        "walk_forward_profitable_pct": float(summary.get("profitable_test_fold_pct", 0.0) or 0.0),
                        "walk_forward_consistency": float(summary.get("consistency_score", 0.0) or 0.0),
                    }
                )
            audit = self.read_audit_results(run_id)
            if not audit.empty:
                messages = " ".join(audit["message"].astype(str).tolist()).lower()
                audit_flags.append(
                    {
                        "run_id": run_id,
                        "high_profit_concentration": int("one trade" in messages or "small cluster of trades" in messages),
                        "regime_dependence_flag": int("regime" in messages),
                        "too_few_trades_flag": int("trade count is low" in messages or "too few trades" in messages),
                    }
                )
            regimes = self.read_regime_metrics(run_id)
            if not regimes.empty:
                trend = regimes[regimes["regime_type"] == "trend"]
                spread = float(trend["cagr"].max() - trend["cagr"].min()) if len(trend) >= 2 else 0.0
                regime_rows.append({"run_id": run_id, "regime_cagr_spread": spread})
            train_test_payload: dict[str, Any] | None = None
            if not train_test.empty:
                train_test_payload = {"degradation": json.loads(train_test.iloc[0]["degradation_json"] or "{}")}
            walk_payload: dict[str, Any] | None = None
            if not walk_runs.empty:
                walk_payload = json.loads(walk_runs.iloc[0]["summary_json"] or "{}")
            concentration = {}
            trades = self.read_backtest_trades(run_id)
            if not trades.empty:
                total_profit = float(trades["pnl"].sum())
                if total_profit > 0:
                    top_pnl = trades.sort_values("pnl", ascending=False)["pnl"]
                    concentration = {
                        "best_trade_profit_share": float(top_pnl.head(1).sum() / total_profit),
                        "top_5_profit_share": float(top_pnl.head(5).sum() / total_profit),
                    }
            parameter_stability = None
            strategy_name = str(saved_run.get("strategy_name") or "")
            matching_sweeps = saved_sweeps[saved_sweeps["strategy_name"] == strategy_name] if not saved_sweeps.empty else pd.DataFrame()
            if not matching_sweeps.empty:
                latest_sweep_id = matching_sweeps.iloc[0]["sweep_id"]
                if latest_sweep_id not in sweep_results_cache:
                    sweep_results_cache[latest_sweep_id] = self.read_sweep_results(latest_sweep_id)
                latest_results = sweep_results_cache[latest_sweep_id]
                if not latest_results.empty:
                    parameter_stability = parameter_stability_summary(
                        latest_results.rename(columns={"cagr": "CAGR", "max_drawdown": "Max Drawdown", "total_return": "Total Return"})
                    )
            assessment = evaluate_options_overlay_candidate(
                {
                    "Number of Trades": saved_run.get("number_of_trades", 0),
                    "CAGR": saved_run.get("cagr", 0.0),
                    "Excess CAGR": saved_run.get("excess_cagr", 0.0),
                    "Max Drawdown": saved_run.get("max_drawdown", 0.0),
                    "Benchmark Max Drawdown": saved_run.get("benchmark_max_drawdown", 0.0),
                    "Beta": saved_run.get("beta", 0.0),
                    "Exposure %": saved_run.get("exposure_pct", 0.0),
                },
                robustness_score=robustness_score,
                concentration=concentration,
                train_test_summary=train_test_payload,
                walk_forward_summary=walk_payload,
                parameter_stability=parameter_stability,
            )
            candidate_rows.append(
                {
                    "run_id": run_id,
                    "options_candidate_flag": int(assessment.flag),
                    "options_candidate_label": assessment.label,
                }
            )

        for extra in [robustness_rows, diagnostics_rows, train_test_rows, walk_rows, audit_flags, regime_rows, candidate_rows]:
            if extra:
                runs = runs.merge(pd.DataFrame(extra), on="run_id", how="left")

        if "robustness_score" not in runs.columns:
            runs["robustness_score"] = 0
        if "high_profit_concentration" not in runs.columns:
            runs["high_profit_concentration"] = 0
        if "regime_dependence_flag" not in runs.columns:
            runs["regime_dependence_flag"] = 0
        if "too_few_trades_flag" not in runs.columns:
            runs["too_few_trades_flag"] = (runs["number_of_trades"] < 10).astype(int)
        if "train_test_cagr_degradation" not in runs.columns:
            runs["train_test_cagr_degradation"] = 0.0
        if "walk_forward_consistency" not in runs.columns:
            runs["walk_forward_consistency"] = 0.0
        if "walk_forward_profitable_pct" not in runs.columns:
            runs["walk_forward_profitable_pct"] = 0.0
        if "options_candidate_flag" not in runs.columns:
            runs["options_candidate_flag"] = 0
        if "options_candidate_label" not in runs.columns:
            runs["options_candidate_label"] = "Not ready"

        runs["robustness_score"] = runs["robustness_score"].fillna(0)
        runs["high_profit_concentration"] = runs["high_profit_concentration"].fillna(0).astype(int)
        runs["regime_dependence_flag"] = runs["regime_dependence_flag"].fillna(0).astype(int)
        runs["too_few_trades_flag"] = runs["too_few_trades_flag"].fillna((runs["number_of_trades"] < 10).astype(int)).astype(int)
        runs["options_candidate_flag"] = runs["options_candidate_flag"].fillna(0).astype(int)
        runs["options_candidate_label"] = runs["options_candidate_label"].fillna("Not ready")
        runs["underperformed_benchmark_flag"] = (runs["excess_cagr"].fillna(0.0) < 0).astype(int)
        runs["poor_train_test_flag"] = (runs["train_test_cagr_degradation"].fillna(0.0) < -0.05).astype(int)
        runs["poor_walk_forward_flag"] = (
            (runs["walk_forward_consistency"].fillna(0.0) < 0.5)
            | (runs["walk_forward_profitable_pct"].fillna(0.0) < 0.5)
        ).astype(int)
        return runs
