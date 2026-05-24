from __future__ import annotations

from pathlib import Path

from trading_lab.app import default_show_advanced_tools, get_primary_tab_labels
from trading_lab.data.database import TradingLabDatabase


def test_app_imports_and_primary_tabs():
    labels = get_primary_tab_labels()
    assert labels == ["SPY Workbench", "Forward Paper", "Research History", "Market Regime Report", "Data & Settings"]


def test_advanced_mode_defaults_off():
    assert default_show_advanced_tools() is False


def test_no_saved_data_edge_cases(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "empty_navigation.duckdb"))
    assert db.list_spy_strategy_search_runs(limit=10).empty
    assert db.list_active_paper_strategies().empty
    assert db.list_backtest_runs(limit=10).empty


def test_forward_paper_functions_remain_accessible(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "forward_access.duckdb"))
    assert hasattr(db, "list_active_paper_strategies")
    assert hasattr(db, "read_forward_paper_orders")
    assert db.list_active_paper_strategies().empty


def test_saved_research_functions_remain_accessible(tmp_path: Path):
    db = TradingLabDatabase(str(tmp_path / "research_access.duckdb"))
    assert hasattr(db, "list_spy_strategy_search_runs")
    assert hasattr(db, "list_backtest_runs")
    assert hasattr(db, "list_sweep_runs")
    assert db.list_spy_strategy_search_runs(limit=10).empty
