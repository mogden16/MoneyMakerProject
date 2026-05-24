from __future__ import annotations

from pathlib import Path

from trading_lab.data.database import TradingLabDatabase


def test_database_reuses_shared_connection_per_path(tmp_path: Path, monkeypatch):
    db_path = str(tmp_path / "shared.duckdb")
    TradingLabDatabase.close_shared_connection(db_path)

    calls: list[str] = []
    original_connect = __import__("trading_lab.data.database", fromlist=["duckdb"]).duckdb.connect

    def tracking_connect(path, *args, **kwargs):
        calls.append(str(path))
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr("trading_lab.data.database.duckdb.connect", tracking_connect)

    db_one = TradingLabDatabase(db_path)
    db_two = TradingLabDatabase(db_path)

    with db_one.connect() as conn_one:
        with db_two.connect() as conn_two:
            assert conn_one is conn_two

    assert calls.count(db_path) == 1
    TradingLabDatabase.close_shared_connection(db_path)
