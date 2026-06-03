"""DuckDB K-line storage layer.

DB files: .data/crypto/{exchange}-{interval}.db
"""

from pathlib import Path
import duckdb

DATA_ROOT = Path(".data") / "crypto"


def _db_path(exchange: str, interval: str) -> Path:
    return DATA_ROOT / f"{exchange}-{interval}.db"


class KlineStore:
    def __init__(self, exchange: str, interval: str):
        self.path = _db_path(exchange, interval)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.path))
        self._init_table()

    def _init_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS kline (
                symbol      VARCHAR NOT NULL,
                open_time   BIGINT NOT NULL,
                open        DOUBLE NOT NULL,
                high        DOUBLE NOT NULL,
                low         DOUBLE NOT NULL,
                close       DOUBLE NOT NULL,
                volume      DOUBLE NOT NULL,
                PRIMARY KEY (symbol, open_time)
            )
        """)

    def insert(self, rows: list[dict]) -> int:
        existing = self.conn.execute("SELECT COUNT(*) FROM kline").fetchone()[0]
        self.conn.executemany(
            """INSERT OR IGNORE INTO kline
               (symbol, open_time, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(r["symbol"], r["open_time"], r["open"], r["high"],
              r["low"], r["close"], r["volume"]) for r in rows]
        )
        new_count = self.conn.execute("SELECT COUNT(*) FROM kline").fetchone()[0]
        return new_count - existing

    def query(self, symbol: str,
              start_ms: int | None = None, end_ms: int | None = None) -> list[dict]:
        sql = """SELECT symbol, open_time, open, high, low, close, volume
                 FROM kline WHERE symbol = ?"""
        params: list = [symbol]
        if start_ms is not None:
            sql += " AND open_time >= ?"
            params.append(start_ms)
        if end_ms is not None:
            sql += " AND open_time <= ?"
            params.append(end_ms)
        sql += " ORDER BY open_time"

        rows = self.conn.execute(sql, params).fetchall()
        return [
            {"symbol": r[0], "open_time": r[1], "open": r[2],
             "high": r[3], "low": r[4], "close": r[5], "volume": r[6]}
            for r in rows
        ]

    def get_symbols(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT symbol FROM kline ORDER BY symbol"
        ).fetchall()
        return [r[0] for r in rows]

    def count(self, symbol: str | None = None) -> int:
        if symbol:
            return self.conn.execute(
                "SELECT COUNT(*) FROM kline WHERE symbol = ?", [symbol]
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM kline").fetchone()[0]

    def close(self):
        self.conn.close()
