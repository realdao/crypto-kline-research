"""DuckDB K-line storage layer."""

from pathlib import Path
import duckdb


DB_PATH = Path("data") / "crypto" / "kline.db"


class KlineStore:
    def __init__(self, path: Path | str = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.path))
        self._init_table()

    def _init_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS kline (
                symbol      VARCHAR NOT NULL,
                exchange    VARCHAR NOT NULL,
                interval    VARCHAR NOT NULL,
                open_time   BIGINT NOT NULL,
                open        DOUBLE NOT NULL,
                high        DOUBLE NOT NULL,
                low         DOUBLE NOT NULL,
                close       DOUBLE NOT NULL,
                volume      DOUBLE NOT NULL,
                PRIMARY KEY (exchange, symbol, interval, open_time)
            )
        """)

    def insert(self, rows: list[dict]) -> int:
        """Insert klines, skip duplicates. Returns number of rows inserted."""
        existing = self.conn.execute("SELECT COUNT(*) FROM kline").fetchone()[0]
        self.conn.executemany(
            """INSERT OR IGNORE INTO kline
               (exchange, symbol, interval, open_time, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(r["exchange"], r["symbol"], r["interval"], r["open_time"],
              r["open"], r["high"], r["low"], r["close"], r["volume"])
             for r in rows]
        )
        new_count = self.conn.execute("SELECT COUNT(*) FROM kline").fetchone()[0]
        return new_count - existing

    def query(self, exchange: str, symbol: str, interval: str,
              start_ms: int | None = None, end_ms: int | None = None) -> list[dict]:
        """Query klines, ordered by open_time ascending."""
        sql = """SELECT symbol, exchange, interval, open_time, open, high, low, close, volume
                 FROM kline
                 WHERE exchange = ? AND symbol = ? AND interval = ?"""
        params: list = [exchange, symbol, interval]
        if start_ms is not None:
            sql += " AND open_time >= ?"
            params.append(start_ms)
        if end_ms is not None:
            sql += " AND open_time <= ?"
            params.append(end_ms)
        sql += " ORDER BY open_time"

        rows = self.conn.execute(sql, params).fetchall()
        return [
            {"symbol": r[0], "exchange": r[1], "interval": r[2],
             "open_time": r[3], "open": r[4], "high": r[5],
             "low": r[6], "close": r[7], "volume": r[8]}
            for r in rows
        ]

    def get_symbols(self, exchange: str | None = None) -> list[str]:
        """List all symbols in store."""
        if exchange:
            rows = self.conn.execute(
                "SELECT DISTINCT symbol FROM kline WHERE exchange = ? ORDER BY symbol",
                [exchange]
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT DISTINCT symbol FROM kline ORDER BY symbol"
            ).fetchall()
        return [r[0] for r in rows]

    def count(self, exchange: str | None = None, symbol: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM kline WHERE 1=1"
        params: list = []
        if exchange:
            sql += " AND exchange = ?"
            params.append(exchange)
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        return self.conn.execute(sql, params).fetchone()[0]

    def close(self):
        self.conn.close()
