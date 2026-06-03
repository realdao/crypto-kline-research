"""DuckDB K-line storage layer.

Usage:
    store = KlineStore("path/to/foo.db")
    store.insert([...])
    candles = store.query("BTCUSDT", start_ms, end_ms)
    store.close()
"""

import duckdb


class KlineStore:
    def __init__(self, path: str):
        self.conn = duckdb.connect(path)
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
        if not rows:
            return 0

        # Use a single SQL VALUES clause — much faster than executemany on Windows.
        placeholders = ", ".join(["(?, ?, ?, ?, ?, ?, ?)"] * len(rows))
        flat_values = []
        for r in rows:
            flat_values.extend([r["symbol"], r["open_time"], r["open"],
                                r["high"], r["low"], r["close"], r["volume"]])

        before = self.count()
        self.conn.execute("BEGIN")
        self.conn.execute(
            f"INSERT OR IGNORE INTO kline VALUES {placeholders}",
            flat_values,
        )
        self.conn.execute("COMMIT")
        after = self.count()
        return after - before

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

    def symbol_info(self) -> list[tuple]:
        """Return [(symbol, count, first_ts, last_ts), ...] for every symbol."""
        return self.conn.execute("""
            SELECT symbol, COUNT(*), MIN(open_time), MAX(open_time)
            FROM kline GROUP BY symbol ORDER BY symbol
        """).fetchall()

    def count(self, symbol: str | None = None) -> int:
        if symbol:
            return self.conn.execute(
                "SELECT COUNT(*) FROM kline WHERE symbol = ?", [symbol]
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM kline").fetchone()[0]

    def close(self):
        self.conn.close()
