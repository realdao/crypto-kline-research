"""Inspect a KlineStore database: symbols, row counts, time ranges.

Usage:
    python -m scripts.inspect .data/crypto/binance-1d.db
"""

import sys
from datetime import datetime, timezone

from core.data import KlineStore


def _ts_to_str(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.inspect <db_path>")
        return

    db_path = sys.argv[1]
    store = KlineStore(db_path)

    symbols_info = store.symbol_info()
    total_rows = store.count()

    print(f"DB: {db_path}")
    print(f"Symbols: {len(symbols_info)}")
    print(f"Total rows: {total_rows:,}")
    print()
    print(f"{'Symbol':<16} {'Count':>8} {'First':<20} {'Last':<20}")
    print("-" * 70)

    for sym, cnt, first_ts, last_ts in symbols_info:
        print(f"{sym:<16} {cnt:>8,} {_ts_to_str(first_ts):<20} {_ts_to_str(last_ts):<20}")

    store.close()


if __name__ == "__main__":
    main()
