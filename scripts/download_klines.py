"""Download kline data from Binance into local DuckDB store."""

import sys
from datetime import datetime, timezone

from core.data import KlineStore
from core.binance_client import create_session, fetch_klines, fetch_symbols


def parse_date(s: str) -> datetime:
    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s}")


def main():
    if len(sys.argv) < 4:
        print("Usage: python -m scripts.download_klines <symbol> <interval> <start_date> [end_date]")
        print("  symbol:  BTCUSDT, ETHUSDT, or 'ALL' for all USDT pairs")
        print("  interval: 1d, 4h, 1h")
        print("  dates:    2023-06-01")
        return

    symbol = sys.argv[1]
    interval = sys.argv[2]
    start_str = sys.argv[3]
    end_str = sys.argv[4] if len(sys.argv) > 4 else "now"

    start_dt = parse_date(start_str)
    if end_str.lower() == "now":
        end_dt = datetime.now(timezone.utc)
    else:
        end_dt = parse_date(end_str)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    session = create_session()
    db_path = f".data/crypto/binance-{interval}.db"
    store = KlineStore(db_path)

    if symbol.upper() == "ALL":
        symbols = fetch_symbols(session)
        print(f"Found {len(symbols)} USDT spot pairs")
    else:
        symbols = [symbol]

    for i, sym in enumerate(symbols):
        existing = store.count(symbol=sym)
        print(f"[{i+1}/{len(symbols)}] {sym} (stored: {existing})")

        try:
            candles = fetch_klines(sym, interval, start_ms, end_ms, session)
            if candles:
                inserted = store.insert(candles)
                print(f"  Downloaded {len(candles)} klines, inserted {inserted} new rows")
            else:
                print("  No data")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nDone. Total rows in DB: {store.count()}")
    store.close()


if __name__ == "__main__":
    main()
