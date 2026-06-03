"""Diagnose: where does the slowness come from?

Breaks down API fetch + DB insert timings step by step.
"""

import time, os, sys
from datetime import datetime, timezone

# Ensure project root is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.binance_client import create_session, fetch_klines
from core.data import KlineStore

DB_PATH = ".data/crypto/_bench_btc-1d.db"


def main():
    if os.path.exists(DB_PATH):
        os.unlink(DB_PATH)

    start_ms = int(datetime(2023, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # --- Step 1: API fetch ---
    print("1. Fetching from Binance ...", flush=True)
    t0 = time.perf_counter()
    session = create_session()
    candles = fetch_klines("BTCUSDT", "1d", start_ms, end_ms, session)
    t1 = time.perf_counter()
    print(f"   Done: {len(candles)} rows in {t1 - t0:.2f}s", flush=True)

    # --- Step 2: DB init ---
    print("2. Opening DB ...", flush=True)
    t0 = time.perf_counter()
    store = KlineStore(DB_PATH)
    t1 = time.perf_counter()
    print(f"   Done: {t1 - t0:.3f}s", flush=True)

    # --- Step 3: Count before insert ---
    print("3. COUNT(*) before insert ...", flush=True)
    t0 = time.perf_counter()
    cnt = store.count()
    t1 = time.perf_counter()
    print(f"   Count={cnt}, took {t1 - t0:.4f}s", flush=True)

    # --- Step 4: Insert ---
    print("4. Inserting ...", flush=True)
    t0 = time.perf_counter()
    inserted = store.insert(candles)
    t1 = time.perf_counter()
    print(f"   Inserted {inserted} rows in {t1 - t0:.2f}s", flush=True)

    # --- Step 5: Count after insert ---
    print("5. COUNT(*) after insert ...", flush=True)
    t0 = time.perf_counter()
    cnt = store.count()
    t1 = time.perf_counter()
    print(f"   Count={cnt}, took {t1 - t0:.4f}s", flush=True)

    # --- Step 6: Query test ---
    print("6. Query ...", flush=True)
    t0 = time.perf_counter()
    rows = store.query("BTCUSDT")
    t1 = time.perf_counter()
    print(f"   Got {len(rows)} rows in {t1 - t0:.3f}s", flush=True)

    store.close()
    os.unlink(DB_PATH)

    # --- Summary ---
    print(f"\n=== Summary ===", flush=True)
    print(f"API fetch: ~{len(candles)/1000:.1f}k rows in one batch", flush=True)
    print(f"If the total took minutes, the bottleneck is likely:", flush=True)
    print(f"  - DuckDB executemany on file-based DB (try :memory: or batched COPY)", flush=True)
    print(f"  - Or file-locking if another process has the DB open", flush=True)


if __name__ == "__main__":
    main()
