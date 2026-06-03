"""DuckDB insert: compare with/without PRIMARY KEY."""

import time, os, duckdb

DB_PATH = ".data/crypto/_bench.db"

rows_1k = [(f"SYM{i}", 1685577600000 + i * 86400000, 27000.0, 27500.0, 26800.0, 27200.0, 50000.0) for i in range(1000)]
rows_10k = rows_1k * 10

def bench(label, create_sql, insert_fn, n_rows):
    if os.path.exists(DB_PATH):
        os.unlink(DB_PATH)
    c = duckdb.connect(DB_PATH)
    c.execute(create_sql)
    t0 = time.perf_counter()
    insert_fn(c)
    t1 = time.perf_counter()
    cnt = c.execute("SELECT COUNT(*) FROM kline").fetchone()[0]
    c.close()
    per_row = 1000 * (t1 - t0) / n_rows
    print(f"  {label:<40} {t1-t0:>7.3f}s  ({cnt:>5} rows)  {per_row:>6.1f}ms/row")

create_pk = """CREATE TABLE kline (symbol VARCHAR, open_time BIGINT, open DOUBLE,
               high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
               PRIMARY KEY (symbol, open_time))"""
create_no_pk = """CREATE TABLE kline (symbol VARCHAR, open_time BIGINT, open DOUBLE,
                  high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE)"""

# Test 1k
print("=== 1,000 rows ===")
bench("PK + executemany", create_pk,
      lambda c: c.executemany("INSERT OR IGNORE INTO kline VALUES (?,?,?,?,?,?,?)", rows_1k), 1000)

bench("NO_PK + executemany", create_no_pk,
      lambda c: c.executemany("INSERT INTO kline VALUES (?,?,?,?,?,?,?)", rows_1k), 1000)

bench("NO_PK + BEGIN+executemany+COMMIT", create_no_pk,
      lambda c: (c.execute("BEGIN"), c.executemany("INSERT INTO kline VALUES (?,?,?,?,?,?,?)", rows_1k), c.execute("COMMIT")), 1000)

# Test 10k
print("\n=== 10,000 rows ===")
bench("PK + B+C+executemany", create_pk,
      lambda c: (c.execute("BEGIN"), c.executemany("INSERT OR IGNORE INTO kline VALUES (?,?,?,?,?,?,?)", rows_10k), c.execute("COMMIT")), 10000)

bench("NO_PK + B+C+executemany", create_no_pk,
      lambda c: (c.execute("BEGIN"), c.executemany("INSERT INTO kline VALUES (?,?,?,?,?,?,?)", rows_10k), c.execute("COMMIT")), 10000)

bench("NO_PK + executemany (no B+C)", create_no_pk,
      lambda c: c.executemany("INSERT INTO kline VALUES (?,?,?,?,?,?,?)", rows_10k), 10000)

# Test: duplicate insert (important: actual download re-inserts existing data)
if os.path.exists(DB_PATH):
    os.unlink(DB_PATH)
c = duckdb.connect(DB_PATH)
c.execute(create_no_pk)
c.execute("BEGIN")
c.executemany("INSERT INTO kline VALUES (?,?,?,?,?,?,?)", rows_10k)
c.execute("COMMIT")

# Now insert same data again (simulating re-download)
t0 = time.perf_counter()
c.execute("BEGIN")
c.executemany("INSERT INTO kline VALUES (?,?,?,?,?,?,?)", rows_10k)
c.execute("COMMIT")
t1 = time.perf_counter()
cnt = c.execute("SELECT COUNT(*) FROM kline").fetchone()[0]
c.close()
print(f"\n  {'NO_PK INSERT dups 10k':<40} {t1-t0:>7.3f}s  ({cnt:>5} rows)  {1000*(t1-t0)/10000:.1f}ms/row")

os.unlink(DB_PATH)
