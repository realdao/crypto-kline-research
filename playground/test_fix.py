"""Test: executemany vs raw SQL VALUES vs COPY from DataFrame."""

import time, os, duckdb

path = ".data/crypto/_bench_final.db"

rows = [(f"SYM{i}", 1685577600000 + i * 86400000, 27000.0, 27500.0, 26800.0, 27200.0, 50000.0)
        for i in range(1099)]

# 1. Baseline: executemany + BEGIN + COMMIT
if os.path.exists(path): os.unlink(path)
c = duckdb.connect(path)
c.execute("CREATE TABLE kline (symbol VARCHAR, open_time BIGINT, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, PRIMARY KEY (symbol, open_time))")
t0 = time.perf_counter()
c.execute("BEGIN")
c.executemany("INSERT OR IGNORE INTO kline VALUES (?,?,?,?,?,?,?)", rows)
c.execute("COMMIT")
t1 = time.perf_counter()
cnt = c.execute("SELECT COUNT(*) FROM kline").fetchone()[0]
c.close()
print(f"executemany+B+C                : {t1-t0:.3f}s  ({cnt})")

# 2. Single SQL VALUES clause
if os.path.exists(path): os.unlink(path)
c = duckdb.connect(path)
c.execute("CREATE TABLE kline (symbol VARCHAR, open_time BIGINT, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, PRIMARY KEY (symbol, open_time))")
placeholders = ",".join(["(?,?,?,?,?,?,?)"] * len(rows))
flat = [v for r in rows for v in r]
t0 = time.perf_counter()
c.execute(f"INSERT OR IGNORE INTO kline VALUES {placeholders}", flat)
t1 = time.perf_counter()
cnt = c.execute("SELECT COUNT(*) FROM kline").fetchone()[0]
c.close()
print(f"SQL VALUES (one call)          : {t1-t0:.3f}s  ({cnt})")

# 3. Raw SQL text construction (no param binding)
if os.path.exists(path): os.unlink(path)
c = duckdb.connect(path)
c.execute("CREATE TABLE kline (symbol VARCHAR, open_time BIGINT, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, PRIMARY KEY (symbol, open_time))")
def escape(v):
    if isinstance(v, str): return f"'{v}'"
    if isinstance(v, float): return str(v)
    return str(v)
values_str = ",".join(f"('{s}',{ot},{o},{h},{l},{cl},{v})" for s,ot,o,h,l,cl,v in rows)
t0 = time.perf_counter()
c.execute(f"INSERT OR IGNORE INTO kline VALUES {values_str}")
t1 = time.perf_counter()
cnt = c.execute("SELECT COUNT(*) FROM kline").fetchone()[0]
c.close()
print(f"SQL VALUES (literal inline)    : {t1-t0:.3f}s  ({cnt})")

os.unlink(path)
