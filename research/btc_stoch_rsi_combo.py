"""BTC Stoch RSI combo: weekly >90 & daily <10 — forward return stats.

Usage:
    uv run python -m research.btc_stoch_rsi_combo

Signal: daily Stoch RSI %K < 10 while the latest completed weekly bar's
Stoch RSI %K > 90 (deep pullback inside an extreme weekly uptrend).

Sections:
    1. Current status (latest daily/weekly Stoch RSI, recent weekly top fractal)
    2. Episode detail table (de-clustered signal days, forward returns)
    3. Summary vs baseline at 7d/14d/30d/60d/90d/180d/365d
    4. Threshold sensitivity grid
    5. 60-day drawdown distribution after weekly StochRSI>90 alone
    6. Weekly top-fractal (顶分型) filter comparison
"""

from datetime import datetime, timezone
from statistics import mean, median

from core.data import KlineStore
from core.metrics import calc_stoch_rsi

DAY_MS = 24 * 60 * 60 * 1000
WEEK_MS = 7 * DAY_MS
HORIZONS = [7, 14, 30, 60, 90, 180, 365]
MFE_MAE_HORIZONS = [30, 90]
FRACTAL_LOOKBACK = 3
DAILY_DB = ".data/crypto/binance-BTCUSDT-1d.db"
WEEKLY_DB = ".data/crypto/binance-BTCUSDT-1w.db"


def load(store: KlineStore, now_ms: int, interval_ms: int) -> list[dict]:
    candles = store.query("BTCUSDT")
    return [c for c in candles if c["open_time"] + interval_ms <= now_ms]


def forward_ret(candles: list[dict], i: int, days: int) -> float | None:
    if i + days >= len(candles):
        return None
    entry = candles[i]["close"]
    return (candles[i + days]["close"] - entry) / entry * 100


def fwd_mfe_mae(candles: list[dict], i: int, days: int) -> tuple:
    if i + days >= len(candles):
        return None, None
    entry = candles[i]["close"]
    window = candles[i + 1:i + days + 1]
    mfe = (max(c["high"] for c in window) - entry) / entry * 100
    mae = (min(c["low"] for c in window) - entry) / entry * 100
    return mfe, mae


def fwd_drawdown(candles: list[dict], i: int, days: int) -> tuple:
    """(peak-to-trough DD%, MAE from entry close) within `days` after day i."""
    if i + days >= len(candles):
        return None, None
    entry_close = candles[i]["close"]
    run_high = candles[i]["high"]
    min_low = candles[i]["low"]
    dd = 0.0
    for c in candles[i + 1:i + days + 1]:
        if c["high"] > run_high:
            run_high = c["high"]
        if c["low"] < min_low:
            min_low = c["low"]
        d = (c["low"] - run_high) / run_high * 100
        if d < dd:
            dd = d
    mae = (min_low - entry_close) / entry_close * 100
    return dd, mae


def pct(v: float | None) -> str:
    return "N/A" if v is None else f"{v:+.1f}%"


def percentile(vals: list[float], p: float) -> float:
    k = (len(vals) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def map_daily_to_weekly(daily: list[dict], weekly: list[dict],
                        wk: list[float | None]) -> list[float | None]:
    """For each daily bar, weekly Stoch RSI of last *completed* weekly bar."""
    starts = [c["open_time"] for c in weekly]
    out: list[float | None] = []
    wi = -1
    for c in daily:
        while wi + 1 < len(starts) and starts[wi + 1] + WEEK_MS <= c["open_time"]:
            wi += 1
        out.append(wk[wi] if wi >= 0 else None)
    return out


def calc_top_fractals(highs: list[float], lows: list[float]) -> list[bool]:
    """顶分型: middle bar's high AND low both higher than the two neighbours."""
    n = len(highs)
    frac = [False] * n
    for i in range(1, n - 1):
        if (highs[i] > highs[i - 1] and highs[i] > highs[i + 1]
                and lows[i] > lows[i - 1] and lows[i] > lows[i + 1]):
            frac[i] = True
    return frac


def map_daily_fractal(daily: list[dict], weekly: list[dict],
                      frac: list[bool], lookback: int) -> list[bool]:
    """Day t has a recent weekly top fractal if one sits in the last `lookback`
    completed weekly bars *before* the latest completed bar (needs confirmation)."""
    starts = [c["open_time"] for c in weekly]
    out = []
    wi = -1
    for c in daily:
        while wi + 1 < len(starts) and starts[wi + 1] + WEEK_MS <= c["open_time"]:
            wi += 1
        if wi < 1:
            out.append(False)
            continue
        lo = max(0, wi - lookback)
        out.append(any(frac[j] for j in range(lo, wi)))
    return out


def build_episodes(daily: list[dict], dk: list[float | None],
                   dwk: list[float | None], daily_under: float,
                   weekly_over: float,
                   flags: list[bool] | None = None) -> list[dict]:
    """De-cluster consecutive signal days; entry = first day of each run."""
    episodes = []
    for i in range(len(daily)):
        if dk[i] is None or dwk[i] is None:
            continue
        hit = dk[i] < daily_under and dwk[i] > weekly_over
        if flags is not None:
            hit = hit and flags[i]
        if not hit:
            continue
        date_s = datetime.fromtimestamp(
            daily[i]["open_time"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if episodes and i == episodes[-1]["end_idx"] + 1:
            episodes[-1]["end_idx"] = i
            episodes[-1]["end_date"] = date_s
        else:
            episodes.append({
                "idx": i,
                "end_idx": i,
                "date": date_s,
                "end_date": date_s,
                "dk": dk[i],
                "dwk": dwk[i],
            })
    return episodes


def print_current(daily: list[dict], dk: list[float | None],
                  weekly: list[dict], wk: list[float | None],
                  frac: list[bool], now_ms: int):
    print(f"\n{'=' * 100}")
    print("CURRENT STATUS (in-progress weekly bar included)")
    print(f"{'=' * 100}")
    last_d = datetime.fromtimestamp(daily[-1]["open_time"] / 1000, tz=timezone.utc)
    last_w = datetime.fromtimestamp(weekly[-1]["open_time"] / 1000, tz=timezone.utc)
    print(f"Data end  daily: {last_d.strftime('%Y-%m-%d')}   weekly: {last_w.strftime('%Y-%m-%d')}")
    print(f"Daily   Stoch RSI K (latest 3): "
          f"{[f'{v:.1f}' if v is not None else 'N/A' for v in dk[-3:]]}")
    print(f"Weekly  Stoch RSI K (latest 3): "
          f"{[f'{v:.1f}' if v is not None else 'N/A' for v in wk[-3:]]}")
    recent = []
    for j in range(max(1, len(weekly) - 6), len(weekly) - 1):
        wd = datetime.fromtimestamp(weekly[j]["open_time"] / 1000, tz=timezone.utc)
        recent.append(f"{wd.strftime('%Y-%m-%d')}{'*' if frac[j] else ''}")
    print(f"Recent completed weekly top fractals (last 6w, '*' = top fractal): {recent}")
    print(f"BTC last daily close: {daily[-1]['close']:,.0f}")


def print_episodes(episodes: list[dict], daily: list[dict]):
    print(f"\n{'=' * 170}")
    print(f"EPISODES - weekly StochRSI>90 & daily StochRSI<10 "
          f"(entry = close of first signal day)")
    print(f"{'=' * 170}")
    print(f"{'Date':<12} {'End':<12} {'Len':>3} {'dK':>6} {'wK':>6} "
          + " ".join(f"{'+%dd' % h:>8}" for h in HORIZONS)
          + f" {'MFE90':>8} {'MAE90':>8}")
    print("-" * 170)
    for e in episodes:
        i = e["idx"]
        fwd = {h: forward_ret(daily, i, h) for h in HORIZONS}
        mfe90, mae90 = fwd_mfe_mae(daily, i, 90)
        row = (f"{e['date']:<12} {e['end_date']:<12} "
               f"{e['end_idx'] - e['idx'] + 1:>3} {e['dk']:>6.1f} {e['dwk']:>6.1f} ")
        row += " ".join(f"{pct(fwd[h]):>8}" for h in HORIZONS)
        row += f" {pct(mfe90):>8} {pct(mae90):>8}"
        print(row)


def summarize(title: str, episodes: list[dict], daily: list[dict],
              baseline: dict):
    print(f"\n{'=' * 130}")
    print(f"SUMMARY - {title} (N={len(episodes)})")
    print(f"{'=' * 130}")
    print(f"{'Horizon':>8} {'N':>3} {'Avg':>8} {'Med':>8} {'Win%':>6} "
          f"{'MFE':>8} {'MAE':>8} | {'BaseN':>6} {'BaseAvg':>8} {'Excess':>8} "
          f"{'BaseWin':>8}")
    print("-" * 130)
    for h in HORIZONS:
        rets = []
        mfes, maes = [], []
        for e in episodes:
            r = forward_ret(daily, e["idx"], h)
            if r is not None:
                rets.append(r)
            if h in MFE_MAE_HORIZONS:
                m, a = fwd_mfe_mae(daily, e["idx"], h)
                if m is not None:
                    mfes.append(m)
                    maes.append(a)
        if not rets:
            print(f"{h:>3}d {'0':>3} {'N/A':>8}")
            continue
        b = baseline[h]
        excess = mean(rets) - b["avg"]
        win = sum(1 for v in rets if v > 0) / len(rets) * 100
        mfe_s = f"{mean(mfes):+.1f}%" if mfes else "N/A"
        mae_s = f"{mean(maes):+.1f}%" if maes else "N/A"
        print(f"{h:>3}d {len(rets):>3} {mean(rets):>+8.1f} {median(rets):>+8.1f} "
              f"{win:>5.0f}% {mfe_s:>8} {mae_s:>8} | {b['n']:>6} {b['avg']:>+8.1f} "
              f"{excess:>+8.1f} {b['win']:>7.0f}%")


def compute_baseline(daily: list[dict], dk: list[float | None],
                     dwk: list[float | None], signal_idx: set,
                     daily_under: float, weekly_over: float) -> dict:
    baseline = {}
    for h in HORIZONS:
        rets = []
        for i in range(len(daily)):
            if dk[i] is None or dwk[i] is None or i in signal_idx:
                continue
            if dk[i] < daily_under and dwk[i] > weekly_over:
                continue
            r = forward_ret(daily, i, h)
            if r is not None:
                rets.append(r)
        baseline[h] = {
            "n": len(rets),
            "avg": mean(rets),
            "win": sum(1 for v in rets if v > 0) / len(rets) * 100,
        }
    return baseline


def print_sensitivity(daily: list[dict], dk: list[float | None],
                      dwk: list[float | None]):
    print(f"\n{'=' * 100}")
    print("SENSITIVITY - episode count / avg 30d return (weekly-over x daily-under)")
    print(f"{'=' * 100}")
    header = f"{'wK>\\dK<':>10} "
    for d in [5.0, 10.0, 15.0, 20.0]:
        header += f"{'<%.0f' % d:>14}"
    print(header)
    for w in [95.0, 90.0, 85.0, 80.0]:
        row = f"{'>%.0f' % w:>10} "
        for d in [5.0, 10.0, 15.0, 20.0]:
            eps = build_episodes(daily, dk, dwk, d, w)
            rets = [forward_ret(daily, e["idx"], 30) for e in eps]
            rets = [r for r in rets if r is not None]
            if rets:
                row += f"{'%d %+.0f%%' % (len(rets), mean(rets)):>14}"
            else:
                row += f"{'0 N/A':>14}"
        print(row)
    print("Cell = N / avg +30d return from first signal day of each episode.")


def print_drawdown_dist(daily: list[dict], dk: list[float | None],
                        dwk: list[float | None]):
    eps = build_episodes(daily, dk, dwk, float("inf"), 90.0)
    rows = []
    for e in eps:
        dd, mae = fwd_drawdown(daily, e["idx"], 60)
        if dd is not None:
            rows.append({"date": e["date"], "dd": dd, "mae": mae,
                         "ret60": forward_ret(daily, e["idx"], 60)})
    rows.sort(key=lambda r: r["dd"])

    base_dds = []
    base_idx = {e["idx"] for e in eps}
    for i in range(len(daily)):
        if dk[i] is None or dwk[i] is None or i in base_idx:
            continue
        dd, _ = fwd_drawdown(daily, i, 60)
        if dd is not None:
            base_dds.append(dd)

    print(f"\n{'=' * 110}")
    print(f"60-DAY DRAWDOWN DISTRIBUTION after weekly StochRSI>90 "
          f"(N={len(rows)} de-clustered episodes, entry = first-day close)")
    print(f"{'=' * 110}")
    print(f"{'':>14} {'N':>4} {'Mean':>7} {'Med':>7} {'P10':>7} {'P25':>7} "
          f"{'P75':>7} {'P90':>7} {'Worst':>7}")
    dds = [r["dd"] for r in rows]
    maes = sorted(r["mae"] for r in rows)
    base_dds.sort()
    print(f"{'PeakDD60':>14} {len(dds):>4} {mean(dds):>+7.1f} {median(dds):>+7.1f} "
          f"{percentile(dds, 10):>+7.1f} {percentile(dds, 25):>+7.1f} "
          f"{percentile(dds, 75):>+7.1f} {percentile(dds, 90):>+7.1f} "
          f"{min(dds):>+7.1f}")
    print(f"{'MAE60':>14} {len(maes):>4} {mean(maes):>+7.1f} {median(maes):>+7.1f} "
          f"{percentile(maes, 10):>+7.1f} {percentile(maes, 25):>+7.1f} "
          f"{percentile(maes, 75):>+7.1f} {percentile(maes, 90):>+7.1f} "
          f"{min(maes):>+7.1f}")
    print(f"{'BaseDD60':>14} {len(base_dds):>4} {mean(base_dds):>+7.1f} "
          f"{median(base_dds):>+7.1f} {percentile(base_dds, 10):>+7.1f} "
          f"{percentile(base_dds, 25):>+7.1f} {percentile(base_dds, 75):>+7.1f} "
          f"{percentile(base_dds, 90):>+7.1f} {min(base_dds):>+7.1f}")
    print("\nExceedance probabilities (PeakDD60):")
    for t in [-5, -10, -20, -30, -40, -50]:
        n_le = sum(1 for v in dds if v <= t)
        print(f"  P(DD <= {t:>3}%) = {n_le / len(dds) * 100:>5.1f}%")

    print(f"\nWorst 10 episodes by peak drawdown:")
    print(f"{'Date':<12} {'PeakDD60':>9} {'MAE60':>8} {'Ret60':>8}")
    for r in rows[:10]:
        print(f"{r['date']:<12} {r['dd']:>+8.1f}% {r['mae']:>+7.1f}% "
              f"{pct(r['ret60']):>8}")


def print_fractal_compare(daily: list[dict], weekly: list[dict],
                          frac: list[bool], dk: list[float | None],
                          dwk: list[float | None], flags: list[bool]):
    print(f"\n{'=' * 120}")
    print("TOP-FRACTAL FILTER - combo (wK>90 & dK<10) with/without weekly top fractal")
    print(f"Top fractal: middle weekly bar's high AND low both higher than the two "
          f"neighbours; formed within last {FRACTAL_LOOKBACK} completed weekly bars")
    print(f"{'=' * 120}")
    print(f"{'Group':<12} {'N':>3} {'+7d':>8} {'+30d':>8} {'+60d':>8} {'+90d':>8} "
          f"{'+365d':>8} {'Win90':>7} {'MAE60avg':>9} {'MAE60worst':>11}")
    print("-" * 120)

    for label, fl in [("WithFrac", flags), ("NoFrac", [not f for f in flags])]:
        eps = build_episodes(daily, dk, dwk, 10.0, 90.0, fl)
        row = f"{label:<12} {len(eps):>3} "
        for h in [7, 30, 60, 90, 365]:
            rets = [forward_ret(daily, e["idx"], h) for e in eps]
            rets = [r for r in rets if r is not None]
            row += f"{mean(rets):>+8.1f} " if rets else f"{'N/A':>8} "
        rets90 = [forward_ret(daily, e["idx"], 90) for e in eps]
        rets90 = [r for r in rets90 if r is not None]
        win90 = sum(1 for v in rets90 if v > 0) / len(rets90) * 100 if rets90 else float("nan")
        maes60 = [fwd_drawdown(daily, e["idx"], 60)[1] for e in eps]
        maes60 = [m for m in maes60 if m is not None]
        row += f"{win90:>6.0f}% "
        row += f"{mean(maes60):>+8.1f}% " if maes60 else f"{'N/A':>9} "
        row += f"{min(maes60):>+10.1f}%" if maes60 else f"{'N/A':>11}"
        print(row)

    print("\nSensitivity - fractal lookback (completed weeks scanned):")
    print(f"{'Lookback':>10} {'N':>4} {'+30d_avg':>9} {'+90d_avg':>9} "
          f"{'Win90':>7} {'MAE60_avg':>10}")
    for lb in [1, 2, 3, 4, 8]:
        fl = map_daily_fractal(daily, weekly, frac, lb)
        eps = build_episodes(daily, dk, dwk, 10.0, 90.0, fl)
        rets30 = [forward_ret(daily, e["idx"], 30) for e in eps]
        rets30 = [r for r in rets30 if r is not None]
        rets90 = [forward_ret(daily, e["idx"], 90) for e in eps]
        rets90 = [r for r in rets90 if r is not None]
        win90 = sum(1 for v in rets90 if v > 0) / len(rets90) * 100 if rets90 else float("nan")
        maes60 = [fwd_drawdown(daily, e["idx"], 60)[1] for e in eps]
        maes60 = [m for m in maes60 if m is not None]
        r30 = f"{mean(rets30):+.1f}" if rets30 else "N/A"
        r90 = f"{mean(rets90):+.1f}" if rets90 else "N/A"
        w90 = f"{win90:.0f}%" if rets90 else "N/A"
        m60 = f"{mean(maes60):+.1f}%" if maes60 else "N/A"
        print(f"{lb:>6}W {len(eps):>4} {r30:>9} {r90:>9} {w90:>7} {m60:>10}")

    eps = build_episodes(daily, dk, dwk, 10.0, 90.0, flags)
    if eps:
        print(f"\nWith-fractal episodes (N={len(eps)}):")
        print(f"{'Date':<12} {'dK':>6} {'wK':>6} {'+7d':>8} {'+30d':>8} "
              f"{'+60d':>8} {'+90d':>8} {'+365d':>8} {'MAE60':>8}")
        for e in eps:
            i = e["idx"]
            mae60 = fwd_drawdown(daily, i, 60)[1]
            print(f"{e['date']:<12} {e['dk']:>6.1f} {e['dwk']:>6.1f} "
                  f"{pct(forward_ret(daily, i, 7)):>8} "
                  f"{pct(forward_ret(daily, i, 30)):>8} "
                  f"{pct(forward_ret(daily, i, 60)):>8} "
                  f"{pct(forward_ret(daily, i, 90)):>8} "
                  f"{pct(forward_ret(daily, i, 365)):>8} "
                  f"{pct(mae60):>8}")


def main():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    store_d = KlineStore(DAILY_DB)
    daily = load(store_d, now_ms, DAY_MS)
    store_d.close()
    store_w = KlineStore(WEEKLY_DB)
    weekly = load(store_w, now_ms, WEEK_MS)
    store_w.close()

    print(f"BTCUSDT klines: daily={len(daily)}, weekly={len(weekly)}")

    dk = calc_stoch_rsi([c["close"] for c in daily])
    wk = calc_stoch_rsi([c["close"] for c in weekly])
    dwk = map_daily_to_weekly(daily, weekly, wk)
    frac = calc_top_fractals([c["high"] for c in weekly],
                             [c["low"] for c in weekly])
    flags = map_daily_fractal(daily, weekly, frac, FRACTAL_LOOKBACK)

    print_current(daily, dk, weekly, wk, frac, now_ms)

    eps = build_episodes(daily, dk, dwk, 10.0, 90.0)
    signal_idx = {e["idx"] for e in eps}
    baseline = compute_baseline(daily, dk, dwk, signal_idx, 10.0, 90.0)

    print(f"\nSignal days: {sum(e['end_idx'] - e['idx'] + 1 for e in eps)} "
          f"in {len(eps)} episodes")
    print_episodes(eps, daily)
    summarize("ALL EPISODES (weekly>90 & daily<10)", eps, daily, baseline)
    print("\nBaseline = all other days (indicators valid, condition false).")
    print("Excess = episode Avg - baseline Avg.")

    print_sensitivity(daily, dk, dwk)

    print_drawdown_dist(daily, dk, dwk)

    print_fractal_compare(daily, weekly, frac, dk, dwk, flags)

    print(f"\nNOTE: only {len(eps)} combo episodes in full history; interpret directionally.")
    print("Weekly bar used = last COMPLETED week (no lookahead). Entry at first-day close.")


if __name__ == "__main__":
    main()
