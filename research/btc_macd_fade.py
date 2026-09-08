"""BTC MACD fade research: after a golden cross, histogram shrinks N days in a row.

Usage:
    uv run python -m research.btc_macd_fade

Context (2026-08-31): BTC golden-crossed on 2026-08-18, histogram expanded for
6 days (peak 2026-08-24), then shrank 7 consecutive days. DIF is still above
DEA but a death cross is imminent; price sits below MA5 with a shallow
correction. This script measures historical analogs.

Questions:
    1. Given the histogram has shrunk K days in a row (still > 0), what is the
       probability of a death cross before the histogram grows again?
    2. After that death cross, how deep is the drawdown vs the rally's stage
       high (max high from golden cross to death cross)?

Definitions (daily klines, MACD 12/26/9, hist = DIF - DEA):
    - phase: golden cross day (hist flips > 0) to death cross day (hist <= 0)
    - signal day: first day per phase where the shrinking streak reaches K
    - outcome CROSS:  hist <= 0 before any hist[t] > hist[t-1]
    - outcome RECOVER: histogram grows again before crossing zero

Measured at the death cross day:
    - stage high = max high of the whole phase (golden cross -> death cross)
    - dd30  = min low of next 30 days vs stage high
    - dd_gc = min low until next golden cross (max 90 days) vs stage high
"""

from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, median

from core.data import KlineStore
from core.metrics import calc_sma

DB_PATH = ".data/crypto/binance-BTCUSDT-1d.db"
DAY_MS = 24 * 60 * 60 * 1000
WARMUP = 60
MAX_K = 12
FWD_DAYS = 30
GC_CAP = 90


def to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def ema(values: list[float], period: int) -> list[float]:
    out = [values[0]]
    k = 2 / (period + 1)
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def calc_macd(closes: list[float]) -> tuple[list[float], list[float], list[float]]:
    dif = [a - b for a, b in zip(ema(closes, 12), ema(closes, 26))]
    dea = ema(dif, 9)
    hist = [a - b for a, b in zip(dif, dea)]
    return dif, dea, hist


def find_phases(hist: list[float]) -> list[tuple[int, int]]:
    """Return [(golden_cross_i, death_cross_i), ...] with hist > 0 in between."""
    phases = []
    i = WARMUP
    n = len(hist)
    while i < n:
        if hist[i] > 0 and hist[i - 1] <= 0:
            j = i + 1
            while j < n and hist[j] > 0:
                j += 1
            if j < n:
                phases.append((i, j))
            i = j + 1
        else:
            i += 1
    return phases


def build_signals(candles: list[dict], hist: list[float],
                  phases: list[tuple[int, int]]) -> list[dict]:
    """One signal per phase per streak length K (first occurrence)."""
    closes = [c["close"] for c in candles]
    ma5 = calc_sma(closes, 5)
    signals = []
    for start, cross in phases:
        peak = max(range(start, cross), key=lambda i: hist[i])
        up_days = peak - start
        seen_ks: set[int] = set()
        streak = 0
        for i in range(peak + 1, cross):
            streak = streak + 1 if hist[i] < hist[i - 1] else 0
            k = streak
            if k == 0 or k > MAX_K or k in seen_ks:
                continue
            seen_ks.add(k)

            outcome = None
            cross_day = None
            for t in range(i + 1, cross + 1):
                if hist[t] <= 0:
                    outcome, cross_day = "CROSS", t
                    break
                if hist[t] > hist[t - 1]:
                    outcome = "RECOVER"
                    break

            stage_high = max(c["high"] for c in candles[start:cross + 1])
            entry = closes[i]
            window = candles[i + 1:i + 1 + FWD_DAYS]
            dd30 = (min(w["low"] for w in window) - stage_high) / stage_high * 100 \
                if len(window) == FWD_DAYS else None

            signals.append({
                "idx": i, "date": to_date(candles[i]["open_time"]),
                "k": k, "up_days": up_days, "gc_date": to_date(candles[start]["open_time"]),
                "outcome": outcome, "cross_day": cross_day, "stage_high": stage_high,
                "sig_dd": (entry - stage_high) / stage_high * 100,
                "below_ma5": ma5[i] is not None and entry < ma5[i], "dd30": dd30,
            })
    return signals


def pct(count: int, total: int) -> str:
    return f"{count / total * 100:.0f}%" if total else "N/A"


def print_prob_table(signals: list[dict]):
    print(f"\n{'=' * 78}")
    print("P(DEATH CROSS) - histogram still > 0, shrunk K days in a row")
    print("outcome CROSS = hist hits 0 before growing again; RECOVER = grows first")
    print(f"{'=' * 78}")
    print(f"{'K':>3} {'N':>4} {'Cross':>6} {'Recover':>8} {'P(cross)':>9} {'days_to_cross':>14}")
    print("-" * 78)
    for k in range(1, MAX_K + 1):
        group = [s for s in signals if s["k"] == k]
        if not group:
            continue
        crossed = [s for s in group if s["outcome"] == "CROSS"]
        days = [s["cross_day"] - s["idx"] for s in crossed]
        d_str = f"med {median(days):.0f}, max {max(days)}" if days else "-"
        print(f"{k:>3} {len(group):>4} {len(crossed):>6} {len(group) - len(crossed):>8} "
              f"{pct(len(crossed), len(group)):>9} {d_str:>14}")


def dd_stats(title: str, events: list[dict], key: str):
    vals = sorted(s[key] for s in events if s[key] is not None)
    print(f"\n{'=' * 78}")
    print(f"{title}  (N={len(vals)})")
    print(f"{'=' * 78}")
    if not vals:
        print("No completed events.")
        return
    p25, p75 = vals[int(len(vals) * 0.25)], vals[min(len(vals) - 1, int(len(vals) * 0.75))]
    print(f"Mean {mean(vals):+.1f}%  Median {median(vals):+.1f}%  "
          f"P25 {p25:+.1f}%  P75 {p75:+.1f}%  Range {vals[0]:+.1f}% ~ {vals[-1]:+.1f}%")
    buckets = defaultdict(int)
    for v in vals:
        b = ("< -25%" if v < -25 else "-25~-15%" if v < -15 else "-15~-10%" if v < -10
             else "-10~-5%" if v < -5 else "-5~0%" if v < 0 else ">= 0%")
        buckets[b] += 1
    for b in ["< -25%", "-25~-15%", "-15~-10%", "-10~-5%", "-5~0%", ">= 0%"]:
        print(f"  {b:<9} {buckets[b]:>3}  {'#' * buckets[b]}")


def terminal_streak(hist: list[float], start: int, cross: int) -> int:
    """Length of the shrinking run that leads directly into the death cross."""
    streak = 0
    for i in range(cross - 1, start, -1):
        if hist[i] < hist[i - 1]:
            streak += 1
        else:
            break
    return streak


def build_phase_drawdowns(candles: list[dict], hist: list[float],
                          phases: list[tuple[int, int]]) -> list[dict]:
    """One record per death-cross phase: post-cross drawdown vs stage high."""
    closes = [c["close"] for c in candles]
    ma200 = calc_sma(closes, 200)
    rows = []
    for start, cross in phases:
        peak = max(range(start, cross), key=lambda i: hist[i])
        stage_high = max(c["high"] for c in candles[start:cross + 1])
        t_streak = terminal_streak(hist, start, cross)

        rec: dict = {"gc_date": to_date(candles[start]["open_time"]),
                     "cross_date": to_date(candles[cross]["open_time"]),
                     "up_days": peak - start, "term_streak": t_streak,
                     "stage_high": stage_high,
                     "bull": ma200[cross] is not None and closes[cross] > ma200[cross],
                     "cross_dd": (closes[cross] - stage_high) / stage_high * 100,
                     "dd30": None, "dd_gc": None, "dd_days": None, "recovered": None}

        window = candles[cross + 1:cross + 1 + FWD_DAYS]
        if len(window) == FWD_DAYS:
            lo = min(window, key=lambda w: w["low"])
            rec["dd30"] = (lo["low"] - stage_high) / stage_high * 100
            rec["dd_days"] = (lo["open_time"] - candles[cross]["open_time"]) // DAY_MS
            if cross + 61 <= len(candles):
                rec["recovered"] = max(
                    w["high"] for w in candles[cross + 1:cross + 61]) > stage_high

        end = min(cross + GC_CAP, len(candles) - 1)
        t = cross + 1
        while t <= end:
            if hist[t] > 0 and hist[t - 1] <= 0:
                break
            t += 1
        seg = candles[cross + 1:t + 1]
        if seg:
            rec["dd_gc"] = (min(w["low"] for w in seg) - stage_high) / stage_high * 100
        rows.append(rec)
    return rows


def build_recover_breakouts(candles: list[dict], hist: list[float],
                           phases: list[tuple[int, int]], min_k: int = 7,
                           horizon: int = 180) -> list[dict]:
    """Phases where a >=min_k shrinking run did NOT dead cross, then track price."""
    closes = [c["close"] for c in candles]
    ma200 = calc_sma(closes, 200)
    rows = []
    for start, cross in phases:
        peak = max(range(start, cross), key=lambda i: hist[i])
        streak = 0
        sig = None
        for i in range(peak + 1, cross):
            streak = streak + 1 if hist[i] < hist[i - 1] else 0
            if streak < min_k:
                continue
            for t in range(i + 1, cross + 1):
                if hist[t] <= 0:
                    break
                if hist[t] > hist[t - 1]:
                    sig = i
                    break
            if sig is not None:
                break
        if sig is None:
            continue

        local_high = max(c["high"] for c in candles[start:sig + 1])
        brk = None
        for t in range(sig + 1, min(sig + 1 + horizon, len(candles))):
            if closes[t] > local_high:
                brk = t
                break

        rec = {"gc_date": to_date(candles[start]["open_time"]),
               "sig_date": to_date(candles[sig]["open_time"]),
               "streak": min_k, "local_high": local_high, "break_idx": brk,
               "break_date": to_date(candles[brk]["open_time"]) if brk else "-",
               "days_to_break": brk - sig if brk else None,
               "bull": ma200[sig] is not None and closes[sig] > ma200[sig],
               "run_after": None, "held_60d": None}
        if brk and brk + 61 <= len(candles):
            rec["run_after"] = (max(c["high"] for c in candles[brk:brk + 61])
                                / local_high - 1) * 100
            rec["held_60d"] = (closes[brk + 60] - local_high) / local_high * 100
        rows.append(rec)
    return rows


def print_recover_breakouts(rows: list[dict]):
    print(f"\n{'=' * 110}")
    print("RECOVERED (no death cross) after >= 7-day shrink, then new high vs local top?")
    print(f"{'=' * 110}")
    print(f"{'GC':<12} {'Signal':<12} {'Regime':<5} {'LocalHigh':>10} {'Breakout':<12} "
          f"{'Days':>5} {'Run60d':>8} {'Close+60d':>9}")
    print("-" * 110)
    for r in sorted(rows, key=lambda x: x["sig_date"]):
        regime = "bull" if r["bull"] else "bear"
        days = f"{r['days_to_break']}" if r["days_to_break"] is not None else "-"
        run = f"{r['run_after']:+.1f}%" if r["run_after"] is not None else "?"
        held = f"{r['held_60d']:+.1f}%" if r["held_60d"] is not None else "?"
        print(f"{r['gc_date']:<12} {r['sig_date']:<12} {regime:<5} {r['local_high']:>10,.0f} "
              f"{r['break_date']:<12} {days:>5} {run:>8} {held:>9}")


def print_phase_details(rows: list[dict], min_streak: int):
    group = [r for r in rows if r["term_streak"] >= min_streak]
    print(f"\n{'=' * 108}")
    print(f"PHASES whose final shrinking streak into the death cross >= {min_streak} days "
          f"(one row per phase)")
    print(f"{'=' * 108}")
    print(f"{'GC':<12} {'Cross':<12} {'Up':>3} {'Term':>4} {'Regime':<5} {'StgHigh':>9} "
          f"{'DD@cross':>8} {'DD30':>7} {'DD->GC':>7} {'DDdays':>6} {'Recov60'}")
    print("-" * 108)
    for r in sorted(group, key=lambda x: x["gc_date"]):
        dd30 = f"{r['dd30']:+.1f}%" if r["dd30"] is not None else "N/A"
        ddgc = f"{r['dd_gc']:+.1f}%" if r["dd_gc"] is not None else "N/A"
        dd = f"{r['dd_days']}" if r["dd_days"] is not None else "-"
        rec = {True: "Y", False: "N", None: "?"}[r["recovered"]]
        regime = "bull" if r["bull"] else "bear"
        print(f"{r['gc_date']:<12} {r['cross_date']:<12} {r['up_days']:>3} {r['term_streak']:>4} "
              f"{regime:<5} {r['stage_high']:>9,.0f} {r['cross_dd']:>+7.1f}% {dd30:>7} {ddgc:>7} {dd:>6} {rec:>7}")


def main():
    store = KlineStore(DB_PATH)
    raw = store.query("BTCUSDT")
    store.close()

    global candles
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    forming = raw[-1] if raw and raw[-1]["open_time"] + DAY_MS > now_ms else None
    candles = raw[:-1] if forming else raw

    closes = [c["close"] for c in candles]
    dif, dea, hist = calc_macd(closes)
    phases = find_phases(hist)

    print(f"BTCUSDT daily: {len(candles)} bars "
          f"({to_date(candles[0]['open_time'])} ~ {to_date(candles[-1]['open_time'])}), "
          f"golden-cross phases: {len(phases)}")

    signals = build_signals(candles, hist, phases)
    print_prob_table(signals)

    phase_rows = build_phase_drawdowns(candles, hist, phases)
    p5 = [r for r in phase_rows if r["term_streak"] >= 5]
    p7 = [r for r in phase_rows if r["term_streak"] >= 7]
    dd_stats("DRAWDOWN vs STAGE HIGH after death cross, terminal streak >= 5, next 30 days", p5, "dd30")
    dd_stats("DRAWDOWN vs STAGE HIGH after death cross, terminal streak >= 7, next 30 days", p7, "dd30")
    dd_stats("DRAWDOWN vs STAGE HIGH after death cross, terminal streak >= 7, until next golden cross", p7, "dd_gc")

    rec = [r for r in p5 if r["recovered"] is not None]
    if rec:
        n_rec = sum(1 for r in rec if r["recovered"])
        print(f"\nReclaim stage high within 60 days after death cross: "
              f"{n_rec}/{len(rec)} ({pct(n_rec, len(rec))}), terminal streak >= 5")

    print(f"\n{'=' * 78}")
    print("REGIME SPLIT at death cross (close vs 200D MA), terminal streak >= 5")
    print(f"{'=' * 78}")
    for label, want in (("BULL (above 200D MA)", True), ("BEAR (below 200D MA)", False)):
        vals = sorted(r["dd30"] for r in p5 if r["bull"] == want and r["dd30"] is not None)
        if vals:
            print(f"  {label:<22} N={len(vals):>2}  mean {mean(vals):+.1f}%  "
                  f"median {median(vals):+.1f}%  range {vals[0]:+.1f}% ~ {vals[-1]:+.1f}%")

    print_phase_details(phase_rows, 7)

    recover_rows = build_recover_breakouts(candles, hist, phases)
    print_recover_breakouts(recover_rows)

    like_now = [s for s in signals if s["k"] >= 6 and s["up_days"] >= 4
                and s["below_ma5"] and s["sig_dd"] > -8]
    print(f"\n{'=' * 78}")
    print(f"CLOSEST ANALOGS to now (K>=6, up_days>=4, close<MA5, corr < 8%): N={len(like_now)}")
    print(f"{'=' * 78}")
    for s in like_now:
        dd30 = f"{s['dd30']:+.1f}%" if s["dd30"] is not None else "N/A"
        cross = to_date(candles[s["cross_day"]]["open_time"]) if s["cross_day"] else "-"
        print(f"  {s['date']}  K={s['k']} up={s['up_days']}  {s['outcome']:<7} cross {cross}  "
              f"high {s['stage_high']:,.0f}  dd@sig {s['sig_dd']:+.1f}%  dd30 {dd30}")

    print(f"\n{'=' * 78}")
    print("CURRENT EPISODE (includes today's forming bar)")
    print(f"{'=' * 78}")
    all_dif, all_dea, all_hist = calc_macd([c["close"] for c in raw])
    for i in range(len(raw) - 9, len(raw)):
        tag = " (forming)" if raw[i] is forming else ""
        print(f"{to_date(raw[i]['open_time'])}{tag:<11} close={raw[i]['close']:>9,.0f} "
              f"DIF={all_dif[i]:>+8.1f} DEA={all_dea[i]:>+8.1f} hist={all_hist[i]:>+8.1f}")
    gap = all_dif[-1] - all_dea[-1]
    shrink = all_hist[-2] - all_hist[-1]
    print(f"\nDIF-DEA gap = {gap:.0f}; yesterday's shrink = {shrink:.0f}; "
          f"at that pace cross in ~{gap / shrink:.1f} day(s)")

    gc = next(i for i in range(len(raw) - 1, WARMUP, -1)
              if all_hist[i] > 0 and all_hist[i - 1] <= 0)
    stage_high = max(c["high"] for c in raw[gc:])
    print(f"Phase golden cross: {to_date(raw[gc]['open_time'])}; "
          f"stage high {stage_high:,.0f}; "
          f"current close vs stage high {(raw[-1]['close'] - stage_high) / stage_high * 100:+.1f}%")


if __name__ == "__main__":
    main()
