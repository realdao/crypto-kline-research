"""BTC bollinger upper-band expansion-reversal research.

Usage:
    uv run python -m research.btc_boll_expansion

Context: 2026-08-19 BTC broke out on heavy volume; the daily Bollinger(20,2)
upper band then rose rapidly, peaked on 2026-09-03, and has started to fall
while the band gap (upper - lower) remains very wide. This script finds
historical analogs and measures what happened next — both price and the
evolution of the band width.

Event definition (daily klines, Bollinger(20,2), population std):
    Day t is an event when ALL hold:
    - upper[t] < upper[t-1]                              upper band starts falling
    - upper[t-1] == max(upper[t-10 .. t-1])              yesterday was a 10d peak
    - upper[t-1] >= 1.10 * min(upper[t-12 .. t-1])       rapid rise into the peak
    - %B[t-1] >= 50                                      upside-driven expansion
    - a "walk" day within the past 20 days: %B > 100 and
      volume percentile (730d) >= 75                     breakout with volume
    - at least 30 days since the previous event           de-cluster

Measured from the event day's close:
    - forward return / MFE / MAE at +5/+10/+20/+60/+90 days
    - band width (upper-lower)/mid evolution: ratio to event-day width
    - days until width halves; whether the upper band renews its high
    - %B path: does price fall back to the mid band?

Outputs:
    1. Current 2026-09 episode recap
    2. Event detail table
    3. Price summary vs baseline (all non-event days)
    4. Band-width evolution summary
    5. Pattern groups: trend / width percentile / %B / upper-band renewal
"""

from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, median

from core.data import KlineStore
from core.metrics import calc_rsi, calc_sma, calc_volume_percentiles

DB_PATH = ".data/crypto/binance-BTCUSDT-1d.db"
DAY_MS = 24 * 60 * 60 * 1000

BB_PERIOD = 20
BB_MULT = 2.0

WALK_PB = 100.0        # %B above this = "walking the band"
WALK_VOL = 75.0        # volume percentile (730d) required on a walk day
WALK_WINDOW = 20       # walk day must be within this many days before event
PEAK_LOOKBACK = 10     # yesterday must be the N-day max of the upper band
RISE_PCT = 10.0        # upper-band rise required over RISE_DAYS into the peak
RISE_DAYS = 12
COOLDOWN = 30

RET_HORIZONS = [5, 10, 20, 60, 90]
BW_HORIZONS = [5, 10, 20, 30, 60, 90]
GROUP_HORIZONS = [10, 20, 60]
SMALL_SAMPLE = 8


def to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def pct(v: float | None, signed: bool = True) -> str:
    if v is None:
        return "N/A"
    return f"{v:+.1f}%" if signed else f"{v:.1f}%"


def compute_bands(closes: list[float]) -> tuple:
    """Bollinger(20,2) mid/upper/lower, width%, %B, width percentile (730d)."""
    n = len(closes)
    mid = [None] * n
    upper = [None] * n
    lower = [None] * n
    bw = [None] * n
    pb = [None] * n
    for i in range(BB_PERIOD - 1, n):
        w = closes[i - BB_PERIOD + 1:i + 1]
        m = sum(w) / BB_PERIOD
        var = sum((x - m) ** 2 for x in w) / BB_PERIOD
        sd = var ** 0.5
        u = m + BB_MULT * sd
        l = m - BB_MULT * sd
        mid[i] = m
        upper[i] = u
        lower[i] = l
        bw[i] = (u - l) / m * 100
        pb[i] = (closes[i] - l) / (u - l) * 100

    bw_pct = [None] * n
    for i in range(n):
        if bw[i] is None:
            continue
        w = [b for b in bw[max(0, i - 729):i + 1] if b is not None]
        bw_pct[i] = sum(1 for b in w if b < bw[i]) / len(w) * 100
    return mid, upper, lower, bw, pb, bw_pct


def build_events(candles: list[dict], series: dict) -> list[dict]:
    upper = series["upper"]
    bw = series["bw"]
    pb = series["pb"]
    bw_pct = series["bw_pct"]
    vol_pct = series["vol_pct"]
    closes = series["closes"]
    ma140 = series["ma140"]
    rsi14 = series["rsi14"]

    events = []
    last_ev = -10**9
    for t in range(30, len(candles)):
        if t - last_ev <= COOLDOWN:
            continue
        if upper[t] is None or upper[t] >= upper[t - 1]:
            continue
        if upper[t - 1] < max(upper[t - PEAK_LOOKBACK:t]):
            continue
        lo_u = min(upper[t - RISE_DAYS:t])
        if upper[t - 1] < lo_u * (1 + RISE_PCT / 100):
            continue
        if pb[t - 1] < 50:
            continue
        walk_days = [j for j in range(t - WALK_WINDOW, t)
                     if pb[j] is not None and pb[j] > WALK_PB
                     and vol_pct[j] is not None and vol_pct[j] >= WALK_VOL]
        if not walk_days:
            continue

        last_ev = t
        entry = closes[t]
        peak_u = upper[t - 1]

        # forward price stats
        fwd = {}
        for h in RET_HORIZONS:
            window = candles[t + 1:t + 1 + h]
            if len(window) < h:
                fwd[h] = (None, None, None)
                continue
            hi = max(w["high"] for w in window)
            lo = min(w["low"] for w in window)
            fwd[h] = ((candles[t + h]["close"] - entry) / entry * 100,
                      (hi - entry) / entry * 100,
                      (lo - entry) / entry * 100)

        # band-width evolution: ratio to event-day width
        bw_fwd = {}
        for h in BW_HORIZONS:
            if t + h < len(candles) and bw[t + h] is not None:
                bw_fwd[h] = bw[t + h] / bw[t] if bw[t] else None
            else:
                bw_fwd[h] = None

        # days until width halves (search up to 90d)
        half_day = None
        for k in range(1, min(91, len(candles) - t)):
            if bw[t + k] is not None and bw[t + k] < bw[t] * 0.5:
                half_day = k
                break

        # does the upper band renew its high within 30d? (signal was premature)
        renewed_30 = any(upper[k] is not None and upper[k] > peak_u
                         for k in range(t, min(t + 31, len(candles))))

        # %B path
        pb_path = {}
        for h in (10, 20, 30):
            pb_path[h] = pb[t + h] if t + h < len(candles) else None

        # price new 60d-high within 30d
        hi60_pre = max(closes[max(0, t - 60):t + 1])
        new_high_30 = any(closes[k] > hi60_pre
                          for k in range(t, min(t + 31, len(candles))))

        ma = ma140[t]
        trend = "N/A" if ma is None else ("ABOVE_MA" if entry >= ma else "BELOW_MA")
        r = rsi14[t]
        rsi_tag = "N/A" if r is None else ("RSI>70" if r > 70 else "RSI50-70" if r >= 50 else "RSI<50")

        events.append({
            "idx": t,
            "date": to_date(candles[t]["open_time"]),
            "close": entry,
            "peak_u": peak_u,
            "rise": (peak_u / lo_u - 1) * 100,
            "walk_date": to_date(candles[walk_days[0]]["open_time"]),
            "walk_gap": t - walk_days[0],
            "bw": bw[t],
            "bw_pct": bw_pct[t],
            "pb": pb[t],
            "fwd": fwd,
            "bw_fwd": bw_fwd,
            "half_day": half_day,
            "renewed_30": renewed_30,
            "pb_path": pb_path,
            "new_high_30": new_high_30,
            "tags": {"trend": trend, "rsi": rsi_tag},
        })
    return events


def compute_baseline(candles: list[dict], event_idx: set) -> dict:
    closes = [c["close"] for c in candles]
    baseline = {}
    for h in RET_HORIZONS:
        rets = []
        for i in range(1, len(candles) - h):
            if i in event_idx:
                continue
            rets.append((closes[i + h] - closes[i]) / closes[i] * 100)
        baseline[h] = {"n": len(rets), "avg": mean(rets)}
    return baseline


def print_current(candles: list[dict], series: dict, events: list[dict]):
    cur = [e for e in events if e["idx"] >= len(candles) - 5]
    print(f"\n{'=' * 110}")
    print("CURRENT EPISODE CHECK (last 5 days)")
    print(f"{'=' * 110}")
    if not cur:
        print("No event at the tail of the data — upper band has not rolled over yet.")
        return
    e = cur[-1]
    print(f"Latest event fires on {e['date']} (close {e['close']:,.0f}):")
    print(f"  breakout/walk day: {e['walk_date']} ({e['walk_gap']}d before peak), "
          f"upper-band rise into peak: +{e['rise']:.1f}% in {RISE_DAYS}d")
    print(f"  width {e['bw']:.1f}% of mid (P{e['bw_pct']:.0f} of trailing 730d), "
          f"%B {e['pb']:.0f}, trend {e['tags']['trend']}, {e['tags']['rsi']}")
    print(f"  upper-band peak: {e['peak_u']:,.0f} on {to_date(candles[e['idx'] - 1]['open_time'])}")
    avail = [h for h in BW_HORIZONS if e["bw_fwd"][h] is not None]
    if avail:
        print("  width ratio so far: " + ", ".join(
            f"+{h}d {e['bw_fwd'][h]:.2f}x" for h in avail))
    print("  (forward windows incomplete — excluded from summary stats)")


def print_details(events: list[dict]):
    print(f"\n{'=' * 175}")
    print("EVENT DETAIL - forward returns measured from event-day close "
          "(day the upper band first turns down after a volume-backed expansion)")
    print(f"{'=' * 175}")
    print(f"{'Date':<12} {'Walk':<12} {'Rise%':>6} {'BW%':>5} {'BWpct':>5} {'%B':>5} "
          f"{'+5d':>7} {'+10d':>7} {'+20d':>7} {'+60d':>7} {'+90d':>7} "
          f"{'MFE90':>7} {'MAE90':>7} {'HalfW':>6} {'Renew':>5} {'NewHi':>5}  Tags")
    print("-" * 175)
    for e in events:
        t = e["tags"]
        f = e["fwd"]
        print(f"{e['date']:<12} {e['walk_date']:<12} {e['rise']:>5.0f}% {e['bw']:>4.1f}% "
              f"{e['bw_pct']:>4.0f}% {e['pb']:>4.0f}% "
              f"{pct(f[5][0]):>7} {pct(f[10][0]):>7} {pct(f[20][0]):>7} "
              f"{pct(f[60][0]):>7} {pct(f[90][0]):>7} "
              f"{pct(f[90][1]):>7} {pct(f[90][2]):>7} "
              f"{str(e['half_day']) if e['half_day'] else '>90d':>6} "
              f"{'Y' if e['renewed_30'] else '-':>5} {'Y' if e['new_high_30'] else '-':>5}  "
              f"{t['trend']}|{t['rsi']}")
    print("HalfW  = days until band width falls below 50% of event-day width")
    print("Renew  = upper band exceeded its peak again within 30d (rollover was premature)")
    print("NewHi  = close made a new 60d high within 30d")


def print_price_summary(events: list[dict], baseline: dict):
    print(f"\n{'=' * 105}")
    print(f"PRICE SUMMARY - ALL EVENTS (N={len(events)})")
    print(f"{'=' * 105}")
    print(f"{'Hor':>5} {'N':>4} {'Avg':>8} {'Med':>8} {'Win%':>6} {'MFE':>8} {'MAE':>8} "
          f"| {'BaseN':>6} {'BaseAvg':>8} {'Excess':>8}")
    print("-" * 105)
    for h in RET_HORIZONS:
        rows = [e["fwd"][h] for e in events if e["fwd"][h][0] is not None]
        if not rows:
            print(f"+{h:<4}d    0     N/A")
            continue
        rets = [r[0] for r in rows]
        b = baseline[h]
        print(f"+{h:<4}d {len(rets):>4} {mean(rets):>+8.1f} {median(rets):>+8.1f} "
              f"{sum(1 for v in rets if v > 0) / len(rets) * 100:>5.0f}% "
              f"{mean(r[1] for r in rows):>+8.1f} {mean(r[2] for r in rows):>+8.1f} "
              f"| {b['n']:>6} {b['avg']:>+8.1f} {mean(rets) - b['avg']:>+8.1f}")


def print_bw_summary(events: list[dict]):
    print(f"\n{'=' * 100}")
    print("BAND-WIDTH EVOLUTION - width(t+h) / width(event day)")
    print(f"{'=' * 100}")
    header = f"{'Hor':>5} {'N':>4} {'AvgRatio':>9} {'MedRatio':>9} {'%Contract':>10}"
    print(header)
    print("-" * 100)
    for h in BW_HORIZONS:
        vals = [e["bw_fwd"][h] for e in events if e["bw_fwd"][h] is not None]
        if not vals:
            print(f"+{h:<4}d    0     N/A")
            continue
        print(f"+{h:<4}d {len(vals):>4} {mean(vals):>9.2f}x {median(vals):>9.2f}x "
              f"{sum(1 for v in vals if v < 1.0) / len(vals) * 100:>9.0f}%")

    halves = [e["half_day"] for e in events if e["half_day"] is not None]
    never = sum(1 for e in events if e["half_day"] is None)
    print(f"\nDays until width halves: avg {mean(halves):.0f}, median {median(halves):.0f} "
          f"({len(halves)} of {len(events)} events; {never} never halved within 90d)")

    renew = sum(1 for e in events if e["renewed_30"])
    print(f"Upper band renewed its high within 30d: {renew} of {len(events)} events "
          f"({renew / len(events) * 100:.0f}%)")

    print("\n%B path after event (50 = mid band):")
    for h in (10, 20, 30):
        vals = [e["pb_path"][h] for e in events if e["pb_path"][h] is not None]
        if vals:
            print(f"  +{h:>2}d  avg {mean(vals):>5.0f}  median {median(vals):>5.0f}  "
                  f"(share below 50: {sum(1 for v in vals if v < 50) / len(vals) * 100:.0f}%)")


def print_groups(title: str, events: list[dict], dim: str, key=None):
    groups = defaultdict(list)
    for e in events:
        groups[key(e) if key else e["tags"][dim]].append(e)

    print(f"\n{'=' * 100}")
    print(f"PATTERN - {title}")
    print(f"{'=' * 100}")
    header = f"{'Group':<12} {'N':>3} "
    for h in GROUP_HORIZONS:
        header += f"{'+%dd_avg' % h:>8} {'hit':>5} "
    header += f"{'BW30x':>7} {'Half':>5}"
    print(header + "  (* = small sample)")
    print("-" * 100)

    for label in sorted(groups, key=lambda k: -len(groups[k])):
        evs = groups[label]
        row = f"{label:<12} {len(evs):>3} "
        for h in GROUP_HORIZONS:
            rets = [e["fwd"][h][0] for e in evs if e["fwd"][h][0] is not None]
            if rets:
                hit = sum(1 for v in rets if v > 0) / len(rets) * 100
                row += f"{mean(rets):>+8.1f} {hit:>4.0f}% "
            else:
                row += f"{'N/A':>8} {'N/A':>5} "
        bw30 = [e["bw_fwd"][30] for e in evs if e["bw_fwd"][30] is not None]
        row += f"{mean(bw30):>6.2f}x " if bw30 else f"{'N/A':>7}"
        halves = [e["half_day"] for e in evs if e["half_day"] is not None]
        row += f"{median(halves):>4.0f}d" if halves else f"{'N/A':>5}"
        if len(evs) < SMALL_SAMPLE:
            row += " *"
        print(row)


def main():
    store = KlineStore(DB_PATH)
    candles = store.query("BTCUSDT")
    store.close()

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if candles and candles[-1]["open_time"] + DAY_MS > now_ms:
        candles = candles[:-1]

    first = to_date(candles[0]["open_time"])
    last = to_date(candles[-1]["open_time"])
    print(f"BTCUSDT daily klines: {len(candles)} ({first} ~ {last})")

    closes = [c["close"] for c in candles]
    series = {
        "closes": closes,
        "vol_pct": calc_volume_percentiles([c["volume"] for c in candles], 730),
        "ma140": calc_sma(closes, 140),
        "rsi14": calc_rsi(closes, 14),
    }
    mid, upper, lower, bw, pb, bw_pct = compute_bands(closes)
    series.update({"mid": mid, "upper": upper, "lower": lower,
                   "bw": bw, "pb": pb, "bw_pct": bw_pct})

    events = build_events(candles, series)
    print(f"Events (upper-band rollover after volume-backed expansion): {len(events)}")

    print_current(candles, series, events)
    print_details(events)

    complete = [e for e in events if e["fwd"][5][0] is not None]
    baseline = compute_baseline(candles, {e["idx"] for e in events})
    print_price_summary(complete, baseline)

    print_bw_summary(complete)

    print_groups("TREND (close vs 140d MA / 20W)", complete, "trend")
    print_groups("BAND-WIDTH PERCENTILE at event (730d)", complete, None,
                 key=lambda e: "WIDE P>80" if e["bw_pct"] > 80 else "MID P50-80" if e["bw_pct"] > 50 else "NARROW P<50")
    print_groups("UPPER-BAND RENEWAL within 30d", complete, None,
                 key=lambda e: "RENEWED" if e["renewed_30"] else "NO-RENEW")
    print_groups("%B AT EVENT", complete, None,
                 key=lambda e: "%B>60" if e["pb"] > 60 else "%B40-60" if e["pb"] > 40 else "%B<40")

    print(f"\nNOTE: N={len(complete)} complete events; groups are small (marked *).")
    print("Interpret directionally. Events span 2017-2026 across full bull/bear cycles.")


if __name__ == "__main__":
    main()
