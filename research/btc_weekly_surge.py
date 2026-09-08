"""BTC weekly-surge research: performance after a >=20% single-week gain.

Usage:
    uv run python -m research.btc_weekly_surge [threshold]

Sections:
    1. Event detail table (date, gain, forward returns, pattern tags)
    2. Summary vs baseline: all events / de-clustered / non-event weeks
    3. Pattern-conditioned groups: trend / position / momentum / breakout /
       volume / magnitude / weekly RSI
    4. Threshold sensitivity (15% / 20% / 25%)

Event definition: weekly close-to-close gain >= threshold.
"""

import sys
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, median

from core.data import KlineStore
from core.metrics import calc_rsi, calc_sma, calc_volume_percentiles

DB_PATH = ".data/crypto/binance-BTCUSDT-1w.db"
HORIZONS = [1, 2, 4, 8, 13, 26, 52]
GROUP_HORIZONS = [4, 13, 26]
CLUSTER_GAP = 4
SMALL_SAMPLE = 10
WEEK_MS = 7 * 24 * 60 * 60 * 1000


def forward_stats(candles: list[dict], i: int, h: int) -> tuple:
    """(ret%, mfe%, mae%) over h weeks after week i close; None if out of range."""
    if i + h >= len(candles):
        return None, None, None
    entry = candles[i]["close"]
    window = candles[i + 1:i + h + 1]
    ret = (candles[i + h]["close"] - entry) / entry * 100
    mfe = (max(w["high"] for w in window) - entry) / entry * 100
    mae = (min(w["low"] for w in window) - entry) / entry * 100
    return ret, mfe, mae


def pct(v: float | None, signed: bool = True) -> str:
    if v is None:
        return "N/A"
    return f"{v:+.1f}%" if signed else f"{v:.1f}%"


def build_events(candles: list[dict], threshold: float) -> list[dict]:
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    ma20w = calc_sma(closes, 20)
    rsi14 = calc_rsi(closes, 14)
    vol_pct = calc_volume_percentiles([c["volume"] for c in candles], 104)

    m1, m2 = threshold + 5, threshold + 15
    mag_lo = f"{threshold:.0f}-{m1:.0f}"
    mag_mid = f"{m1:.0f}-{m2:.0f}"
    mag_hi = f"{m2:.0f}+"

    events = []
    for i in range(1, len(candles)):
        gain = (closes[i] - closes[i - 1]) / closes[i - 1] * 100
        if gain < threshold:
            continue

        if ma20w[i] is None or ma20w[i - 1] is None:
            trend = "N/A"
        elif closes[i] < ma20w[i]:
            trend = "BEAR"
        elif ma20w[i] >= ma20w[i - 1]:
            trend = "BULL"
        else:
            trend = "TRANS"

        hi52 = max(highs[max(0, i - 51):i + 1])
        dd = (closes[i] - hi52) / hi52 * 100
        pos = "NEAR" if dd > -10 else ("MID" if dd > -30 else "DEEP")

        prior4 = (closes[i - 1] - closes[i - 5]) / closes[i - 5] * 100 if i >= 5 else None
        mom4 = "N/A" if prior4 is None else ("BURST" if prior4 < 5 else "ACCEL")

        prior13 = (closes[i - 1] - closes[i - 14]) / closes[i - 14] * 100 if i >= 14 else None
        mom13 = "N/A" if prior13 is None else ("UP" if prior13 >= 0 else "DOWN")

        breakout = "N/A" if i < 26 else (
            "BRK26" if closes[i] > max(closes[i - 26:i]) else "RANGE")

        vp = vol_pct[i]
        vol = "N/A" if (i < 52 or vp is None) else ("HVOL" if vp >= 80 else "NVOL")

        mag = mag_lo if gain < m1 else (mag_mid if gain < m2 else mag_hi)

        r = rsi14[i]
        rsi_tag = "N/A" if r is None else ("RSI>70" if r > 70 else ("RSI60-70" if r >= 60 else "RSI<60"))

        events.append({
            "idx": i,
            "date": datetime.fromtimestamp(
                candles[i]["open_time"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
            "gain": gain,
            "prior4": prior4,
            "dd": dd,
            "tags": {"trend": trend, "pos": pos, "mom4": mom4, "mom13": mom13,
                     "breakout": breakout, "vol": vol, "mag": mag, "rsi": rsi_tag},
            "fwd": {h: forward_stats(candles, i, h) for h in HORIZONS},
        })
    return events


def dedupe(events: list[dict], gap: int = CLUSTER_GAP) -> list[dict]:
    """Keep the first event of each cluster (gap > N weeks from last kept)."""
    kept = []
    last = None
    for e in events:
        if last is None or e["idx"] - last > gap:
            kept.append(e)
            last = e["idx"]
    return kept


def compute_baseline(candles: list[dict], event_idx: set) -> dict:
    baseline = {}
    for h in HORIZONS:
        rets = []
        for i in range(1, len(candles)):
            if i in event_idx:
                continue
            ret, _, _ = forward_stats(candles, i, h)
            if ret is not None:
                rets.append(ret)
        baseline[h] = {
            "n": len(rets),
            "avg": mean(rets),
            "win": sum(1 for v in rets if v > 0) / len(rets) * 100,
        }
    return baseline


def print_details(events: list[dict]):
    print(f"\n{'=' * 160}")
    print("EVENT DETAIL - forward returns measured from event-week close")
    print(f"{'=' * 160}")
    print(f"{'Date':<12} {'Gain':>7} {'Pri4W':>7} {'DD52W':>6} "
          f"{'+1W':>7} {'+2W':>7} {'+4W':>7} {'+8W':>7} {'+13W':>7} {'+26W':>7} {'+52W':>7} "
          f"{'MFE13':>7} {'MAE13':>7}  Tags")
    print("-" * 160)
    for e in events:
        t = e["tags"]
        tag_s = (f"{t['trend']}|{t['pos']}|{t['mom4']}|{t['breakout']}"
                 f"|{t['vol']}|{t['rsi']}")
        f = e["fwd"]
        print(f"{e['date']:<12} {e['gain']:>+6.1f}% {pct(e['prior4']):>7} {e['dd']:>5.0f}% "
              f"{pct(f[1][0]):>7} {pct(f[2][0]):>7} {pct(f[4][0]):>7} {pct(f[8][0]):>7} "
              f"{pct(f[13][0]):>7} {pct(f[26][0]):>7} {pct(f[52][0]):>7} "
              f"{pct(f[13][1]):>7} {pct(f[13][2]):>7}  {tag_s}")
    print("Tags: trend|pos|prior4W-momentum|breakout|volume|RSI  "
          "(BULL/BEAR/TRANS, NEAR/MID/DEEP, BURST/ACCEL, BRK26/RANGE, HVOL/NVOL)")


def print_summary(title: str, events: list[dict], baseline: dict):
    print(f"\n{'=' * 105}")
    print(f"{title}  (N={len(events)})")
    print(f"{'=' * 105}")
    print(f"{'Hor':>4} {'N':>4} {'Avg':>8} {'Med':>8} {'Win%':>6} {'MFE':>8} {'MAE':>8} "
          f"| {'Base_N':>6} {'BaseAvg':>8} {'Excess':>8}")
    print("-" * 105)
    for h in HORIZONS:
        rows = [e["fwd"][h] for e in events if e["fwd"][h][0] is not None]
        if not rows:
            print(f"+{h:<3}W    0     N/A")
            continue
        rets = [r[0] for r in rows]
        mfes = [r[1] for r in rows]
        maes = [r[2] for r in rows]
        b = baseline[h]
        excess = mean(rets) - b["avg"]
        print(f"+{h:<3}W {len(rets):>4} {mean(rets):>+8.1f} {median(rets):>+8.1f} "
              f"{sum(1 for v in rets if v > 0) / len(rets) * 100:>5.0f}% "
              f"{mean(mfes):>+8.1f} {mean(maes):>+8.1f} "
              f"| {b['n']:>6} {b['avg']:>+8.1f} {excess:>+8.1f}")


def print_groups(title: str, events: list[dict], dim: str):
    groups = defaultdict(list)
    for e in events:
        groups[e["tags"][dim]].append(e)

    print(f"\n{'=' * 100}")
    print(f"PATTERN - {title}")
    print(f"{'=' * 100}")
    header = f"{'Group':<10} {'N':>3} "
    for h in GROUP_HORIZONS:
        header += f"{'+%dW_avg' % h:>8} {'hit':>5} "
    print(header + "  (* = small sample)")
    print("-" * 100)

    for label in sorted(groups, key=lambda k: -len(groups[k])):
        evs = groups[label]
        row = f"{label:<10} {len(evs):>3} "
        for h in GROUP_HORIZONS:
            rets = [e["fwd"][h][0] for e in evs if e["fwd"][h][0] is not None]
            if rets:
                hit = sum(1 for v in rets if v > 0) / len(rets) * 100
                row += f"{mean(rets):>+8.1f} {hit:>4.0f}% "
            else:
                row += f"{'N/A':>8} {'N/A':>5} "
        if len(evs) < SMALL_SAMPLE:
            row += " *"
        print(row)


def print_sensitivity(candles: list[dict], thresholds=(15, 20, 25)):
    print(f"\n{'=' * 100}")
    print("THRESHOLD SENSITIVITY - all events at each threshold (no dedup)")
    print(f"{'=' * 100}")
    header = f"{'Thr':>4} {'N':>4} "
    for h in GROUP_HORIZONS:
        header += f"{'+%dW_avg' % h:>8} {'hit':>5} "
    print(header)
    print("-" * 100)

    for t in thresholds:
        evs = build_events(candles, t)
        row = f"{t:>3.0f}% {len(evs):>4} "
        for h in GROUP_HORIZONS:
            rets = [e["fwd"][h][0] for e in evs if e["fwd"][h][0] is not None]
            if rets:
                hit = sum(1 for v in rets if v > 0) / len(rets) * 100
                row += f"{mean(rets):>+8.1f} {hit:>4.0f}% "
            else:
                row += f"{'N/A':>8} {'N/A':>5} "
        print(row)


def main():
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0

    store = KlineStore(DB_PATH)
    candles = store.query("BTCUSDT")
    store.close()

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if candles and candles[-1]["open_time"] + WEEK_MS > now_ms:
        candles = candles[:-1]

    first = datetime.fromtimestamp(candles[0]["open_time"] / 1000, tz=timezone.utc).date()
    last = datetime.fromtimestamp(candles[-1]["open_time"] / 1000, tz=timezone.utc).date()
    print(f"BTCUSDT weekly klines: {len(candles)} ({first} ~ {last})")
    print(f"Event: weekly close-to-close gain >= {threshold:.0f}%")

    events = build_events(candles, threshold)
    deduped = dedupe(events)
    baseline = compute_baseline(candles, {e["idx"] for e in events})

    print(f"Events: {len(events)} total, {len(deduped)} after de-cluster (gap > {CLUSTER_GAP}W)")

    print_details(events)
    print_summary("SUMMARY - ALL EVENTS (forward windows may overlap)", events, baseline)
    print_summary(f"SUMMARY - DE-CLUSTERED (gap > {CLUSTER_GAP}W)", deduped, baseline)
    print("\nBaseline = all non-event weeks at the same horizon (includes post-event weeks).")
    print("Excess = event Avg - baseline Avg.")

    for dim, title in [
        ("trend", "TREND CONTEXT (close vs MA20W + slope)"),
        ("pos", "POSITION (drawdown from 52W high)"),
        ("mom4", "PRIOR 4W MOMENTUM (4 weeks before event week)"),
        ("mom13", "PRIOR 13W MOMENTUM (13 weeks before event week)"),
        ("breakout", "BREAKOUT (event close > max close of prior 26W)"),
        ("vol", "VOLUME (event-week volume percentile, 104W window, HVOL >= P80)"),
        ("mag", "MAGNITUDE (event gain bucket)"),
        ("rsi", "WEEKLY RSI(14) at event week"),
    ]:
        print_groups(title, events, dim)

    print_sensitivity(candles)

    print(f"\nNOTE: only {len(events)} events in full history; pattern groups are tiny (marked *).")
    print("Interpret directionally only. Zero events during the 2022 bear market is itself a finding.")


if __name__ == "__main__":
    main()
