"""BTC short-surge research: drawdown after a 3-7 day rally of 20-30%.

Usage:
    uv run python -m research.btc_short_surge

Context: BTC rose from ~62751 (2026-08-17) to ~79500 high (+26.7% in 4 days).
This script finds historical analogs and measures what happened next.

Event definition (scanned on daily klines, close-to-close):
    - Lookback window L in {3,4,5,6,7} days, surge = close[i]/close[i-L] - 1
    - Keep the FIRST day where 20% <= surge <= 30% (per lookback, dedup by
      starting a new event only after price has closed below the event-day
      close or 30 days have passed)

Measured from the event day's close, over the following 30 days:
    - max further upside (MFE)  = max(high[+1..+30]) / event_close - 1
    - drawdown (MAE)            = min(low[+1..+30]) / event_close - 1
    - retrace-of-surge          = drawdown measured from the post-event peak
    - days-to-max-drawdown

Outputs:
    1. Current 2026-08 episode vs history
    2. Event detail table
    3. Summary: drawdown range / mean / median across all analogs
    4. Conditioned breakdowns (bull/bear regime, 20W MA, RSI, event magnitude)
"""

from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, median

from core.data import KlineStore
from core.metrics import calc_rsi, calc_sma

DB_PATH = ".data/crypto/binance-BTCUSDT-4h.db"
DAY_MS = 24 * 60 * 60 * 1000
LOOKBACKS = [3, 4, 5, 6, 7]
SURGE_MIN, SURGE_MAX = 20.0, 30.0
FWD_DAYS = 30
COOLDOWN = 30


def to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def load_daily(candles_4h: list[dict]) -> list[dict]:
    """Aggregate 4h klines into UTC daily klines (extends history to 2017)."""
    by_day: dict[int, dict] = {}
    for k in candles_4h:
        day = k["open_time"] // DAY_MS * DAY_MS
        d = by_day.get(day)
        if d is None:
            by_day[day] = {"symbol": k["symbol"], "open_time": day,
                           "open": k["open"], "high": k["high"],
                           "low": k["low"], "close": k["close"],
                           "volume": k["volume"]}
        else:
            d["high"] = max(d["high"], k["high"])
            d["low"] = min(d["low"], k["low"])
            d["close"] = k["close"]
            d["volume"] += k["volume"]
    return [by_day[day] for day in sorted(by_day)]


def build_events(candles: list[dict]) -> list[dict]:
    closes = [c["close"] for c in candles]
    ma20w = calc_sma(closes, 140)
    rsi14 = calc_rsi(closes, 14)

    events = []
    last_event_idx = -10**9
    for i in range(max(LOOKBACKS), len(candles)):
        if i - last_event_idx <= COOLDOWN:
            continue

        surge = None
        lb_used = None
        for lb in LOOKBACKS:
            s = (closes[i] - closes[i - lb]) / closes[i - lb] * 100
            if SURGE_MIN <= s <= SURGE_MAX:
                surge = s
                lb_used = lb
                break

        if surge is None:
            continue

        last_event_idx = i

        entry = closes[i]
        window = candles[i + 1:i + 1 + FWD_DAYS]
        complete = len(window) == FWD_DAYS
        if window:
            fwd_hi = max(w["high"] for w in window)
            fwd_lo = min(w["low"] for w in window)
            mfe = (fwd_hi - entry) / entry * 100
            mae = (fwd_lo - entry) / entry * 100
            retrace = (fwd_lo - fwd_hi) / fwd_hi * 100
            dd_day = to_date(
                min(window, key=lambda w: w["low"])["open_time"])
        else:
            mfe = mae = retrace = None
            dd_day = "N/A"

        ma = ma20w[i]
        trend = "N/A" if ma is None else ("ABOVE_MA" if entry >= ma else "BELOW_MA")
        r = rsi14[i]
        rsi_tag = "N/A" if r is None else (
            "RSI>80" if r > 80 else ("RSI70-80" if r > 70 else "RSI<=70"))
        mag = "20-25%" if surge < 25 else "25-30%"

        events.append({
            "idx": i,
            "date": to_date(candles[i]["open_time"]),
            "lb": lb_used,
            "surge": surge,
            "entry": entry,
            "complete": complete,
            "mfe": mfe,
            "mae": mae,
            "retrace": retrace,
            "dd_day": dd_day,
            "trend": trend,
            "rsi": rsi_tag,
            "mag": mag,
        })
    return events


def print_details(events: list[dict]):
    print(f"\n{'=' * 120}")
    print("EVENT DETAIL - measured from event-day close, following 30 days")
    print(f"{'=' * 120}")
    print(f"{'Date':<12} {'Lb':>3} {'Surge':>7} {'Entry':>9} {'MFE':>8} {'Drawdown':>9} "
          f"{'Retrace':>8} {'DD_Date':<12} {'Trend':<10} {'RSI':<8} {'Mag':<7} {'OK'}")
    print("-" * 120)
    for e in events:
        mfe = f"{e['mfe']:+.1f}%" if e["mfe"] is not None else "N/A"
        mae = f"{e['mae']:+.1f}%" if e["mae"] is not None else "N/A"
        rt = f"{e['retrace']:+.1f}%" if e["retrace"] is not None else "N/A"
        print(f"{e['date']:<12} {e['lb']:>3} {e['surge']:>+6.1f}% {e['entry']:>9,.0f} "
              f"{mfe:>8} {mae:>9} {rt:>8} {e['dd_day']:<12} {e['trend']:<10} "
              f"{e['rsi']:<8} {e['mag']:<7} {'Y' if e['complete'] else 'partial'}")


def print_summary(title: str, events: list[dict]):
    done = [e for e in events if e["complete"] and e["mae"] is not None]
    print(f"\n{'=' * 90}")
    print(f"{title}  (N={len(done)} complete)")
    print(f"{'=' * 90}")
    if not done:
        print("No complete events.")
        return
    maes = [e["mae"] for e in done]
    mfes = [e["mfe"] for e in done]
    rts = [e["retrace"] for e in done]
    print(f"{'Metric':<22} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8} {'P25':>8} {'P75':>8}")
    print("-" * 90)
    for name, vals in [("MFE (further up)", mfes), ("Drawdown vs entry", maes),
                       ("Retrace from peak", rts)]:
        v = sorted(vals)
        n = len(v)
        p25 = v[int(n * 0.25)]
        p75 = v[min(n - 1, int(n * 0.75))]
        print(f"{name:<22} {mean(v):>+8.1f} {median(v):>+8.1f} {v[0]:>+8.1f} "
              f"{v[-1]:>+8.1f} {p25:>+8.1f} {p75:>+8.1f}")
    dd_counts = defaultdict(int)
    for m in maes:
        b = ("< -30%" if m < -30 else "-30~-20%" if m < -20 else
             "-20~-10%" if m < -10 else "-10~-5%" if m < -5 else "> -5%")
        dd_counts[b] += 1
    print("\nDrawdown distribution:")
    for b in ["< -30%", "-30~-20%", "-20~-10%", "-10~-5%", "> -5%"]:
        n = dd_counts[b]
        bar = "#" * n
        print(f"  {b:<10} {n:>3}  {bar}")


def print_groups(title: str, events: list[dict], dim: str):
    groups = defaultdict(list)
    for e in events:
        groups[e[dim]].append(e)
    print(f"\n{'=' * 100}")
    print(f"CONDITION - {title}")
    print(f"{'=' * 100}")
    print(f"{'Group':<12} {'N':>3} {'DD_mean':>9} {'DD_med':>8} {'DD_min':>8} "
          f"{'DD_max':>8} {'MFE_mean':>9} {'Retrace':>9}")
    print("-" * 100)
    for label in sorted(groups):
        evs = [e for e in groups[label] if e["complete"] and e["mae"] is not None]
        if not evs:
            continue
        maes = [e["mae"] for e in evs]
        row = (f"{label:<12} {len(evs):>3} {mean(maes):>+9.1f} {median(maes):>+8.1f} "
               f"{min(maes):>+8.1f} {max(maes):>+8.1f} {mean(e['mfe'] for e in evs):>+9.1f} "
               f"{mean(e['retrace'] for e in evs):>+9.1f}")
        if len(evs) < 5:
            row += "  *small"
        print(row)


def main():
    store = KlineStore(DB_PATH)
    candles = load_daily(store.query("BTCUSDT"))
    store.close()

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if candles and candles[-1]["open_time"] + DAY_MS > now_ms:
        candles = candles[:-1]  # drop the still-forming day

    print(f"BTCUSDT daily klines (from 4h): {len(candles)} "
          f"({to_date(candles[0]['open_time'])} ~ {to_date(candles[-1]['open_time'])})")
    print(f"Event: first day when close/close-{LOOKBACKS}d gain within "
          f"[{SURGE_MIN:.0f}%, {SURGE_MAX:.0f}%], cooldown {COOLDOWN}d")

    events = build_events(candles)
    print(f"Events found: {len(events)}")

    print_details(events)

    done = [e for e in events if e["complete"]]
    print_summary("SUMMARY - ALL COMPLETE EVENTS", done)
    print_groups("TREND (close vs 20W MA)", done, "trend")
    print_groups("DAILY RSI(14) at event", done, "rsi")
    print_groups("SURGE MAGNITUDE", done, "mag")
    print_groups("LOOKBACK LENGTH", done, "lb")

    match = [e for e in done if e["trend"] == "ABOVE_MA" and e["rsi"] == "RSI>80"]
    print(f"\nCLOSEST MATCH to current episode (ABOVE_MA + RSI>80): N={len(match)}")
    for e in match:
        print(f"  {e['date']}  surge {e['surge']:+.1f}%/{e['lb']}d  MFE {e['mfe']:+.1f}%  "
              f"DD {e['mae']:+.1f}%  retrace {e['retrace']:+.1f}%")
    if match:
        maes = [e["mae"] for e in match]
        print(f"  -> Drawdown: mean {mean(maes):+.1f}%, median {median(maes):+.1f}%, "
              f"range {min(maes):+.1f}% ~ {max(maes):+.1f}%")

    print("\nNOTE: drawdown = min low of next 30 days vs event close (negative).")
    print("      retrace  = min low vs post-event max high (peak-to-trough).")


if __name__ == "__main__":
    main()
