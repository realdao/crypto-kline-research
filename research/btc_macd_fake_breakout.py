"""BTC fake-breakout research: strong bounce right after a MACD death cross.

Usage:
    uv run python -m research.btc_macd_fake_breakout

Context (2026-09-05): BTC death-crossed on 2026-09-02 (first red bar), then
surged +5.1% on 09-03, breaking the stage high (81,479) intraday (high 82,300)
but closing back below it - a same-day failed breakout. The histogram flipped
positive for one day (+17) then turned red again on 09-04 (whipsaw) and is
widening on 09-05; RSI shows a bearish divergence at the 09-03 high.

Scans (daily klines, MACD 12/26/9, hist = DIF - DEA):
    A strict: death cross -> within 3d a day gains >= 4%, breaks stage high
      (max high from golden cross to death cross) intraday, breakout fails
      within 2d (close back below)
    B break-only: same but any gain size
    C bounce-only: within 3d after cross a day gains >= 4% (no breakout req.)
    D whipsaw: hist <= 0 (cross) -> > 0 within 2d -> <= 0 again within 2d

Forward metrics from the pattern day, over the following 30 days.
"""

from datetime import datetime, timezone
from statistics import mean, median

from core.data import KlineStore
from core.metrics import calc_rsi, calc_sma
from research.btc_macd_fade import DB_PATH, DAY_MS, calc_macd, find_phases, to_date

SURGE_WINDOW = 3
FAIL_WINDOW = 2
FWD_DAYS = 30


def fwd_metrics(candles: list[dict], i: int, ref_high: float) -> dict:
    window = candles[i + 1:i + 1 + FWD_DAYS]
    out = {"complete": len(window) == FWD_DAYS}
    if window:
        lo = min(window, key=lambda w: w["low"])
        hi = max(window, key=lambda w: w["high"])
        out["dd"] = (lo["low"] - candles[i]["high"]) / candles[i]["high"] * 100
        out["dd_days"] = (lo["open_time"] - candles[i]["open_time"]) // DAY_MS
        out["run"] = (hi["high"] - ref_high) / ref_high * 100
        out["new_high_close"] = max(w["close"] for w in window) > ref_high
        if out["complete"]:
            out["fwd30"] = (candles[i + FWD_DAYS]["close"]
                            - candles[i]["close"]) / candles[i]["close"] * 100
    return out


def scan_a_b(candles, hist, phases, closes, rsi, ma200, min_gain):
    events = []
    n = len(candles)
    for start, cross in phases:
        hi_idx = max(range(start, cross + 1), key=lambda i: candles[i]["high"])
        local_high = candles[hi_idx]["high"]
        rsi_peak = max(r for r in rsi[start:cross + 1] if r is not None)
        for i in range(cross + 1, min(cross + 1 + SURGE_WINDOW, n)):
            gain = (closes[i] - closes[i - 1]) / closes[i - 1] * 100
            if gain < min_gain or candles[i]["high"] <= local_high:
                continue
            fail = closes[i] < local_high
            if not fail:
                for j in range(i + 1, min(i + 1 + FAIL_WINDOW, n)):
                    if closes[j] < local_high:
                        fail = True
                        break
            if not fail:
                continue
            e = {"date": to_date(candles[i]["open_time"]),
                 "cross_date": to_date(candles[cross]["open_time"]),
                 "gain": gain, "local_high": local_high,
                 "surge_high": candles[i]["high"],
                 "hist_surge": hist[i],
                 "hist_next": hist[i + 1] if i + 1 < n else None,
                 "rsi": rsi[i], "rsi_peak": rsi_peak,
                 "divergence": rsi[i] is not None and rsi[i] < rsi_peak,
                 "bull": ma200[i] is not None and closes[i] > ma200[i],
                 "whipsaw": (hist[i] is not None and hist[i] > 0
                             and i + 1 < n and hist[i + 1] <= 0)}
            e.update(fwd_metrics(candles, i, local_high))
            events.append(e)
            break
    return events


def scan_c(candles, hist, phases, closes, rsi, ma200, min_gain):
    events = []
    n = len(candles)
    for start, cross in phases:
        local_high = max(c["high"] for c in candles[start:cross + 1])
        for i in range(cross + 1, min(cross + 1 + SURGE_WINDOW, n)):
            gain = (closes[i] - closes[i - 1]) / closes[i - 1] * 100
            if gain < min_gain:
                continue
            e = {"date": to_date(candles[i]["open_time"]),
                 "cross_date": to_date(candles[cross]["open_time"]),
                 "gain": gain, "local_high": local_high,
                 "surge_high": candles[i]["high"],
                 "broke_high": candles[i]["high"] > local_high,
                 "bull": ma200[i] is not None and closes[i] > ma200[i],
                 "rsi": rsi[i]}
            e.update(fwd_metrics(candles, i, local_high))
            events.append(e)
            break
    return events


def scan_d(candles, hist, phases, closes, ma200):
    events = []
    n = len(candles)
    for start, cross in phases:
        local_high = max(c["high"] for c in candles[start:cross + 1])
        end = None
        for i in range(cross + 1, min(cross + 3, n)):
            if hist[i] > 0:
                for j in range(i + 1, min(i + 3, n)):
                    if hist[j] <= 0:
                        end = j
                        break
                break
        if end is None:
            continue
        peak = max(candles[cross + 1:end + 1], key=lambda c: c["high"])
        e = {"date": to_date(candles[end]["open_time"]),
             "cross_date": to_date(candles[cross]["open_time"]),
             "peak": peak["high"],
             "bull": ma200[end] is not None and closes[end] > ma200[end]}
        e.update(fwd_metrics(candles, end, peak["high"]))
        events.append(e)
    return events


def stats(title: str, vals: list[float]):
    if not vals:
        print(f"{title}: none")
        return
    v = sorted(vals)
    print(f"{title}  N={len(v)}  mean {mean(v):+.1f}%  median {median(v):+.1f}%  "
          f"range {v[0]:+.1f}% ~ {v[-1]:+.1f}%")


def main():
    store = KlineStore(DB_PATH)
    raw = store.query("BTCUSDT")
    store.close()

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    forming = raw[-1] if raw and raw[-1]["open_time"] + DAY_MS > now_ms else None
    candles = raw[:-1] if forming else raw

    closes = [c["close"] for c in candles]
    dif, dea, hist = calc_macd(closes)
    rsi = calc_rsi(closes, 14)
    ma200 = calc_sma(closes, 200)
    phases = find_phases(hist)

    print(f"BTCUSDT daily: {len(candles)} bars "
          f"({to_date(candles[0]['open_time'])} ~ {to_date(candles[-1]['open_time'])})")

    ev_a = scan_a_b(candles, hist, phases, closes, rsi, ma200, 4.0)
    ev_b = scan_a_b(candles, hist, phases, closes, rsi, ma200, 0.0)
    ev_c = scan_c(candles, hist, phases, closes, rsi, ma200, 4.0)
    ev_d = scan_d(candles, hist, phases, closes, ma200)

    print(f"\n{'=' * 122}")
    print(f"SCAN A - strict: cross -> <=3d day gains >=4%, breaks stage high, fails <=2d "
          f"(N={len(ev_a)})")
    print(f"{'=' * 122}")
    print(f"{'Surge':<12} {'Cross':<12} {'Gain':>6} {'StgHigh':>9} {'SurgeHi':>9} "
          f"{'hist@':>7} {'hist+1':>7} {'RSI(pk)':>8} {'Div':>4} {'Reg':>5} {'Whip':>4} "
          f"{'DD30':>8} {'DDd':>4} {'Run':>7} {'NH':>3} {'F30':>7}")
    print("-" * 122)
    for e in ev_a:
        dd = f"{e['dd']:+.1f}%" if "dd" in e else "N/A"
        ddd = f"{e['dd_days']}" if "dd_days" in e else "-"
        run = f"{e['run']:+.1f}%" if "run" in e else "N/A"
        f30 = f"{e['fwd30']:+.1f}%" if "fwd30" in e else "part."
        hs = f"{e['hist_surge']:+.0f}" if e["hist_surge"] is not None else "-"
        hn = f"{e['hist_next']:+.0f}" if e["hist_next"] is not None else "-"
        print(f"{e['date']:<12} {e['cross_date']:<12} {e['gain']:>+5.1f}% {e['local_high']:>9,.0f} "
              f"{e['surge_high']:>9,.0f} {hs:>7} {hn:>7} {e['rsi']:.0f}({e['rsi_peak']:.0f}) "
              f"{'Y' if e['divergence'] else 'n':>4} {'bull' if e['bull'] else 'bear':>5} "
              f"{'Y' if e['whipsaw'] else 'n':>4} {dd:>8} {ddd:>4} {run:>7} "
              f"{'Y' if e.get('new_high_close') else 'n':>3} {f30:>7}")

    done_a = [e for e in ev_a if e["complete"] and "dd" in e]
    stats("A: drawdown from surge-day high, 30d:", [e["dd"] for e in done_a])
    stats("A: close vs surge close, +30d:", [e["fwd30"] for e in done_a])
    n_nh = sum(1 for e in done_a if e.get("new_high_close"))
    print(f"A: reclaim stage high (close) within 30d: {n_nh}/{len(done_a)}")

    print(f"\n{'=' * 122}")
    print(f"SCAN B - break-only (any gain), fails <=2d (N={len(ev_b)})")
    print(f"{'=' * 122}")
    for e in ev_b:
        dd = f"{e['dd']:+.1f}%" if "dd" in e else "N/A"
        f30 = f"{e['fwd30']:+.1f}%" if "fwd30" in e else "part."
        run = f"{e['run']:+.1f}%" if "run" in e else "N/A"
        print(f"  {e['date']}  gain {e['gain']:+5.1f}%  high {e['surge_high']:,.0f} "
              f"vs stg {e['local_high']:,.0f}  {'bull' if e['bull'] else 'bear'}  "
              f"DD30 {dd:>8}  Run {run:>8}  F30 {f30:>8}")
    done_b = [e for e in ev_b if e["complete"] and "dd" in e]
    stats("B: drawdown from surge-day high, 30d:", [e["dd"] for e in done_b])
    stats("B: close vs surge close, +30d:", [e["fwd30"] for e in done_b])
    for label, want in (("bull", True), ("bear", False)):
        sub = [e for e in done_b if e["bull"] == want]
        stats(f"B-{label}: drawdown 30d:", [e["dd"] for e in sub])

    print(f"\n{'=' * 122}")
    print(f"SCAN C - bounce-only: <=3d after cross a day gains >=4% (N={len(ev_c)})")
    print(f"{'=' * 122}")
    for e in ev_c:
        dd = f"{e['dd']:+.1f}%" if "dd" in e else "N/A"
        f30 = f"{e['fwd30']:+.1f}%" if "fwd30" in e else "part."
        print(f"  {e['date']}  gain {e['gain']:+5.1f}%  broke_high "
              f"{'Y' if e['broke_high'] else 'n'}  {'bull' if e['bull'] else 'bear'}  "
              f"DD30 {dd:>8}  F30 {f30:>8}")
    done_c = [e for e in ev_c if e["complete"] and "dd" in e]
    stats("C: drawdown from bounce-day high, 30d:", [e["dd"] for e in done_c])
    stats("C: close vs bounce close, +30d:", [e["fwd30"] for e in done_c])

    print(f"\n{'=' * 122}")
    print(f"SCAN D - whipsaw: cross -> hist>0 <=2d -> hist<=0 again <=2d (N={len(ev_d)})")
    print(f"{'=' * 122}")
    for e in ev_d:
        dd = f"{e['dd']:+.1f}%" if "dd" in e else "N/A"
        f30 = f"{e['fwd30']:+.1f}%" if "fwd30" in e else "part."
        print(f"  end {e['date']}  cross {e['cross_date']}  peak {e['peak']:,.0f}  "
              f"{'bull' if e['bull'] else 'bear'}  DD30 {dd:>8}  F30 {f30:>8}")
    done_d = [e for e in ev_d if e["complete"] and "dd" in e]
    stats("D: drawdown from whipsaw peak, 30d:", [e["dd"] for e in done_d])
    stats("D: close vs whipsaw-end close, +30d:", [e["fwd30"] for e in done_d])
    for label, want in (("bull", True), ("bear", False)):
        sub = [e for e in done_d if e["bull"] == want]
        stats(f"D-{label}: drawdown 30d:", [e["dd"] for e in sub])

    ctrl = []
    for start, cross in phases:
        local_high = max(c["high"] for c in candles[start:cross + 1])
        window = candles[cross + 1:cross + 1 + FWD_DAYS]
        if len(window) == FWD_DAYS:
            ctrl.append((min(w["low"] for w in window) - local_high)
                        / local_high * 100)
    stats("\nCONTROL - ALL death crosses, dd vs stage high, 30d:", ctrl)

    print(f"\n{'=' * 122}")
    print("CURRENT EPISODE (raw data incl. forming bar)")
    print(f"{'=' * 122}")
    all_closes = [c["close"] for c in raw]
    _, _, all_hist = calc_macd(all_closes)
    all_rsi = calc_rsi(all_closes, 14)
    for i in range(len(raw) - 7, len(raw)):
        tag = " (forming)" if raw[i] is forming else ""
        print(f"{to_date(raw[i]['open_time'])}{tag:<11} close={raw[i]['close']:>9,.0f} "
              f"high={raw[i]['high']:>9,.0f} hist={all_hist[i]:>+8.1f} RSI={all_rsi[i]:.1f}")
    print("\nRead: death cross 09-02; surge 09-03 +5.1% broke stage high 81,479 "
          "(high 82,300), closed back below; hist whipsawed green (+17) for one day "
          "then red again; RSI 73.7 vs phase peak 81.8 = divergence. "
          "Matches SCAN A row 2026-09-03 (partial).")


if __name__ == "__main__":
    main()
