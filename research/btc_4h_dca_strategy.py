"""BTC 4h DCA strategy: RSI<25 & hammer entry, 3-batch scale-in, exit-rule comparison.

Usage:
    uv run python -m research.btc_4h_dca_strategy

Entry trigger: 4h RSI(14)<25 and the next candle is a hammer (lower shadow
>=1.5x body, upper shadow <=2x body). No decline filter.
Batch 1 buys at hammer close; batches 2/3 buy when price falls `step%` below the
previous batch, only within `window` bars of the first entry. Signals closer
than MIN_SPACING bars to the previous one are skipped (no re-entry).

Sections:
    1. Current status
    2. Signal list (with 24-bar dip depth after entry1)
    3. Exit-rule comparison on single batch (1-0-0)
    4. DCA grid: step x ratio (fixed rule trail12, window 24)
    5. Full ranking: rule x ratio x step x window (top 15 by avg return)
    6. Best config: trade detail + position-sizing / equity simulation
    7. Robustness: RSI<25 vs <30, loose vs strict hammer
"""

from datetime import datetime, timezone
from itertools import product
from statistics import mean, median

from core.data import KlineStore
from core.metrics import calc_rsi, calc_sma

H4_MS = 4 * 60 * 60 * 1000
DB = ".data/crypto/binance-BTCUSDT-4h.db"
MAX_HOLD = 300
FEE_PCT = 0.1
MIN_SPACING = 12

RSI_MAX = 25.0
SHADOW_RATIO = 1.5
UPPER_MAX = 2.0

RULES = [
    ("hold24", {"kind": "hold", "n": 24}),
    ("hold48", {"kind": "hold", "n": 48}),
    ("hold72", {"kind": "hold", "n": 72}),
    ("hold120", {"kind": "hold", "n": 120}),
    ("hold240", {"kind": "hold", "n": 240}),
    ("hammerlow", {"kind": "hammer_low"}),
    ("rsi70", {"kind": "rsi_over", "x": 70}),
    ("rsi60", {"kind": "rsi_over", "x": 60}),
    ("sma20", {"kind": "sma_cross", "p": 20}),
    ("sma50", {"kind": "sma_cross", "p": 50}),
    ("tp5_sl3", {"kind": "tp_sl", "tp": 5, "sl": 3}),
    ("tp8_sl5", {"kind": "tp_sl", "tp": 8, "sl": 5}),
    ("tp10_sl5", {"kind": "tp_sl", "tp": 10, "sl": 5}),
    ("tp15_sl8", {"kind": "tp_sl", "tp": 15, "sl": 8}),
    ("trail8", {"kind": "trail", "gb": 8}),
    ("trail12", {"kind": "trail", "gb": 12}),
    ("trail15", {"kind": "trail", "gb": 15}),
    ("tp10_trail10", {"kind": "tp_trail", "tp": 10, "gb": 10}),
]

RATIOS = {
    "1-0-0": [1, 0, 0],
    "1-1-1": [1, 1, 1],
    "1-1-2": [1, 1, 2],
    "1-2-3": [1, 2, 3],
    "1-2-4": [1, 2, 4],
    "1-3-6": [1, 3, 6],
    "2-1-1": [2, 1, 1],
    "3-2-1": [3, 2, 1],
    "2-2-1": [2, 2, 1],
}

STEPS = [3.0, 5.0, 8.0, 10.0, 12.0]
WINDOWS = [12, 24, 48]


def body_shadows(c: dict) -> tuple:
    body = abs(c["close"] - c["open"])
    lower = min(c["open"], c["close"]) - c["low"]
    upper = c["high"] - max(c["open"], c["close"])
    return body, lower, upper


def is_hammer(c: dict, shadow_ratio: float = 2.0, upper_max: float = 1.0) -> bool:
    body, lower, upper = body_shadows(c)
    if body <= 0:
        return False
    return lower >= shadow_ratio * body and upper <= upper_max * body


def find_signals(candles: list[dict], rsi: list[float | None],
                 rsi_max: float, shadow_ratio: float,
                 upper_max: float) -> list[int]:
    idxs = []
    for i in range(len(candles) - 1):
        if rsi[i] is None or rsi[i] >= rsi_max:
            continue
        if is_hammer(candles[i + 1], shadow_ratio, upper_max):
            idxs.append(i + 1)
    spaced = []
    for i in idxs:
        if not spaced or i - spaced[-1] >= MIN_SPACING:
            spaced.append(i)
    return spaced


def percentile(vals: list[float], p: float) -> float:
    vals = sorted(vals)
    k = (len(vals) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def pct(v: float) -> str:
    return f"{v:+.1f}%"


def find_exit(candles: list[dict], rsi: list[float | None],
              sma: dict[int, list[float | None]], start_idx: int,
              exit_start: int, entry: float, rule: dict) -> tuple[int, float]:
    n = len(candles)
    cap = min(start_idx + MAX_HOLD, n - 1)
    kind = rule["kind"]

    if kind == "hold":
        j = min(max(start_idx + rule["n"], exit_start), n - 1)
        return j, candles[j]["close"]

    if kind == "hammer_low":
        stop = candles[start_idx]["low"]
        for j in range(start_idx + 1, cap + 1):
            if candles[j]["close"] <= stop:
                px = stop if candles[j]["open"] > stop else candles[j]["open"]
                return j, px
        return cap, candles[cap]["close"]

    if kind == "rsi_over":
        for j in range(exit_start, cap + 1):
            if rsi[j] is not None and rsi[j] >= rule["x"]:
                return j, candles[j]["close"]
        return cap, candles[cap]["close"]

    if kind == "sma_cross":
        s = sma[rule["p"]]
        for j in range(exit_start, cap + 1):
            if s[j] is not None and candles[j]["close"] < s[j]:
                return j, candles[j]["close"]
        return cap, candles[cap]["close"]

    if kind == "tp_sl":
        tp_p = entry * (1 + rule["tp"] / 100)
        sl_p = entry * (1 - rule["sl"] / 100)
        for j in range(exit_start, cap + 1):
            c = candles[j]
            if c["low"] <= sl_p:
                px = sl_p if c["open"] > sl_p else c["open"]
                return j, px
            if c["high"] >= tp_p:
                px = tp_p if c["open"] < tp_p else c["open"]
                return j, px
        return cap, candles[cap]["close"]

    if kind == "trail":
        peak = max(c["high"] for c in candles[start_idx:exit_start + 1])
        for j in range(exit_start + 1, cap + 1):
            c = candles[j]
            peak = max(peak, c["high"])
            if c["close"] <= peak * (1 - rule["gb"] / 100):
                return j, c["close"]
        return cap, candles[cap]["close"]

    if kind == "tp_trail":
        peak = max(c["high"] for c in candles[start_idx:exit_start + 1])
        tp_p = entry * (1 + rule["tp"] / 100)
        for j in range(exit_start + 1, cap + 1):
            c = candles[j]
            if c["high"] >= tp_p:
                px = tp_p if c["open"] < tp_p else c["open"]
                return j, px
            peak = max(peak, c["high"])
            if c["close"] <= peak * (1 - rule["gb"] / 100):
                return j, c["close"]
        return cap, candles[cap]["close"]

    raise ValueError(f"Unknown rule kind: {kind}")


def sim_trade(candles: list[dict], rsi: list[float | None],
              sma: dict[int, list[float | None]], start_idx: int,
              weights: list[int], step_pct: float, win_bars: int,
              rule: dict) -> dict:
    n = len(candles)
    entry_prices = [candles[start_idx]["close"]]
    entry_bars = [start_idx]
    last_px = entry_prices[0]
    stop = candles[start_idx]["low"] if rule.get("kind") == "hammer_low" else None
    early_exit = None

    for k in range(1, len(weights)):
        if weights[k] <= 0:
            continue
        target = last_px * (1 - step_pct / 100)
        filled = False
        for j in range(start_idx + 1, min(start_idx + win_bars + 1, n)):
            c = candles[j]
            if stop is not None and c["close"] <= stop:
                px = stop if c["open"] > stop else c["open"]
                early_exit = (j, px)
                break
            if c["low"] <= target:
                px = c["open"] if c["open"] < target else target
                entry_prices.append(px)
                entry_bars.append(j)
                last_px = px
                filled = True
                break
        if early_exit is not None or not filled:
            break

    wsum = sum(weights[:len(entry_prices)])
    entry = sum(w * p for w, p in zip(weights, entry_prices)) / wsum
    exit_start = entry_bars[-1]

    if early_exit is not None:
        exit_bar, exit_px = early_exit
    else:
        exit_bar, exit_px = find_exit(candles, rsi, sma, start_idx,
                                      exit_start, entry, rule)
    ret = (exit_px - entry) / entry * 100 - 2 * FEE_PCT
    mae_from = min(exit_start, exit_bar)
    mae = min((c["low"] - entry) / entry * 100
              for c in candles[mae_from:exit_bar + 1])
    mfe = max((c["high"] - entry) / entry * 100
              for c in candles[mae_from:exit_bar + 1])
    dip = min((c["low"] - entry_prices[0]) / entry_prices[0] * 100
              for c in candles[start_idx:exit_start + 1])
    dt = datetime.fromtimestamp(candles[exit_bar]["open_time"] / 1000,
                                tz=timezone.utc)
    return {"ret": ret, "mae": mae, "mfe": mfe, "dip": dip,
            "fills": len(entry_prices), "bars": exit_bar - start_idx,
            "exit_bar": exit_bar, "entry": entry, "exit_px": exit_px,
            "exit_dt": dt.strftime("%Y-%m-%d %H:%M"),
            "start_idx": start_idx,
            "entry1": entry_prices[0]}


def aggregate(trades: list[dict]) -> dict:
    rets = [t["ret"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    return {
        "n": len(rets),
        "avg": mean(rets) if rets else 0.0,
        "med": median(rets) if rets else 0.0,
        "win": sum(1 for r in rets if r > 0) / len(rets) * 100 if rets else 0.0,
        "pf": pf,
        "mae_mean": mean(t["mae"] for t in trades) if trades else 0.0,
        "mae_worst": min(t["mae"] for t in trades) if trades else 0.0,
        "bars": mean(t["bars"] for t in trades) if trades else 0.0,
        "fills": mean(t["fills"] for t in trades) if trades else 0.0,
    }


def equity_sim(trades: list[dict], alloc: float) -> tuple[float, float]:
    eq = 1.0
    peak = 1.0
    maxdd = 0.0
    for t in trades:
        eq *= 1 + alloc * t["ret"] / 100
        peak = max(peak, eq)
        maxdd = min(maxdd, (eq - peak) / peak * 100)
    return eq, maxdd


def run_config(candles: list[dict], rsi: list[float | None],
               sma: dict[int, list[float | None]], idxs: list[int],
               weights: list[int], step_pct: float, win_bars: int,
               rule: dict) -> list[dict]:
    return [sim_trade(candles, rsi, sma, i, weights, step_pct, win_bars, rule)
            for i in idxs]


def print_current(candles: list[dict], rsi: list[float | None]):
    print(f"\n{'=' * 100}")
    print("CURRENT STATUS")
    print(f"{'=' * 100}")
    last = candles[-1]
    dt = datetime.fromtimestamp(last["open_time"] / 1000, tz=timezone.utc)
    print(f"Data end: {dt.strftime('%Y-%m-%d %H:%M')} UTC  bars={len(candles)}  "
          f"close={last['close']:,.0f}")
    print(f"RSI(14) last 5: {[f'{v:.1f}' if v is not None else 'N/A' for v in rsi[-5:]]}")
    for j in range(max(0, len(candles) - 4), len(candles)):
        c = candles[j]
        d = datetime.fromtimestamp(c["open_time"] / 1000, tz=timezone.utc)
        chg = (c["close"] - c["open"]) / c["open"] * 100
        tag = "HAMMER" if is_hammer(c, SHADOW_RATIO, UPPER_MAX) else ""
        print(f"  {d.strftime('%Y-%m-%d %H:%M')}  O={c['open']:,.0f} "
              f"C={c['close']:,.0f} ({chg:+.2f}%)  {tag}")


def print_signals(candles: list[dict], rsi: list[float | None],
                  idxs: list[int]):
    print(f"\n{'=' * 110}")
    print(f"BASE SIGNALS - RSI<{RSI_MAX:.0f} & next hammer (loose: lower>=1.5x "
          f"body, upper<=2x)  N={len(idxs)}  spacing>={MIN_SPACING} bars")
    print(f"{'=' * 110}")
    print(f"{'#':>3} {'Date(UTC)':<17} {'RSI':>5} {'Entry1':>10} "
          f"{'24barLow':>10} {'dip24':>8} {'5%trig':>7}")
    print("-" * 110)
    for n, i in enumerate(idxs, 1):
        d = datetime.fromtimestamp(candles[i]["open_time"] / 1000,
                                   tz=timezone.utc)
        entry1 = candles[i]["close"]
        win = candles[i + 1:i + 25]
        low24 = min(c["low"] for c in win)
        dip = (low24 - entry1) / entry1 * 100
        trig5 = sum(1 for _ in range(3) if low24 <= entry1 * (1 - 0.05))
        print(f"{n:>3} {d.strftime('%Y-%m-%d %H:%M'):<17} "
              f"{rsi[i - 1]:>5.1f} {entry1:>10,.0f} {low24:>10,.0f} "
              f"{dip:>+7.1f}% {trig5:>7}")


def print_exit_rules(candles: list[dict], rsi: list[float | None],
                     sma: dict[int, list[float | None]], idxs: list[int]):
    print(f"\n{'=' * 130}")
    print(f"EXIT RULES - single batch (ratio 1-0-0, N={len(idxs)} trades)")
    print(f"{'=' * 130}")
    print(f"{'Rule':<16} {'Avg':>8} {'Med':>8} {'Win%':>6} {'PF':>7} "
          f"{'MAEavg':>8} {'MAEwst':>8} {'Bars':>6}")
    print("-" * 130)
    rows = []
    for name, rule in RULES:
        trades = run_config(candles, rsi, sma, idxs, [1, 0, 0], 5.0, 24, rule)
        a = aggregate(trades)
        rows.append((name, a))
    rows.sort(key=lambda r: -r[1]["avg"])
    for name, a in rows:
        pf_s = f"{a['pf']:.2f}" if a["pf"] != float("inf") else "inf"
        print(f"{name:<16} {a['avg']:>+8.1f} {a['med']:>+8.1f} {a['win']:>5.0f}% "
              f"{pf_s:>7} {a['mae_mean']:>+8.1f} {a['mae_worst']:>+8.1f} "
              f"{a['bars']:>6.0f}")


def print_dca_grid(candles: list[dict], rsi: list[float | None],
                   sma: dict[int, list[float | None]], idxs: list[int],
                   rule_name: str = "trail12"):
    rule = dict(RULES)[rule_name]
    print(f"\n{'=' * 130}")
    print(f"DCA GRID - rule={rule_name}, window=24 bars  (cell = avg% / win%)")
    print(f"{'=' * 130}")
    header = f"{'step\\ratio':>12} " + "".join(f"{k:>14}" for k in RATIOS)
    print(header)
    for step in STEPS:
        row = f"{'-%.0f%%' % step:>12} "
        for k, w in RATIOS.items():
            trades = run_config(candles, rsi, sma, idxs, w, step, 24, rule)
            a = aggregate(trades)
            row += f"{'+%.1f %.0f%%' % (a['avg'], a['win']):>14}"
        print(row)

    print(f"\nWindow sensitivity (step=5%, rule={rule_name}):")
    print(f"{'ratio':<10} " + "".join(f"{'w%d' % wn:>18}" for wn in WINDOWS))
    for k, w in RATIOS.items():
        row = f"{k:<10} "
        for wn in WINDOWS:
            trades = run_config(candles, rsi, sma, idxs, w, 5.0, wn, rule)
            a = aggregate(trades)
            row += f"{'+%.1f%% f%.1f' % (a['avg'], a['fills']):>18}"
        print(row)


def rank_all(candles: list[dict], rsi: list[float | None],
             sma: dict[int, list[float | None]], idxs: list[int]) -> list[tuple]:
    results = []
    for (name, rule), (rk, w), step, wn in \
            product(RULES, RATIOS.items(), STEPS, WINDOWS):
        trades = run_config(candles, rsi, sma, idxs, w, step, wn, rule)
        a = aggregate(trades)
        results.append((a["avg"], a, name, rk, step, wn))
    results.sort(key=lambda r: -r[0])
    return results


def print_ranking(results: list[tuple]):
    print(f"\n{'=' * 140}")
    print("TOP 15 CONFIGURATIONS by avg return (rule x ratio x step x window)")
    print(f"{'=' * 140}")
    print(f"{'Rank':>4} {'Rule':<16} {'Ratio':>8} {'Step':>6} {'Win':>5} "
          f"{'Avg':>8} {'Med':>8} {'Win%':>6} {'PF':>6} {'MAEwst':>8} {'Fills':>5}")
    print("-" * 140)
    for rank, (_, a, name, rk, step, wn) in enumerate(results[:15], 1):
        pf_s = f"{a['pf']:.2f}" if a["pf"] != float("inf") else "inf"
        print(f"{rank:>4} {name:<16} {rk:>8} {'%.0f%%' % step:>6} {wn:>4} "
              f"{a['avg']:>+8.1f} {a['med']:>+8.1f} {a['win']:>5.0f}% "
              f"{pf_s:>6} {a['mae_worst']:>+8.1f} {a['fills']:>5.1f}")


def print_best_per_rule(results: list[tuple]):
    best = {}
    for _, a, name, rk, step, wn in results:
        if name not in best:
            best[name] = (a, rk, step, wn)
    print(f"\n{'=' * 120}")
    print("BEST CONFIG PER EXIT RULE")
    print(f"{'=' * 120}")
    print(f"{'Rule':<16} {'Ratio':>8} {'Step':>6} {'Win':>5} {'Avg':>8} "
          f"{'Med':>8} {'Win%':>6} {'PF':>6} {'MAEwst':>8} {'Fills':>5}")
    print("-" * 120)
    for name, (a, rk, step, wn) in best.items():
        pf_s = f"{a['pf']:.2f}" if a["pf"] != float("inf") else "inf"
        print(f"{name:<16} {rk:>8} {'%.0f%%' % step:>6} {wn:>4} "
              f"{a['avg']:>+8.1f} {a['med']:>+8.1f} {a['win']:>5.0f}% "
              f"{pf_s:>6} {a['mae_worst']:>+8.1f} {a['fills']:>5.1f}")


def print_best_detail(candles: list[dict], rsi: list[float | None],
                      sma: dict[int, list[float | None]], idxs: list[int],
                      top: tuple):
    _, a, name, rk, step, wn = top
    rule = dict(RULES)[name]
    w = RATIOS[rk]
    trades = run_config(candles, rsi, sma, idxs, w, step, wn, rule)

    print(f"\n{'=' * 150}")
    print(f"BEST CONFIG DETAIL - {name} / ratio {rk} / step {step:.0f}% / "
          f"window {wn}  (avg={a['avg']:+.1f}%)")
    print(f"{'=' * 150}")
    print(f"{'Date(UTC)':<17} {'Entry1':>10} {'Entry':>10} {'F':>2} {'Exit':>10} "
          f"{'ExitDt':<17} {'Ret':>8} {'Lowest':>8} {'Dip':>8} {'Bars':>5}")
    print("-" * 150)
    for t in trades:
        d = datetime.fromtimestamp(candles[t["start_idx"]]["open_time"] / 1000,
                                   tz=timezone.utc)
        print(f"{d.strftime('%Y-%m-%d %H:%M'):<17} {t['entry1']:>10,.0f} "
              f"{t['entry']:>10,.0f} {t['fills']:>2} {t['exit_px']:>10,.0f} "
              f"{t['exit_dt']:<17} {t['ret']:>+7.1f}% {t['mae']:>+7.1f}% "
              f"{t['dip']:>+7.1f}% {t['bars']:>5}")
    print("Lowest = lowest bar low vs blended entry (+ = never below entry). "
          "Dip = lowest bar low vs entry1 during scale-in.")

    dips = sorted(t["dip"] for t in trades)
    rets = [t["ret"] for t in trades]
    print(f"\nDip (vs entry1) dist: med={median(dips):+.1f}%  "
          f"p90={percentile(dips, 90):+.1f}%  worst={min(dips):+.1f}%")

    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    b = (mean(wins) / abs(mean(losses))) if losses else float("inf")
    p = sum(1 for r in rets if r > 0) / len(rets)
    kelly = p - (1 - p) / b if b != float("inf") and b > 0 else p
    print(f"Kelly reference: win%={p * 100:.0f}%, b(avgW/avgL)={b:.2f}, "
          f"f*={kelly * 100:.0f}%, half-Kelly={kelly * 50:.0f}% (N={len(rets)} "
          f"trades, treat as rough reference)")

    print(f"\nEquity simulation (per-trade allocation, compounded, "
          f"fees {FEE_PCT}%/side):")
    print(f"{'Alloc':>7} {'FinalX':>9} {'EqMaxDD':>9} {'Avg/tr':>8}")
    print("-" * 40)
    eq_trades = decluster_trades(trades)
    for alloc in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        eq, mdd = equity_sim(eq_trades, alloc)
        print(f"{alloc * 100:>6.0f}% {eq:>9.2f}x {mdd:>+8.1f}% "
              f"{a['avg'] * alloc:>+8.2f}%")
    print(f"({len(eq_trades)} non-overlapping trades in equity sim)")

    sorted_rets = sorted(rets)
    p10_loss = abs(sorted_rets[0]) if sorted_rets else 0.0
    p20_loss = abs(percentile(rets, 20)) if len(rets) >= 5 else p10_loss
    print(f"\nRisk-budget sizing (alloc = budget / |loss|):")
    for rb in [1.0, 2.0]:
        print(f"  budget {rb:.0f}%: on worst trade -> {rb / p10_loss * 100:.0f}% "
              f"of portfolio; on P20 loss -> {rb / p20_loss * 100:.0f}%")
    print("Note: equity DD already includes the full scale-in dip on "
          "deployed batches.")


def decluster_trades(trades: list[dict]) -> list[dict]:
    kept = []
    for t in trades:
        if not kept or t["start_idx"] > kept[-1]["exit_bar"]:
            kept.append(t)
    return kept


def print_robustness(candles: list[dict], rsi: list[float | None],
                     sma: dict[int, list[float | None]]):
    rule = dict(RULES)["trail12"]
    print(f"\n{'=' * 130}")
    print("ROBUSTNESS - RSI threshold x hammer strictness "
          f"(rule=trail12, ratio=1-2-3, step=5%, window=24)")
    print(f"{'=' * 130}")
    print(f"{'Config':<28} {'N':>4} {'Avg':>8} {'Med':>8} {'Win%':>6} "
          f"{'PF':>6} {'MAEwst':>8} {'Fills':>6}")
    print("-" * 130)
    for rsi_max, (sr, um, label) in [
        (25.0, (1.5, 2.0, "loose")),
        (25.0, (2.0, 1.0, "strict")),
        (30.0, (1.5, 2.0, "loose")),
        (30.0, (2.0, 1.0, "strict")),
    ]:
        idxs = find_signals(candles, rsi, rsi_max, sr, um)
        if not idxs:
            continue
        trades = run_config(candles, rsi, sma, idxs, RATIOS["1-2-3"], 5.0, 24, rule)
        a = aggregate(trades)
        pf_s = f"{a['pf']:.2f}" if a["pf"] != float("inf") else "inf"
        print(f"{'RSI<%.0f + %s hammer' % (rsi_max, label):<28} {a['n']:>4} "
              f"{a['avg']:>+8.1f} {a['med']:>+8.1f} {a['win']:>5.0f}% "
              f"{pf_s:>6} {a['mae_worst']:>+8.1f} {a['fills']:>6.1f}")


def main():
    store = KlineStore(DB)
    candles = store.query("BTCUSDT")
    store.close()

    closes = [c["close"] for c in candles]
    rsi = calc_rsi(closes)
    sma = {p: calc_sma(closes, p) for p in [20, 50]}

    print(f"BTCUSDT 4h klines: {len(candles)}")

    idxs = find_signals(candles, rsi, RSI_MAX, SHADOW_RATIO, UPPER_MAX)

    print_current(candles, rsi)
    print_signals(candles, rsi, idxs)
    print_exit_rules(candles, rsi, sma, idxs)
    print_dca_grid(candles, rsi, sma, idxs)

    results = rank_all(candles, rsi, sma, idxs)
    print_ranking(results)
    print_best_per_rule(results)
    print_best_detail(candles, rsi, sma, idxs, results[0])

    print_robustness(candles, rsi, sma)

    print(f"\nNOTE: entry1 = hammer close; batches fill at limit price when a bar "
          f"low reaches `step%` below the last batch price, within the window. "
          f"Exit rules (except hold*) start after the last batch. Fees "
          f"{FEE_PCT}%/side, no slippage. Signals de-clustered with "
          f">={MIN_SPACING} bar spacing. N={len(idxs)} base signals, "
          f"2017-08 ~ 2026-10.")


if __name__ == "__main__":
    main()
