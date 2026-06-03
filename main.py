import sys
from collections import defaultdict
from datetime import datetime, timezone
from statistics import median, mean

import requests

PROXY = "http://192.168.1.9:7897"
BINANCE_KLINE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1d"
RSI_PERIOD = 14
VOL_LOOKBACK_DAYS = 730  # 2 years for volume percentile
EXCLUDE_BEFORE = datetime(2023, 4, 1, tzinfo=timezone.utc)


def fetch_klines(start_ms: int, end_ms: int) -> list[dict]:
    session = requests.Session()
    session.proxies = {"http": PROXY, "https": PROXY}
    all_candles = []
    limit = 1000
    while True:
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
        resp = session.get(BINANCE_KLINE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        for item in data:
            all_candles.append({
                "open_time": item[0],
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
            })
        if len(data) < limit:
            break
        start_ms = data[-1][6] + 1
    return all_candles


def calc_rsi(closes: list[float], period: int = RSI_PERIOD) -> list[float | None]:
    if len(closes) < period + 1:
        return [None] * len(closes)
    rsi_values: list[float | None] = [None] * period
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        gains.append(diff if diff >= 0 else 0.0)
        losses.append(-diff if diff < 0 else 0.0)
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
    rsi_values.append(100.0 - (100.0 / (1.0 + rs)))
    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gain = diff if diff > 0 else 0.0
        loss = -diff if diff < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        rsi_values.append(100.0 - (100.0 / (1.0 + rs)))
    return rsi_values


def calc_volume_percentiles(volumes: list[float], lookback: int = VOL_LOOKBACK_DAYS) -> list[float | None]:
    """For each day, compute volume percentile within lookback window, excluding pre-2023-04."""
    result: list[float | None] = [None] * len(volumes)
    exclude_idx = None
    for i in range(len(volumes)):
        start = max(0, i - lookback + 1)
        window_vols = []
        for j in range(start, i + 1):
            if volumes[j] is not None:
                window_vols.append(volumes[j])
        if not window_vols:
            continue
        vol = volumes[i]
        below = sum(1 for v in window_vols if v < vol)
        result[i] = below / len(window_vols) * 100
    return result


def check_c1_rsi_oversold(rsi_values: list[float | None], idx: int, threshold: float = 30) -> bool:
    if rsi_values[idx] is None:
        return False
    return rsi_values[idx] < threshold


def check_c2_giant_volume(
    vol_pct: list[float | None], idx: int, percentile: float = 95
) -> bool:
    if vol_pct[idx] is None:
        return False
    return vol_pct[idx] >= percentile


def check_c3_rsi_divergence(
    candles: list[dict],
    rsi_values: list[float | None],
    idx: int,
    lookback: int = 90,
    rsi_oversold_threshold: float = 30,
    min_rsi_gap: float = 8,
    current_rsi_min: float = 35,
    local_window: int = 5,
) -> bool:
    """RSI divergence: find prior oversold day in lookback; today's low < prior low
    AND current RSI > prior RSI + gap AND current RSI >= current_rsi_min
    AND today's low is a local minimum (lowest in +/- local_window days)."""
    if idx < lookback or idx < local_window or idx >= len(candles) - local_window:
        return False
    current_low = candles[idx]["low"]
    current_rsi = rsi_values[idx]
    if current_rsi is None or current_rsi < current_rsi_min:
        return False

    # Must be a local low minimum
    for d in range(1, local_window + 1):
        if candles[idx - d]["low"] <= current_low or candles[idx + d]["low"] <= current_low:
            return False

    for j in range(idx - lookback, idx):
        rsi_j = rsi_values[j]
        if rsi_j is None:
            continue
        if rsi_j < rsi_oversold_threshold:
            if current_low < candles[j]["low"] and (current_rsi - rsi_j) >= min_rsi_gap:
                return True
    return False


def find_divergence_partner(
    candles: list[dict],
    rsi_values: list[float | None],
    idx: int,
    lookback: int = 90,
    rsi_oversold_threshold: float = 30,
    min_rsi_gap: float = 8,
    current_rsi_min: float = 35,
) -> int | None:
    if idx < lookback:
        return None
    current_low = candles[idx]["low"]
    current_rsi = rsi_values[idx]
    if current_rsi is None or current_rsi < current_rsi_min:
        return None

    best = None
    best_rsi = float("inf")
    for j in range(idx - lookback, idx):
        rsi_j = rsi_values[j]
        if rsi_j is None:
            continue
        if rsi_j < rsi_oversold_threshold:
            if current_low < candles[j]["low"] and (current_rsi - rsi_j) >= min_rsi_gap:
                if rsi_j < best_rsi:
                    best_rsi = rsi_j
                    best = j
    return best


def parse_date(s: str) -> datetime:
    formats = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s}")


def run_with_params(
    candles: list[dict],
    rsi_values: list[float | None],
    vol_pct: list[float | None],
    *,
    rsi_threshold: float = 30,
    vol_percentile: float = 95,
    div_lookback: int = 90,
    div_rsi_threshold: float = 30,
    div_rsi_gap: float = 8,
    div_current_rsi_min: float = 35,
) -> tuple[list[dict], dict]:
    """Run detection and return details + combination stats."""
    bottom_details = []

    for i in range(len(candles)):
        c1 = check_c1_rsi_oversold(rsi_values, i, rsi_threshold)
        c2 = check_c2_giant_volume(vol_pct, i, vol_percentile)
        c3 = check_c3_rsi_divergence(candles, rsi_values, i, div_lookback, div_rsi_threshold, div_rsi_gap, div_current_rsi_min)

        if not (c1 or c2 or c3):
            continue

        c = candles[i]
        dt = datetime.fromtimestamp(c["open_time"] / 1000, tz=timezone.utc)

        week_fwd = None
        if i + 7 < len(candles):
            week_fwd = (candles[i + 7]["close"] - c["close"]) / c["close"] * 100
        month_fwd = None
        if i + 30 < len(candles):
            month_fwd = (candles[i + 30]["close"] - c["close"]) / c["close"] * 100

        # 3-month: best exit = max close in 90d; worst = min low in 90d
        q3_best = None
        q3_worst = None
        if i + 90 < len(candles):
            window = candles[i + 1 : i + 91]
            max_close = max(w["close"] for w in window)
            min_low = min(w["low"] for w in window)
            q3_best = (max_close - c["close"]) / c["close"] * 100
            q3_worst = (min_low - c["low"]) / c["low"] * 100  # drawdown from entry low

        week_prev = None
        if i >= 7:
            week_prev = (c["close"] - candles[i - 7]["close"]) / candles[i - 7]["close"] * 100
        month_prev = None
        if i >= 30:
            month_prev = (c["close"] - candles[i - 30]["close"]) / candles[i - 30]["close"] * 100

        div_partner = None
        if c3:
            partner = find_divergence_partner(candles, rsi_values, i, div_lookback, div_rsi_threshold, div_rsi_gap, div_current_rsi_min)
            if partner is not None:
                div_partner = datetime.fromtimestamp(candles[partner]["open_time"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

        conds = []
        if c1:
            conds.append("C1")
        if c2:
            conds.append("C2")
        if c3:
            conds.append("C3")

        bottom_details.append({
            "date": dt.strftime("%Y-%m-%d"),
            "volume": c["volume"],
            "close": c["close"],
            "low": c["low"],
            "rsi": round(rsi_values[i], 2) if rsi_values[i] is not None else None,
            "vol_pct": round(vol_pct[i], 1) if vol_pct[i] is not None else None,
            "conditions": conds,
            "cond_key": "+".join(sorted(conds)),
            "week_fwd": week_fwd,
            "month_fwd": month_fwd,
            "q3_best": q3_best,
            "q3_worst": q3_worst,
            "week_prev": week_prev,
            "month_prev": month_prev,
            "div_partner": div_partner,
        })

    # Stats by combination
    combo_stats = defaultdict(lambda: {
        "count": 0,
        "volumes": [],
        "vol_pcts": [],
        "week_fwd": [],
        "month_fwd": [],
        "q3_best": [],
        "q3_worst": [],
    })
    for d in bottom_details:
        key = d["cond_key"]
        combo_stats[key]["count"] += 1
        combo_stats[key]["volumes"].append(d["volume"])
        if d["vol_pct"]:
            combo_stats[key]["vol_pcts"].append(d["vol_pct"])
        if d["week_fwd"] is not None:
            combo_stats[key]["week_fwd"].append(d["week_fwd"])
        if d["month_fwd"] is not None:
            combo_stats[key]["month_fwd"].append(d["month_fwd"])
        if d["q3_best"] is not None:
            combo_stats[key]["q3_best"].append(d["q3_best"])
        if d["q3_worst"] is not None:
            combo_stats[key]["q3_worst"].append(d["q3_worst"])

    return bottom_details, dict(combo_stats)


def evaluate_params(
    bottom_details: list[dict],
    combo_stats: dict,
) -> float:
    """Score a parameter set. Higher = better.
    Focus: C1-containing combos are the most meaningful per user guidance.
    Score = weighted avg of (q3_fwd * C1_weight) across combos.
    We want high q3_fwd from C1-containing, decent-volume signals.
    """
    score = 0.0
    total_signals = len(bottom_details)

    for key, stats in combo_stats.items():
        has_c1 = "C1" in key
        has_c2 = "C2" in key
        has_c3 = "C3" in key
        cnt = stats["count"]
        avg_q3 = mean(stats["q3_best"]) if stats["q3_best"] else 0
        q3_hit = sum(1 for v in stats["q3_best"] if v > 0) / max(len(stats["q3_best"]), 1)

        # C1 is the most reliable signal
        weight = 1.0
        if has_c1 and has_c2:
            weight = 3.0
        elif has_c1:
            weight = 2.5
        elif has_c3 and has_c2:
            weight = 1.5
        elif has_c3:
            weight = 0.8
        elif has_c2:
            weight = 0.5

        combo_score = cnt * weight * (avg_q3 * 0.5 + q3_hit * 100 * 0.5)
        score += combo_score

    # Penalize too many signals
    score -= max(0, total_signals - 60) * 0.5
    return score


def main():
    start_str = sys.argv[1] if len(sys.argv) > 1 else "2023-06-01"
    end_str = sys.argv[2] if len(sys.argv) > 2 else "now"
    mode = sys.argv[3] if len(sys.argv) > 3 else "sweep"

    start_dt = parse_date(start_str)
    if end_str.lower() == "now":
        end_dt = datetime.now(timezone.utc)
    else:
        end_dt = parse_date(end_str)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"Fetching {SYMBOL} daily klines from {start_dt.date()} to {end_dt.date()} ...")
    candles = fetch_klines(start_ms, end_ms)
    print(f"Fetched {len(candles)} daily candles.")

    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]
    rsi_values = calc_rsi(closes)
    vol_pct = calc_volume_percentiles(volumes)

    print(f"Computing volume percentiles (2yr lookback, excluding pre-2023-04)...")

    if mode == "sweep":
        # Parameter sweep
        print("\n" + "=" * 120)
        print("PARAMETER SWEEP — Finding best condition combination for BTC bottom detection")
        print("=" * 120)

        param_grid = [
            # (rsi_threshold, vol_percentile, div_lookback, div_rsi_threshold, div_rsi_gap, div_current_rsi_min)
            (30, 95, 60, 30, 12, 35),
            (30, 95, 60, 35, 15, 40),
            (30, 95, 90, 30, 12, 35),
            (30, 95, 90, 35, 15, 40),
            (30, 97, 60, 35, 15, 40),
            (30, 97, 90, 35, 15, 40),
            (28, 95, 90, 35, 15, 40),
            (28, 97, 90, 35, 15, 40),
            (32, 95, 90, 35, 15, 40),
            (32, 97, 90, 35, 15, 40),
        ]

        results = []
        for params in param_grid:
            rsi_t, vol_p, div_lb, div_rsi_t, div_gap, div_cur_min = params
            details, stats = run_with_params(
                candles, rsi_values, vol_pct,
                rsi_threshold=rsi_t,
                vol_percentile=vol_p,
                div_lookback=div_lb,
                div_rsi_threshold=div_rsi_t,
                div_rsi_gap=div_gap,
                div_current_rsi_min=div_cur_min,
            )
            score = evaluate_params(details, stats)
            # Focus metrics on C1-containing combos
            c1_q3 = []
            for d in details:
                if "C1" in d["cond_key"] and d["q3_best"] is not None:
                    c1_q3.append(d["q3_best"])

            results.append({
                "params": params,
                "score": score,
                "total": len(details),
                "c1_count": sum(1 for d in details if "C1" in d["cond_key"]),
                "c1_avg_q3": mean(c1_q3) if c1_q3 else 0,
                "c1_q3_hit": sum(1 for v in c1_q3 if v > 0) / max(len(c1_q3), 1) * 100 if c1_q3 else 0,
                "stats": stats,
            })

        results.sort(key=lambda r: r["score"], reverse=True)

        print(f"\n{'Rank':>4} {'RSI':>4} {'V%':>4} {'Lb':>4} {'DR':>4} {'Gap':>4} {'CRM':>4} {'Score':>8} {'Tot':>5} {'C1':>4} {'C1_Bst':>8} {'Hit':>5}")
        print("-" * 80)
        for rank, r in enumerate(results, 1):
            p = r["params"]
            print(f"{rank:>4} {p[0]:>4} {p[1]:>3.0f}% {p[2]:>3}d {p[3]:>4} {p[4]:>4} {p[5]:>4} {r['score']:>8.1f} {r['total']:>5} {r['c1_count']:>4} {r['c1_avg_q3']:>+6.2f}% {r['c1_q3_hit']:>4.0f}%")

        # Show top 3 in detail
        print(f"\n{'='*140}")
        print("TOP 3 PARAMETER SETS — Detailed Combination Analysis")
        for rank, r in enumerate(results[:3], 1):
            p = r["params"]
            print(f"\n{'='*140}")
            print(f"#{rank}  RSI_th={p[0]}  Vol>{p[1]:.0f}%ile  Div_lb={p[2]}d  Div_RSI_th={p[3]}  Gap={p[4]}  CurRSI>={p[5]}")
            print(f"     Total={r['total']}  C1={r['c1_count']}  C1_3M={r['c1_avg_q3']:+.2f}%  C1_hit={r['c1_q3_hit']:.0f}%")
            print(f"{'Combination':<20} {'Count':>6} {'Avg_Vol':>14} {'Avg_Vol%':>10} {'Avg_1W':>9} {'Avg_1M':>9} {'3M_best':>10} {'3M_worst':>11} {'Hit':>5}")
            print("-" * 102)

            stats = r["stats"]
            for key in sorted(stats.keys(), key=lambda k: -stats[k]["count"]):
                s = stats[key]
                avg_w = mean(s["week_fwd"]) if s["week_fwd"] else 0
                avg_m = mean(s["month_fwd"]) if s["month_fwd"] else 0
                avg_q = mean(s["q3_best"]) if s["q3_best"] else 0
                q_hit = sum(1 for v in s["q3_best"] if v > 0) / max(len(s["q3_best"]), 1) * 100
                avg_worst = mean(s["q3_worst"]) if s["q3_worst"] else 0
                avg_vol = mean(s["volumes"]) if s["volumes"] else 0
                avg_vp = mean(s["vol_pcts"]) if s["vol_pcts"] else 0
                print(f"{key:<20} {s['count']:>6} {avg_vol:>14,.2f} {avg_vp:>9.1f}% {avg_w:>+8.2f}% {avg_m:>+8.2f}% {avg_q:>+8.2f}% {avg_worst:>+8.2f}% {q_hit:>6.0f}%")

        # --- Final output: best params, all signals ---
        best = results[0]
        bp = best["params"]
        print(f"\n\n{'='*165}")
        print(f"BEST: RSI_th={bp[0]}  Vol>{bp[1]:.0f}%ile(2yr)  Div_lb={bp[2]}d  Div_RSI_th={bp[3]}  Gap={bp[4]}  CurRSI>={bp[5]}")
        print(f"ALL BOTTOM SIGNALS (sorted by volume)")
        print(f"{'='*165}")

        details, _ = run_with_params(
            candles, rsi_values, vol_pct,
            rsi_threshold=bp[0],
            vol_percentile=bp[1],
            div_lookback=bp[2],
            div_rsi_threshold=bp[3],
            div_rsi_gap=bp[4],
            div_current_rsi_min=bp[5],
        )

        all_vols = [d["volume"] for d in details]
        print(f"Total: {len(details)}  Avg Vol: {mean(all_vols):,.2f}  Median Vol: {median(all_vols):,.2f}")
        print(f"{'='*165}")
        print(f"{'Date':<12} {'Volume':>16} {'V/MA':>7} {'Low':>12} {'Close':>12} {'RSI':>8} {'1W_fwd':>8} {'1M_fwd':>8} {'3M_best':>8} {'3M_worst':>9} {'DivPartner':>12} {'Cond'}")
        print("-" * 165)
        sorted_details = sorted(details, key=lambda x: x["volume"], reverse=True)
        for d in sorted_details:
            rsi_str = f"{d['rsi']:.2f}" if d["rsi"] is not None else "N/A"
            vp_str = f"{d['vol_pct']:.1f}%" if d["vol_pct"] is not None else "N/A"
            wf_str = f"{d['week_fwd']:+.2f}%" if d["week_fwd"] is not None else "N/A"
            mf_str = f"{d['month_fwd']:+.2f}%" if d["month_fwd"] is not None else "N/A"
            qb_str = f"{d['q3_best']:+.2f}%" if d["q3_best"] is not None else "N/A"
            qw_str = f"{d['q3_worst']:+.2f}%" if d["q3_worst"] is not None else "N/A"
            dp_str = d["div_partner"] if d["div_partner"] else ""
            cond_str = "+".join(d["conditions"])
            print(f"{d['date']:<12} {d['volume']:>16,.2f} {vp_str:>7} {d['low']:>12,.2f} {d['close']:>12,.2f} {rsi_str:>8} {wf_str:>8} {mf_str:>8} {qb_str:>8} {qw_str:>9} {dp_str:>12} {cond_str}")

    else:
        # Default: best known params
        details, _ = run_with_params(
            candles, rsi_values, vol_pct,
            rsi_threshold=30, vol_percentile=95, div_lookback=90, div_rsi_threshold=30,
        )
        all_vols = [d["volume"] for d in details]
        print(f"\nTotal signals: {len(details)}  Avg Vol: {mean(all_vols):,.2f}  Median Vol: {median(all_vols):,.2f}")
        print(f"{'Date':<12} {'Volume':>16} {'Low':>12} {'Close':>12} {'RSI':>8} {'1W_fwd':>8} {'1M_fwd':>8} {'3M_best':>8} {'3M_worst':>9} {'DivPartner':>12} {'Cond'}")
        print("-" * 145)
        sorted_details = sorted(details, key=lambda x: x["volume"], reverse=True)
        for d in sorted_details:
            rsi_str = f"{d['rsi']:.2f}" if d["rsi"] is not None else "N/A"
            wf_str = f"{d['week_fwd']:+.2f}%" if d["week_fwd"] is not None else "N/A"
            mf_str = f"{d['month_fwd']:+.2f}%" if d["month_fwd"] is not None else "N/A"
            qb_str = f"{d['q3_best']:+.2f}%" if d["q3_best"] is not None else "N/A"
            qw_str = f"{d['q3_worst']:+.2f}%" if d["q3_worst"] is not None else "N/A"
            dp_str = d["div_partner"] if d["div_partner"] else ""
            cond_str = "+".join(d["conditions"])
            print(f"{d['date']:<12} {d['volume']:>16,.2f} {d['low']:>12,.2f} {d['close']:>12,.2f} {rsi_str:>8} {wf_str:>8} {mf_str:>8} {qb_str:>8} {qw_str:>9} {dp_str:>12} {cond_str}")


if __name__ == "__main__":
    main()
