"""BTC bottom-detection research: signal discovery via parameter sweep.

Usage:
    uv run python research/btc_bottom.py [start_date] [end_date]
"""

import sys
from collections import defaultdict
from datetime import datetime, timezone
from statistics import median, mean

from core.data import KlineStore
from core.scanner import sweep_params


def parse_date(s: str) -> datetime:
    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s}")


PARAM_GRID = [
    {"rsi_threshold": 30, "vol_percentile": 95, "div_lookback": 90,
     "div_anchor_rsi": 30, "div_rsi_gap": 12, "div_current_rsi_min": 35},
    {"rsi_threshold": 30, "vol_percentile": 95, "div_lookback": 90,
     "div_anchor_rsi": 35, "div_rsi_gap": 15, "div_current_rsi_min": 40},
    {"rsi_threshold": 30, "vol_percentile": 97, "div_lookback": 90,
     "div_anchor_rsi": 35, "div_rsi_gap": 15, "div_current_rsi_min": 40},
    {"rsi_threshold": 32, "vol_percentile": 95, "div_lookback": 90,
     "div_anchor_rsi": 35, "div_rsi_gap": 15, "div_current_rsi_min": 40},
    {"rsi_threshold": 32, "vol_percentile": 97, "div_lookback": 90,
     "div_anchor_rsi": 35, "div_rsi_gap": 15, "div_current_rsi_min": 40},
    {"rsi_threshold": 28, "vol_percentile": 95, "div_lookback": 90,
     "div_anchor_rsi": 35, "div_rsi_gap": 15, "div_current_rsi_min": 40},
]


def main():
    start_str = sys.argv[1] if len(sys.argv) > 1 else "2023-06-01"
    end_str = sys.argv[2] if len(sys.argv) > 2 else "now"

    start_dt = parse_date(start_str)
    end_dt = parse_date(end_str) if end_str.lower() != "now" else datetime.now(timezone.utc)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    store = KlineStore(exchange="binance", interval="1d")
    candles = store.query("BTCUSDT", start_ms, end_ms)
    store.close()

    print(f"BTCUSDT daily klines: {len(candles)} ({start_dt.date()} ~ {end_dt.date()})")

    if len(candles) < 90:
        print("Not enough data (need >= 90 days).")
        return

    results = sweep_params(candles, PARAM_GRID)

    # --- Rank table ---
    print(f"\n{'='*90}")
    print("PARAMETER SWEEP — BTC Bottom Detection Signal Discovery")
    print(f"{'='*90}")
    print(f"{'Rank':>4} {'RSI':>4} {'V%':>4} {'Lb':>4} {'AR':>4} {'Gap':>4} {'CRM':>4} "
          f"{'Score':>8} {'Tot':>5} {'C1':>4} {'3M_best':>9} {'Hit':>5}")
    print("-" * 78)
    for rank, r in enumerate(results, 1):
        p = r["params"]
        print(f"{rank:>4} {p['rsi_threshold']:>4} {p['vol_percentile']:>3.0f}% "
              f"{p['div_lookback']:>3}d {p['div_anchor_rsi']:>4} {p['div_rsi_gap']:>4} "
              f"{p['div_current_rsi_min']:>4} {r['score']:>8.1f} {r['total']:>5} "
              f"{r['c1_count']:>4} {r['c1_avg_q3_best']:>+8.2f}% {r['c1_q3_hit']:>4.0f}%")

    # --- Top 2 combination analysis ---
    for rank, r in enumerate(results[:2], 1):
        p = r["params"]
        print(f"\n{'='*110}")
        print(f"#{rank}  RSI<{p['rsi_threshold']}  Vol>P{p['vol_percentile']:.0f}  "
              f"Div_lb={p['div_lookback']}d  Anchor<{p['div_anchor_rsi']}  "
              f"Gap>={p['div_rsi_gap']}  Cur>={p['div_current_rsi_min']}")
        print(f"     Total={r['total']}  C1={r['c1_count']}  "
              f"C1_3M_best={r['c1_avg_q3_best']:+.2f}%  C1_hit={r['c1_q3_hit']:.0f}%")
        print(f"{'Combination':<20} {'Count':>6} {'Avg_Vol':>14} {'Avg_V%':>9} "
              f"{'1W':>8} {'1M':>8} {'3M_best':>8} {'3M_worst':>9} {'Hit':>5}")
        print("-" * 102)

        for key in sorted(r["stats"].keys(), key=lambda k: -r["stats"][k]["count"]):
            s = r["stats"][key]
            avg_w = mean(s["week_fwd"]) if s["week_fwd"] else 0
            avg_m = mean(s["month_fwd"]) if s["month_fwd"] else 0
            avg_q = mean(s["q3_best"]) if s["q3_best"] else 0
            avg_worst = mean(s["q3_worst"]) if s["q3_worst"] else 0
            q_hit = sum(1 for v in s["q3_best"] if v > 0) / max(len(s["q3_best"]), 1) * 100
            avg_vol = mean(s["volumes"]) if s["volumes"] else 0
            avg_vp = mean(s["vol_pcts"]) if s["vol_pcts"] else 0
            print(f"{key:<20} {s['count']:>6} {avg_vol:>14,.2f} {avg_vp:>8.1f}% "
                  f"{avg_w:>+7.2f}% {avg_m:>+7.2f}% {avg_q:>+7.2f}% {avg_worst:>+8.2f}% {q_hit:>4.0f}%")

    # --- Full signal list of the best result ---
    best = results[0]
    bp = best["params"]
    details = best["details"]
    all_vols = [d["volume"] for d in details]

    print(f"\n\n{'='*170}")
    print(f"BEST: RSI<{bp['rsi_threshold']}  Vol>P{bp['vol_percentile']:.0f} "
          f"Div_lb={bp['div_lookback']}d  Anchor<{bp['div_anchor_rsi']}  "
          f"Gap>={bp['div_rsi_gap']}  Cur>={bp['div_current_rsi_min']}")
    print(f"Total signals: {len(details)}  "
          f"Avg Vol: {mean(all_vols):,.2f}  Median Vol: {median(all_vols):,.2f}")
    print(f"{'='*170}")
    print(f"{'Date':<12} {'Volume':>16} {'V%':>7} {'Low':>12} {'Close':>12} {'RSI':>8} "
          f"{'1W_fwd':>8} {'1M_fwd':>8} {'3M_best':>8} {'3M_worst':>9} {'DivPart':>12} {'Cond'}")
    print("-" * 170)

    sorted_details = sorted(details, key=lambda x: x["volume"], reverse=True)
    for d in sorted_details:
        rsi_s = f"{d['rsi']:.2f}" if d["rsi"] is not None else "N/A"
        vp_s = f"{d['vol_pct']:.1f}%" if d["vol_pct"] is not None else "N/A"
        wf_s = f"{d['week_fwd']:+.2f}%" if d["week_fwd"] is not None else "N/A"
        mf_s = f"{d['month_fwd']:+.2f}%" if d["month_fwd"] is not None else "N/A"
        qb_s = f"{d['q3_best']:+.2f}%" if d["q3_best"] is not None else "N/A"
        qw_s = f"{d['q3_worst']:+.2f}%" if d["q3_worst"] is not None else "N/A"
        dp_s = d["div_partner"] if d["div_partner"] else ""
        cond_s = "+".join(d["conditions"])
        print(f"{d['date']:<12} {d['volume']:>16,.2f} {vp_s:>7} {d['low']:>12,.2f} "
              f"{d['close']:>12,.2f} {rsi_s:>8} {wf_s:>8} {mf_s:>8} {qb_s:>8} "
              f"{qw_s:>9} {dp_s:>12} {cond_s}")


if __name__ == "__main__":
    main()
