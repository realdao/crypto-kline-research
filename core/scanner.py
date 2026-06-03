"""Parameter-scanning engine for strategy discovery."""

from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean

from core.metrics import calc_rsi, calc_volume_percentiles, forward_best_worst
from core.signals import (
    check_c1_rsi_oversold,
    check_c2_giant_volume,
    check_c3_rsi_divergence,
    find_divergence_anchor,
)


def run_scan(
    candles: list[dict],
    *,
    rsi_threshold: float = 30,
    vol_percentile: float = 95,
    div_lookback: int = 90,
    div_anchor_rsi: float = 35,
    div_rsi_gap: float = 15,
    div_current_rsi_min: float = 40,
) -> tuple[list[dict], dict]:
    """Scan candles and return (signal_details, combination_stats)."""
    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]
    rsi_values = calc_rsi(closes)
    vol_pct = calc_volume_percentiles(volumes)

    details = []

    for i in range(len(candles)):
        c1 = check_c1_rsi_oversold(rsi_values, i, rsi_threshold)
        c2 = check_c2_giant_volume(vol_pct, i, vol_percentile)
        c3 = check_c3_rsi_divergence(
            candles, rsi_values, i,
            div_lookback, div_anchor_rsi, div_rsi_gap, div_current_rsi_min
        )

        if not (c1 or c2 or c3):
            continue

        c = candles[i]
        dt = datetime.fromtimestamp(c["open_time"] / 1000, tz=timezone.utc)

        conds = []
        if c1:
            conds.append("C1")
        if c2:
            conds.append("C2")
        if c3:
            conds.append("C3")

        week_fwd = None
        if i + 7 < len(candles):
            week_fwd = (candles[i + 7]["close"] - c["close"]) / c["close"] * 100

        month_fwd = None
        if i + 30 < len(candles):
            month_fwd = (candles[i + 30]["close"] - c["close"]) / c["close"] * 100

        q3_best, q3_worst = forward_best_worst(candles, i, 90)

        week_prev = None
        if i >= 7:
            week_prev = (c["close"] - candles[i - 7]["close"]) / candles[i - 7]["close"] * 100

        month_prev = None
        if i >= 30:
            month_prev = (c["close"] - candles[i - 30]["close"]) / candles[i - 30]["close"] * 100

        div_partner = None
        if c3:
            partner = find_divergence_anchor(
                candles, rsi_values, i,
                div_lookback, div_anchor_rsi, div_rsi_gap, div_current_rsi_min
            )
            if partner is not None:
                div_partner = datetime.fromtimestamp(
                    candles[partner]["open_time"] / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d")

        vol_ratio = c["volume"] / vol_pct[i] if vol_pct[i] else None

        details.append({
            "date": dt.strftime("%Y-%m-%d"),
            "open_time": c["open_time"],
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

    # Build combination stats
    combo_stats = defaultdict(lambda: {
        "count": 0,
        "volumes": [],
        "vol_pcts": [],
        "week_fwd": [],
        "month_fwd": [],
        "q3_best": [],
        "q3_worst": [],
    })
    for d in details:
        key = d["cond_key"]
        combo_stats[key]["count"] += 1
        combo_stats[key]["volumes"].append(d["volume"])
        if d["vol_pct"] is not None:
            combo_stats[key]["vol_pcts"].append(d["vol_pct"])
        if d["week_fwd"] is not None:
            combo_stats[key]["week_fwd"].append(d["week_fwd"])
        if d["month_fwd"] is not None:
            combo_stats[key]["month_fwd"].append(d["month_fwd"])
        if d["q3_best"] is not None:
            combo_stats[key]["q3_best"].append(d["q3_best"])
        if d["q3_worst"] is not None:
            combo_stats[key]["q3_worst"].append(d["q3_worst"])

    return details, dict(combo_stats)


def score_params(details: list[dict], combo_stats: dict) -> float:
    """Score a parameter set. Higher is better."""
    score = 0.0
    total = len(details)

    for key, stats in combo_stats.items():
        has_c1 = "C1" in key
        has_c2 = "C2" in key
        cnt = stats["count"]
        avg_q = mean(stats["q3_best"]) if stats["q3_best"] else 0
        q3_hit = sum(1 for v in stats["q3_best"] if v > 0) / max(len(stats["q3_best"]), 1)

        weight = 1.0
        if has_c1 and has_c2:
            weight = 3.0
        elif has_c1:
            weight = 2.5
        elif has_c2:
            weight = 0.5

        score += cnt * weight * (avg_q * 0.5 + q3_hit * 100 * 0.5)

    score -= max(0, total - 80) * 2
    return score


def sweep_params(
    candles: list[dict],
    param_grid: list[dict],
) -> list[dict]:
    """Run parameter sweep and return sorted results."""
    results = []
    for params in param_grid:
        details, stats = run_scan(candles, **params)
        score = score_params(details, stats)

        c1_q3 = [d["q3_best"] for d in details if "C1" in d["cond_key"] and d["q3_best"] is not None]

        results.append({
            "params": params,
            "score": score,
            "total": len(details),
            "c1_count": sum(1 for d in details if "C1" in d["cond_key"]),
            "c1_avg_q3_best": mean(c1_q3) if c1_q3 else 0,
            "c1_q3_hit": sum(1 for v in c1_q3 if v > 0) / max(len(c1_q3), 1) * 100 if c1_q3 else 0,
            "details": details,
            "stats": stats,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results
