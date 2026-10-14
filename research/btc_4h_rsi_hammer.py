"""BTC 4h: drop>1.5% & RSI<25 & next candle hammer — forward 24-bar stats.

Usage:
    uv run python -m research.btc_4h_rsi_hammer

Signal: candle i drops >1.5% with RSI(14)<25, and candle i+1 is a hammer.
Entry = hammer close. Forward window = next 24 4h candles (4 days).

Sections:
    1. Current status (latest RSI, recent candles)
    2. Signal list (MFE24 = max gain, MAE24 = max drawdown from entry)
    3. Summary vs baseline (random 24-bar windows)
    4. Hammer-definition sensitivity grid
"""

from datetime import datetime, timezone
from statistics import mean, median

from core.data import KlineStore
from core.metrics import calc_rsi

H4_MS = 4 * 60 * 60 * 1000
FORWARD_BARS = 24
DROP_PCT = -1.5
RSI_MAX = 25.0
DB = ".data/crypto/binance-BTCUSDT-4h.db"


def body_shadows(c: dict) -> tuple[float, float, float]:
    body = abs(c["close"] - c["open"])
    lower = min(c["open"], c["close"]) - c["low"]
    upper = c["high"] - max(c["open"], c["close"])
    return body, lower, upper


def is_hammer(c: dict, shadow_ratio: float = 2.0,
              upper_max: float = 1.0) -> bool:
    body, lower, upper = body_shadows(c)
    if body <= 0:
        return False
    if lower < shadow_ratio * body:
        return False
    if upper > upper_max * body:
        return False
    return True


def fwd_mfe_mae(candles: list[dict], i: int, bars: int) -> tuple:
    if i + bars >= len(candles):
        return None, None
    entry = candles[i]["close"]
    window = candles[i + 1:i + bars + 1]
    mfe = (max(c["high"] for c in window) - entry) / entry * 100
    mae = (min(c["low"] for c in window) - entry) / entry * 100
    return mfe, mae


def percentile(vals: list[float], p: float) -> float:
    vals = sorted(vals)
    k = (len(vals) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def pct(v: float | None) -> str:
    return "N/A" if v is None else f"{v:+.1f}%"


def find_signals(candles: list[dict], rsi: list[float | None],
                 shadow_ratio: float, upper_max: float) -> list[int]:
    idxs = []
    for i in range(len(candles) - 1):
        c = candles[i]
        nxt = candles[i + 1]
        drop = (c["close"] - c["open"]) / c["open"] * 100
        if drop <= DROP_PCT and rsi[i] is not None and rsi[i] < RSI_MAX \
                and is_hammer(nxt, shadow_ratio, upper_max):
            idxs.append(i + 1)
    return idxs


def print_current(candles: list[dict], rsi: list[float | None]):
    print(f"\n{'=' * 100}")
    print("CURRENT STATUS")
    print(f"{'=' * 100}")
    last = candles[-1]
    dt = datetime.fromtimestamp(last["open_time"] / 1000, tz=timezone.utc)
    print(f"Data end: {dt.strftime('%Y-%m-%d %H:%M')} UTC  "
          f"bars={len(candles)}  close={last['close']:,.0f}")
    print(f"RSI(14) last 5: {[f'{v:.1f}' if v is not None else 'N/A' for v in rsi[-5:]]}")
    for j in range(max(0, len(candles) - 4), len(candles)):
        c = candles[j]
        d = datetime.fromtimestamp(c["open_time"] / 1000, tz=timezone.utc)
        chg = (c["close"] - c["open"]) / c["open"] * 100
        ham = "HAMMER" if is_hammer(c) else ""
        print(f"  {d.strftime('%Y-%m-%d %H:%M')}  O={c['open']:,.0f} "
              f"C={c['close']:,.0f} ({chg:+.2f}%)  {ham}")


def print_signals(candles: list[dict], rsi: list[float | None], idxs: list[int],
                  shadow_ratio: float, upper_max: float):
    print(f"\n{'=' * 150}")
    print(f"SIGNALS - drop>{-DROP_PCT:.1f}% & RSI<{RSI_MAX:.0f} & next hammer "
          f"(shadow>={shadow_ratio}x body, upper<={upper_max}x body)  N={len(idxs)}")
    print(f"{'=' * 150}")
    print(f"{'#':>3} {'Date(UTC)':<17} {'Drop%':>7} {'RSI':>5} "
          f"{'HamO':>10} {'HamC':>10} {'Entry':>10} {'MFE24':>8} {'MAE24':>8}")
    print("-" * 150)
    for n, i in enumerate(idxs, 1):
        drop = (candles[i - 1]["close"] - candles[i - 1]["open"]) \
            / candles[i - 1]["open"] * 100
        ham = candles[i]
        d = datetime.fromtimestamp(ham["open_time"] / 1000, tz=timezone.utc)
        mfe, mae = fwd_mfe_mae(candles, i, FORWARD_BARS)
        print(f"{n:>3} {d.strftime('%Y-%m-%d %H:%M'):<17} {drop:>+6.2f}% "
              f"{rsi[i - 1]:>5.1f} {ham['open']:>10,.0f} {ham['close']:>10,.0f} "
              f"{ham['close']:>10,.0f} {pct(mfe):>8} {pct(mae):>8}")


def summarize(idxs: list[int], candles: list[dict], rsi: list[float | None]):
    mfes = []
    maes = []
    for i in idxs:
        mfe, mae = fwd_mfe_mae(candles, i, FORWARD_BARS)
        if mfe is not None:
            mfes.append(mfe)
            maes.append(mae)

    base_mfes = []
    base_maes = []
    sig_set = set(idxs)
    for i in range(len(candles)):
        if i in sig_set:
            continue
        mfe, mae = fwd_mfe_mae(candles, i, FORWARD_BARS)
        if mfe is not None:
            base_mfes.append(mfe)
            base_maes.append(mae)

    print(f"\n{'=' * 100}")
    print(f"SUMMARY - next {FORWARD_BARS} bars (={FORWARD_BARS * 4}h) after hammer close")
    print(f"{'=' * 100}")
    print(f"{'Group':<14} {'N':>6} {'Avg':>8} {'Med':>8} {'P10':>8} {'P25':>8} "
          f"{'P75':>8} {'P90':>8} {'Best':>8} {'Worst':>8}")
    print("-" * 100)
    for label, vals, base in [("MFE24 signal", mfes, base_mfes),
                              ("MAE24 signal", maes, base_maes)]:
        if not vals:
            print(f"{label:<14} {'0':>6}")
            continue
        print(f"{label:<14} {len(vals):>6} {mean(vals):>+8.1f} {median(vals):>+8.1f} "
              f"{percentile(vals, 10):>+8.1f} {percentile(vals, 25):>+8.1f} "
              f"{percentile(vals, 75):>+8.1f} {percentile(vals, 90):>+8.1f} "
              f"{max(vals):>+8.1f} {min(vals):>+8.1f}")

    print(f"\nBaseline (all {len(base_mfes)} non-signal bars):")
    print(f"{'':<14} {'Avg':>8} {'Med':>8} {'P10':>8} {'P25':>8} "
          f"{'P75':>8} {'P90':>8} {'Best':>8} {'Worst':>8}")
    print(f"{'MFE24 base':<14} {mean(base_mfes):>+8.1f} {median(base_mfes):>+8.1f} "
          f"{percentile(base_mfes, 10):>+8.1f} {percentile(base_mfes, 25):>+8.1f} "
          f"{percentile(base_mfes, 75):>+8.1f} {percentile(base_mfes, 90):>+8.1f} "
          f"{max(base_mfes):>+8.1f} {min(base_mfes):>+8.1f}")
    print(f"{'MAE24 base':<14} {mean(base_maes):>+8.1f} {median(base_maes):>+8.1f} "
          f"{percentile(base_maes, 10):>+8.1f} {percentile(base_maes, 25):>+8.1f} "
          f"{percentile(base_maes, 75):>+8.1f} {percentile(base_maes, 90):>+8.1f} "
          f"{max(base_maes):>+8.1f} {min(base_maes):>+8.1f}")

    if mfes:
        print(f"\nExceedance probabilities (signal):")
        print(f"  MFE24 > +5% : {sum(1 for v in mfes if v > 5) / len(mfes) * 100:>5.1f}%")
        print(f"  MFE24 > +10%: {sum(1 for v in mfes if v > 10) / len(mfes) * 100:>5.1f}%")
        print(f"  MAE24 < -5% : {sum(1 for v in maes if v < -5) / len(maes) * 100:>5.1f}%")
        print(f"  MAE24 < -10%: {sum(1 for v in maes if v < -10) / len(maes) * 100:>5.1f}%")


def print_sensitivity(candles: list[dict], rsi: list[float | None]):
    print(f"\n{'=' * 100}")
    print("SENSITIVITY - hammer shadow ratio x upper-shadow cap (N / avg MFE24 / avg MAE24)")
    print(f"{'=' * 100}")
    header = f"{'shdw\\up':>10} "
    for up in [0.5, 1.0, 2.0]:
        header += f"{'<=%.1fx' % up:>24}"
    print(header)
    for ratio in [1.5, 2.0, 2.5, 3.0]:
        row = f"{'>=%.1fx' % ratio:>10} "
        for up in [0.5, 1.0, 2.0]:
            idxs = find_signals(candles, rsi, ratio, up)
            mfes, maes = [], []
            for i in idxs:
                mfe, mae = fwd_mfe_mae(candles, i, FORWARD_BARS)
                if mfe is not None:
                    mfes.append(mfe)
                    maes.append(mae)
            if mfes:
                row += f"{'%d %+.1f%% %+.1f%%' % (len(mfes), mean(mfes), mean(maes)):>24}"
            else:
                row += f"{'0 N/A':>24}"
        print(row)


def main():
    store = KlineStore(DB)
    candles = store.query("BTCUSDT")
    store.close()

    closes = [c["close"] for c in candles]
    rsi = calc_rsi(closes)

    print(f"BTCUSDT 4h klines: {len(candles)}")

    print_current(candles, rsi)

    idxs = find_signals(candles, rsi, 2.0, 1.0)
    print_signals(candles, rsi, idxs, 2.0, 1.0)
    summarize(idxs, candles, rsi)

    print_sensitivity(candles, rsi)

    print(f"\nNOTE: entry = hammer close; MFE/MAE over next {FORWARD_BARS} 4h bars "
          f"(={FORWARD_BARS * 4}h). Decline = (close-open)/open of candle i.")


if __name__ == "__main__":
    main()
