"""Technical indicators: RSI, volume percentile, moving averages."""

from math import inf


def calc_rsi(closes: list[float], period: int = 14) -> list[float | None]:
    if len(closes) < period + 1:
        return [None] * len(closes)

    rsi: list[float | None] = [None] * period
    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        gains.append(diff if diff >= 0 else 0.0)
        losses.append(-diff if diff < 0 else 0.0)

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    rs = avg_gain / avg_loss if avg_loss != 0 else inf
    rsi.append(100.0 - (100.0 / (1.0 + rs)))

    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gain = diff if diff > 0 else 0.0
        loss = -diff if diff < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else inf
        rsi.append(100.0 - (100.0 / (1.0 + rs)))

    return rsi


def calc_volume_percentiles(volumes: list[float], lookback: int = 730) -> list[float | None]:
    """For each day, compute volume percentile within sliding lookback window."""
    result: list[float | None] = [None] * len(volumes)
    for i in range(len(volumes)):
        start = max(0, i - lookback + 1)
        window = volumes[start:i + 1]
        if not window:
            continue
        below = sum(1 for v in window if v < volumes[i])
        result[i] = below / len(window) * 100
    return result


def calc_sma(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * (period - 1)
    for i in range(period - 1, len(values)):
        result.append(sum(values[i - period + 1:i + 1]) / period)
    return result


def calc_local_min(values: list[float], window: int = 5) -> list[bool]:
    """Mark local minima in a series."""
    n = len(values)
    result = [False] * n
    for i in range(window, n - window):
        val = values[i]
        if all(val <= values[i - d] and val <= values[i + d] for d in range(1, window + 1)):
            result[i] = True
    return result


def forward_best_worst(
    candles: list[dict], idx: int, days: int = 90
) -> tuple[float | None, float | None]:
    """Return (best_exit_%chg, worst_drawdown_%chg) within days after idx."""
    if idx + days >= len(candles):
        return None, None
    window = candles[idx + 1:idx + days + 1]
    max_close = max(w["close"] for w in window)
    min_low = min(w["low"] for w in window)
    entry_close = candles[idx]["close"]
    entry_low = candles[idx]["low"]
    best = (max_close - entry_close) / entry_close * 100
    worst = (min_low - entry_low) / entry_low * 100
    return best, worst
