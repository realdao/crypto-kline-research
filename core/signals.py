"""Bottom-detection signal functions."""


def check_c1_rsi_oversold(rsi: list[float | None], idx: int, threshold: float = 30) -> bool:
    if rsi[idx] is None:
        return False
    return rsi[idx] < threshold


def check_c2_giant_volume(
    vol_pct: list[float | None], idx: int, percentile: float = 95
) -> bool:
    if vol_pct[idx] is None:
        return False
    return vol_pct[idx] >= percentile


def check_c3_rsi_divergence(
    candles: list[dict],
    rsi: list[float | None],
    idx: int,
    lookback: int = 90,
    anchor_rsi_max: float = 35,
    min_rsi_gap: float = 15,
    current_rsi_min: float = 40,
    local_window: int = 5,
) -> bool:
    """RSI divergence: local price low, RSI recovered, gap from anchor oversold day."""
    if idx < lookback or idx < local_window or idx >= len(candles) - local_window:
        return False
    current_low = candles[idx]["low"]
    current_rsi_val = rsi[idx]
    if current_rsi_val is None or current_rsi_val < current_rsi_min:
        return False

    for d in range(1, local_window + 1):
        if candles[idx - d]["low"] <= current_low or candles[idx + d]["low"] <= current_low:
            return False

    for j in range(idx - lookback, idx):
        rsi_j = rsi[j]
        if rsi_j is None:
            continue
        if rsi_j < anchor_rsi_max:
            if current_low < candles[j]["low"] and (current_rsi_val - rsi_j) >= min_rsi_gap:
                return True
    return False


def find_divergence_anchor(
    candles: list[dict],
    rsi: list[float | None],
    idx: int,
    lookback: int = 90,
    anchor_rsi_max: float = 35,
    min_rsi_gap: float = 15,
    current_rsi_min: float = 40,
) -> int | None:
    """Return index of the divergence anchor day, or None."""
    if idx < lookback:
        return None
    current_low = candles[idx]["low"]
    current_rsi_val = rsi[idx]
    if current_rsi_val is None or current_rsi_val < current_rsi_min:
        return None

    best = None
    best_rsi = float("inf")
    for j in range(idx - lookback, idx):
        rsi_j = rsi[j]
        if rsi_j is None:
            continue
        if rsi_j < anchor_rsi_max:
            if current_low < candles[j]["low"] and (current_rsi_val - rsi_j) >= min_rsi_gap:
                if rsi_j < best_rsi:
                    best_rsi = rsi_j
                    best = j
    return best
