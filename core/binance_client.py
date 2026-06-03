"""Binance REST API client for kline data."""

import requests
from datetime import datetime, timezone

PROXY = "http://192.168.1.9:7897"
BASE_URL = "https://api.binance.com"


def create_session(proxy: str = PROXY) -> requests.Session:
    s = requests.Session()
    s.proxies = {"http": proxy, "https": proxy}
    return s


def fetch_klines(
    symbol: str,
    interval: str = "1d",
    start_ms: int | None = None,
    end_ms: int | None = None,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch klines from Binance. Handles pagination automatically."""
    if session is None:
        session = create_session()

    candles = []
    limit = 1000

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms

        resp = session.get(f"{BASE_URL}/api/v3/klines", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        for item in data:
            candles.append({
                "exchange": "binance",
                "symbol": symbol,
                "interval": interval,
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

    return candles


def fetch_symbols(session: requests.Session | None = None) -> list[str]:
    """Return all USDT-margined spot trading symbols."""
    if session is None:
        session = create_session()

    resp = session.get(f"{BASE_URL}/api/v3/exchangeInfo", timeout=15)
    resp.raise_for_status()
    data = resp.json()

    return sorted([
        s["symbol"]
        for s in data["symbols"]
        if s["status"] == "TRADING"
        and s["quoteAsset"] == "USDT"
        and s.get("isSpotTradingAllowed", False)
    ])


def fetch_24hr_tickers(session: requests.Session | None = None) -> list[dict]:
    """Fetch 24hr ticker for all symbols."""
    if session is None:
        session = create_session()

    resp = session.get(f"{BASE_URL}/api/v3/ticker/24hr", timeout=15)
    resp.raise_for_status()
    data = resp.json()

    return [
        {"symbol": t["symbol"], "volume_24h": float(t["volume"]),
         "quote_volume_24h": float(t["quoteVolume"])}
        for t in data
    ]
