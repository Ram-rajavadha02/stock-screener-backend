"""
Mock OHLCV data generator for 10 Indian large-cap stocks.
Uses NumPy seeded random walks so data is deterministic per symbol.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 10 representative stocks with realistic base prices (INR)
STOCKS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "WIPRO", "BAJFINANCE", "AXISBANK", "KOTAKBANK",
]

BASE_PRICES: dict[str, float] = {
    "RELIANCE":  2500.0,
    "TCS":       3800.0,
    "INFY":      1500.0,
    "HDFCBANK":  1600.0,
    "ICICIBANK":  950.0,
    "SBIN":       620.0,
    "WIPRO":      450.0,
    "BAJFINANCE": 7000.0,
    "AXISBANK":  1100.0,
    "KOTAKBANK": 1800.0,
}


def generate_ohlcv(symbol: str, days: int = 60) -> pd.DataFrame:
    """
    Generate `days` trading days of OHLCV data for a symbol.
    The seed is derived from the symbol name so results are reproducible.
    """
    seed = int.from_bytes(symbol.encode(), "big") % (2**31)
    rng = np.random.default_rng(seed)

    base = BASE_PRICES[symbol]

    # Build trading-day date range (skip weekends for realism)
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = []
    d = end_date - timedelta(days=days * 2)  # start far back, then filter
    while len(dates) < days:
        if d.weekday() < 5:  # Mon–Fri
            dates.append(d)
        d += timedelta(days=1)
    dates = dates[-days:]

    # Simulate close prices with a geometric random walk
    daily_returns = rng.normal(loc=0.0005, scale=0.015, size=days)
    closes = base * np.cumprod(1 + daily_returns)

    records = []
    for i, (date, close) in enumerate(zip(dates, closes)):
        intraday_range = close * rng.uniform(0.005, 0.025)
        high = close + intraday_range * rng.uniform(0.3, 1.0)
        low  = close - intraday_range * rng.uniform(0.3, 1.0)
        open_ = low + (high - low) * rng.uniform(0.2, 0.8)
        volume = int(rng.uniform(500_000, 8_000_000))
        records.append({
            "date":   date,
            "open":   round(float(open_), 2),
            "high":   round(float(high),  2),
            "low":    round(float(low),   2),
            "close":  round(float(close), 2),
            "volume": volume,
        })

    df = pd.DataFrame(records)
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    return df


# ── In-memory cache so we only generate once per process ──────────────────────
_cache: dict[str, pd.DataFrame] = {}


def get_all_stock_data() -> dict[str, pd.DataFrame]:
    global _cache
    if not _cache:
        for symbol in STOCKS:
            _cache[symbol] = generate_ohlcv(symbol, days=260)
    return _cache


def get_stock_data(symbol: str) -> pd.DataFrame | None:
    return get_all_stock_data().get(symbol)
