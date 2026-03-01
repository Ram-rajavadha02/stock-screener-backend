"""
Technical indicator calculator — zero external TA library dependencies.
All indicators are implemented directly with pandas / numpy.

Supported keys
--------------
Price    : close  open  high  low  volume
SMA      : sma_20  sma_50  sma_200
EMA      : ema_9   ema_20  ema_50
Momentum : rsi_14
Numeric  : any string that parses as float → constant series
"""
import re
import pandas as pd
import numpy as np


# ── Pure-pandas indicator functions ──────────────────────────────────────────

def _sma(df: pd.DataFrame, length: int) -> pd.Series:
    return df["close"].rolling(window=length, min_periods=length).mean()


def _ema(df: pd.DataFrame, length: int) -> pd.Series:
    # adjust=False gives the classic recursive EMA used in finance
    return df["close"].ewm(span=length, adjust=False, min_periods=length).mean()


def _vol_ema(df: pd.DataFrame, length: int) -> pd.Series:
    """EMA of daily volume — approximates Chartink's Weekly EMA(volume, 20)."""
    return df["volume"].astype(float).ewm(span=length, adjust=False, min_periods=length).mean()


def _vol_sma(df: pd.DataFrame, length: int) -> pd.Series:
    """Simple moving average of volume."""
    return df["volume"].astype(float).rolling(window=length, min_periods=length).mean()


def _rolling_high(df: pd.DataFrame, length: int) -> pd.Series:
    """
    Highest High over the previous `length` trading days (excluding today).
    shift(1) ensures today's candle is not included — used for breakout checks
    like "close > 10-day high" or "close > 26-week high".
    """
    return df["high"].shift(1).rolling(window=length, min_periods=length).max()


def _wilder_rma(series: pd.Series, length: int) -> pd.Series:
    """
    Wilder's RMA (smoothed moving average) — exact match for TradingView's ta.rma().

    Initialization: first output = SMA of first `length` bars.
    Subsequent    : y[i] = alpha * x[i] + (1 - alpha) * y[i-1]  (alpha = 1/length)

    Using ewm(adjust=False) alone gives a different first value because pandas
    seeds y[0] = x[0] rather than the SMA.  This function corrects that.
    """
    alpha  = 1.0 / length
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)

    # Skip leading NaNs (delta has NaN at index 0)
    start = 0
    while start < len(values) and np.isnan(values[start]):
        start += 1

    seed_end = start + length
    if seed_end > len(values):
        return pd.Series(result, index=series.index)

    # TradingView seeds with the simple average of the first `length` values
    result[seed_end - 1] = np.nanmean(values[start:seed_end])

    # Wilder smoothing for all subsequent bars
    for i in range(seed_end, len(values)):
        if not np.isnan(values[i]):
            result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]

    return pd.Series(result, index=series.index)


def _rsi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Wilder-smoothed RSI — TradingView / Chartink compatible.

    Key difference from naive ewm(): the first smoothed value is seeded with the
    SMA of the first `length` gain/loss bars (TradingView's ta.rma() behaviour).
    This eliminates the warm-up drift that causes crossover mismatches when only
    a few months of history are available.
    """
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = _wilder_rma(gain, length)
    avg_loss = _wilder_rma(loss, length)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _wma(series: pd.Series, length: int) -> pd.Series:
    """
    Weighted Moving Average — linearly increasing weights [1, 2, ..., length].
    WMA = sum(price[i] * i) / sum(1..length)
    """
    weights = np.arange(1, length + 1, dtype=float)

    def _apply(x: np.ndarray) -> float:
        return float(np.dot(x, weights) / weights.sum())

    return series.rolling(window=length, min_periods=length).apply(_apply, raw=True)


def _rsi_wma(df: pd.DataFrame, rsi_period: int, wma_period: int) -> pd.Series:
    """WMA of RSI — the WMA signal line used in 'RSI with WMA, EMA by Traders Point'."""
    return _wma(_rsi(df, rsi_period), wma_period)


def _rsi_ema(df: pd.DataFrame, rsi_period: int, ema_period: int) -> pd.Series:
    """EMA of RSI — the EMA signal line used in 'RSI with WMA, EMA by Traders Point'."""
    rsi_vals = _rsi(df, rsi_period)
    return rsi_vals.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()


# ── Registry ──────────────────────────────────────────────────────────────────

INDICATOR_REGISTRY: dict[str, callable] = {
    # Raw OHLCV
    "close":   lambda df: df["close"],
    "open":    lambda df: df["open"],
    "high":    lambda df: df["high"],
    "low":     lambda df: df["low"],
    "volume":  lambda df: df["volume"].astype(float),

    # Simple Moving Averages
    "sma_20":  lambda df: _sma(df, 20),
    "sma_50":  lambda df: _sma(df, 50),
    "sma_200": lambda df: _sma(df, 200),

    # Exponential Moving Averages
    "ema_5":   lambda df: _ema(df, 5),
    "ema_9":   lambda df: _ema(df, 9),
    "ema_20":  lambda df: _ema(df, 20),
    "ema_50":  lambda df: _ema(df, 50),

    # Volume Moving Averages
    "vol_ema_20": lambda df: _vol_ema(df, 20),  # daily EMA of volume
    "vol_sma_20": lambda df: _vol_sma(df, 20),  # daily SMA of volume

    # Momentum
    "rsi_14":  lambda df: _rsi(df, 14),

    # Rolling High (previous N days, excluding today — for breakout detection)
    "high_10d": lambda df: _rolling_high(df, 10),    # 10-trading-day high
    "high_26w": lambda df: _rolling_high(df, 130),   # 26-week high (130 trading days)

    # Percentage bands — useful for "20 MA within X% of 200 MA" conditions
    "sma_200_band_low":  lambda df: _sma(df, 200) * 1.005,  # 200 SMA + 0.5%
    "sma_200_band_high": lambda df: _sma(df, 200) * 1.030,  # 200 SMA + 3.0%
    "sma_20_band_high":  lambda df: _sma(df, 20)  * 1.020,  # 20 SMA  + 2.0%
}


# ── Timeframe resampling ──────────────────────────────────────────────────────

_RESAMPLE_RULES: dict[str, str | None] = {
    "daily":   None,
    "weekly":  "W-FRI",   # week ending Friday (NSE market)
    "monthly": "ME",      # month end
}


def _resample_df(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = _RESAMPLE_RULES.get(timeframe)
    if rule is None:
        return df
    return df.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(how="all")


# ── Dynamic indicator patterns (sma_N, ema_N, rsi_N, ...) ────────────────────
# Allows any period — e.g. sma_15, ema_7, rsi_21, high_10, high_10d

_DYNAMIC_PATTERNS: list[tuple] = [
    (re.compile(r"^sma_(\d+)$"),             lambda df, n: _sma(df, int(n))),
    (re.compile(r"^ema_(\d+)$"),             lambda df, n: _ema(df, int(n))),
    (re.compile(r"^rsi_(\d+)$"),             lambda df, n: _rsi(df, int(n))),
    (re.compile(r"^vol_ema_(\d+)$"),         lambda df, n: _vol_ema(df, int(n))),
    (re.compile(r"^vol_sma_(\d+)$"),         lambda df, n: _vol_sma(df, int(n))),
    # high_10 or high_10d — both resolve to rolling_high(N)
    (re.compile(r"^high_(\d+)d?$"),          lambda df, n: _rolling_high(df, int(n))),
    # RSI with WMA/EMA signal lines — 'RSI with WMA, EMA by Traders Point'
    (re.compile(r"^rsi_wma_(\d+)_(\d+)$"),  lambda df, n, m: _rsi_wma(df, int(n), int(m))),
    (re.compile(r"^rsi_ema_(\d+)_(\d+)$"),  lambda df, n, m: _rsi_ema(df, int(n), int(m))),
]


# ── Public API ────────────────────────────────────────────────────────────────

def get_series(df: pd.DataFrame, key: str) -> pd.Series:
    """
    Resolve `key` to a pd.Series aligned to df.index.

    Priority:
      1. Static registry  → pre-computed named indicator
      2. Dynamic pattern  → sma_N / ema_N / rsi_N / vol_ema_N / vol_sma_N / high_N[d]
      3. Parseable float  → constant series
      4. Unknown          → ValueError
    """
    # 1. Static registry
    if key in INDICATOR_REGISTRY:
        series = INDICATOR_REGISTRY[key](df)
        series.name = key
        return series

    # 2. Dynamic patterns — any user-supplied period(s)
    for pattern, fn in _DYNAMIC_PATTERNS:
        m = pattern.match(key)
        if m:
            series = fn(df, *m.groups())  # unpack all capture groups (1 or 2)
            series.name = key
            return series

    # 3. Numeric constant
    try:
        val = float(key)
        return pd.Series([val] * len(df), index=df.index, name=key, dtype=float)
    except ValueError:
        pass

    raise ValueError(
        f"Unknown indicator '{key}'. "
        f"Valid keys: {list(INDICATOR_REGISTRY.keys())} — or pass any numeric string."
    )


def get_series_tf(df: pd.DataFrame, key: str, timeframe: str = "daily") -> pd.Series:
    """
    Compute indicator `key` on the given timeframe (daily / weekly / monthly),
    then forward-fill the result back to the original daily index.

    Examples
    --------
    get_series_tf(df, "close",  "weekly")   → weekly close, ffilled to daily
    get_series_tf(df, "sma_20", "weekly")   → SMA(20) on weekly bars, ffilled
    get_series_tf(df, "70",     "daily")    → constant 70 (numeric passthrough)
    """
    resampled = _resample_df(df, timeframe)
    series = get_series(resampled, key)
    if timeframe == "daily":
        return series
    # Reindex weekly/monthly series to every daily bar using forward-fill
    return series.reindex(df.index, method="ffill")
