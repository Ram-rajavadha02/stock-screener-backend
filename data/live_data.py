"""
Live OHLCV data via yfinance (Yahoo Finance).

Stock universe: ALL active NSE EQ-series equities (~2200 stocks), fetched
dynamically from the NSE archives CSV on first use (cached daily).

OHLCV history is downloaded in batches using yf.download() and cached
in-process for CACHE_TTL_SECONDS so repeated scans are instant.
"""
import time
import requests
import pandas as pd
import yfinance as yf
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait

# ── NSE equity list ───────────────────────────────────────────────────────────

_NSE_EQUITY_CSV = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
_NSE_CSV_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Fallback list used if the NSE CSV is unreachable
_FALLBACK_SYMBOLS: list[tuple[str, str]] = [
    ("RELIANCE",   "RELIANCE.NS"), ("TCS",        "TCS.NS"),
    ("INFY",       "INFY.NS"),     ("HDFCBANK",   "HDFCBANK.NS"),
    ("ICICIBANK",  "ICICIBANK.NS"),("SBIN",       "SBIN.NS"),
    ("WIPRO",      "WIPRO.NS"),    ("BAJFINANCE", "BAJFINANCE.NS"),
    ("AXISBANK",   "AXISBANK.NS"), ("KOTAKBANK",  "KOTAKBANK.NS"),
    ("HCLTECH",    "HCLTECH.NS"),  ("LT",         "LT.NS"),
    ("MARUTI",     "MARUTI.NS"),   ("ASIANPAINT", "ASIANPAINT.NS"),
    ("SUNPHARMA",  "SUNPHARMA.NS"),("TITAN",      "TITAN.NS"),
    ("BHARTIARTL", "BHARTIARTL.NS"),("TATAMOTORS","TATAMOTORS.NS"),
    ("HINDUNILVR", "HINDUNILVR.NS"),("ITC",       "ITC.NS"),
    ("MPHASIS",    "MPHASIS.NS"),  ("ROUTE",      "ROUTE.NS"),
    ("MCL",        "MCL.NS"),      ("SBGLP",      "SBGLP.NS"),
    ("LYKALABS",   "LYKALABS.NS"), ("NTPC",       "NTPC.NS"),
    ("POWERGRID",  "POWERGRID.NS"),("ONGC",       "ONGC.NS"),
    ("TATASTEEL",  "TATASTEEL.NS"),("JSWSTEEL",   "JSWSTEEL.NS"),
    ("DRREDDY",    "DRREDDY.NS"),  ("CIPLA",      "CIPLA.NS"),
    ("TECHM",      "TECHM.NS"),    ("PERSISTENT", "PERSISTENT.NS"),
    ("ZOMATO",     "ZOMATO.NS"),   ("POLYCAB",    "POLYCAB.NS"),
    ("TATAPOWER",  "TATAPOWER.NS"),("PFC",        "PFC.NS"),
    ("RECLTD",     "RECLTD.NS"),   ("IRFC",       "IRFC.NS"),
]

# ── Symbol-list cache (refreshed daily) ──────────────────────────────────────

_symbols: list[tuple[str, str]] = []
_symbols_ts: float = 0.0
_SYMBOL_TTL = 86_400  # 24 hours


def _get_nse_symbols() -> list[tuple[str, str]]:
    """Return all active NSE EQ-series stocks as (display_name, yahoo_ticker) pairs."""
    global _symbols, _symbols_ts

    if _symbols and (time.time() - _symbols_ts) < _SYMBOL_TTL:
        return _symbols

    try:
        resp = requests.get(_NSE_EQUITY_CSV, headers=_NSE_CSV_HEADERS, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.strip() for c in df.columns]
        eq = df[df["SERIES"].str.strip() == "EQ"]["SYMBOL"].dropna()
        result = [(s.strip(), f"{s.strip()}.NS") for s in eq.tolist()]
        if result:
            _symbols = result
            _symbols_ts = time.time()
            print(f"[live_data] NSE equity list: {len(result)} EQ stocks loaded.")
            return result
    except Exception as exc:
        print(f"[live_data] Could not fetch NSE symbol list: {exc}")

    print("[live_data] Using fallback symbol list.")
    return _FALLBACK_SYMBOLS


# ── OHLCV batch download ──────────────────────────────────────────────────────

HISTORY_PERIOD   = "1y"   # 1 year ≈ 252 bars — needed for RSI Wilder warmup
CACHE_TTL_SECONDS = 3600  # 1 hour
BATCH_SIZE       = 200    # tickers per yf.download() call
MAX_WORKERS      = 10     # parallel batch downloads
DOWNLOAD_TIMEOUT = 180    # seconds — return whatever finished within this time

_cache:    dict[str, pd.DataFrame] = {}
_cache_ts: float = 0.0


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    needed = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if not needed or "close" not in needed:
        return pd.DataFrame()
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    return df.dropna(how="all")


def _fetch_batch(batch: list[tuple[str, str]]) -> dict[str, pd.DataFrame]:
    """Download a batch of tickers in one yf.download() call."""
    if not batch:
        return {}
    names_map = {ticker: name for name, ticker in batch}
    tickers = [t for _, t in batch]
    try:
        raw = yf.download(
            tickers,
            period=HISTORY_PERIOD,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw.empty:
            return {}

        result = {}
        if len(tickers) == 1:
            # Single ticker: flat DataFrame
            df = _normalise(raw)
            if not df.empty:
                result[names_map[tickers[0]]] = df
        else:
            # Multiple tickers: MultiIndex columns (ticker, price_type)
            for ticker in tickers:
                try:
                    sub = raw[ticker].copy()
                    if sub.empty or sub.isna().all().all():
                        continue
                    df = _normalise(sub)
                    if not df.empty:
                        result[names_map[ticker]] = df
                except (KeyError, Exception):
                    pass
        return result
    except Exception as exc:
        print(f"[live_data] Batch download error: {exc}")
        return {}


def get_all_live_data(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    global _cache, _cache_ts

    age = time.time() - _cache_ts
    if _cache and not force_refresh and age < CACHE_TTL_SECONDS:
        return _cache

    symbols = _get_nse_symbols()
    total   = len(symbols)
    batches = [symbols[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    print(f"[live_data] Downloading {total} NSE stocks in {len(batches)} batches "
          f"({MAX_WORKERS} parallel, period={HISTORY_PERIOD}) …")
    t0 = time.time()

    fresh: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_fetch_batch, b) for b in batches]
        done, pending = futures_wait(futures, timeout=DOWNLOAD_TIMEOUT)

        for f in done:
            try:
                fresh.update(f.result())
            except Exception:
                pass

        if pending:
            print(f"[live_data] Timeout after {DOWNLOAD_TIMEOUT}s — "
                  f"{len(pending)} batch(es) cancelled, "
                  f"{len(fresh)} stocks collected so far.")
            for f in pending:
                f.cancel()

    elapsed = time.time() - t0
    if fresh:
        _cache    = fresh
        _cache_ts = time.time()
        print(f"[live_data] Ready: {len(fresh)}/{total} stocks cached in {elapsed:.1f}s.")
    else:
        print("[live_data] No data fetched — keeping previous cache.")

    return _cache
