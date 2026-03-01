"""
Live OHLCV data via yfinance (Yahoo Finance).

Stock universe: ALL active NSE EQ-series equities (~2200 stocks), fetched
dynamically from the NSE archives CSV on first use (cached daily).

OHLCV history is downloaded in parallel using a thread pool and cached
in-process for CACHE_TTL_SECONDS so repeated scans are instant.

Timings (approx, depends on network):
  Symbol list refresh  :  ~1 s   (once per day)
  First OHLCV fetch    :  ~45 s  (all ~2200 stocks, 30 parallel workers)
  Subsequent scans     :  instant (served from in-memory cache)
"""
import time
import requests
import pandas as pd
import yfinance as yf
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        # Normalise column names (NSE CSV has leading/trailing spaces)
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


# ── OHLCV download ────────────────────────────────────────────────────────────

HISTORY_PERIOD    = "1y"    # 1 year ≈ 252 bars — needed for RSI Wilder warmup + SMA-200
CACHE_TTL_SECONDS = 3600    # 1 hour — daily data doesn't change intraday
MAX_WORKERS       = 30      # parallel yfinance connections

_cache:    dict[str, pd.DataFrame] = {}
_cache_ts: float = 0.0


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    return df.dropna(how="all")


def _fetch_one(name: str, ticker: str) -> tuple[str, pd.DataFrame | None]:
    """Download one stock — executed in a thread-pool worker."""
    try:
        raw = yf.Ticker(ticker).history(period=HISTORY_PERIOD)
        if raw.empty:
            return name, None
        return name, _normalise(raw)
    except Exception:
        return name, None


def get_all_live_data(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    global _cache, _cache_ts

    age = time.time() - _cache_ts
    if _cache and not force_refresh and age < CACHE_TTL_SECONDS:
        return _cache

    symbols = _get_nse_symbols()
    total   = len(symbols)
    print(f"[live_data] Downloading {total} NSE stocks "
          f"({MAX_WORKERS} parallel workers, period={HISTORY_PERIOD}) …")
    t0 = time.time()

    fresh: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, n, t): n for n, t in symbols}
        done = 0
        for future in as_completed(futures):
            name, df = future.result()
            done += 1
            if df is not None:
                fresh[name] = df
            if done % 200 == 0:
                print(f"  … {done}/{total} done")

    elapsed = time.time() - t0
    if fresh:
        _cache    = fresh
        _cache_ts = time.time()
        print(f"[live_data] Ready: {len(fresh)}/{total} stocks cached in {elapsed:.1f}s.")
    else:
        print("[live_data] No data fetched — keeping previous cache.")

    return _cache
