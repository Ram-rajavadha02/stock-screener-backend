"""
Live OHLCV data via yfinance (Yahoo Finance).

Stock universe: ALL active NSE EQ-series equities (~2200 stocks), fetched
dynamically from the NSE archives CSV on first use (cached daily).

OHLCV history is downloaded in parallel batches and cached in-process for
CACHE_TTL_SECONDS so repeated scans are instant.

First load (cold start) may take 3–8 minutes on the free tier — this is
normal. Call preload_background() at server startup to begin warming the
cache immediately so scans are fast when the user first connects.
"""
import time
import threading
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


# ── OHLCV download ────────────────────────────────────────────────────────────

HISTORY_PERIOD    = "1y"   # 1 year ≈ 252 bars for RSI Wilder warmup
CACHE_TTL_SECONDS = 3600   # 1-hour cache — re-download once per hour
BATCH_SIZE        = 100    # tickers per yf.download() call
MAX_WORKERS       = 15     # parallel batch downloads

_cache:     dict[str, pd.DataFrame] = {}
_cache_ts:  float = 0.0
_load_lock  = threading.Lock()   # only one download runs at a time


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "close" not in df.columns:
        return pd.DataFrame()
    df = df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    return df.dropna(how="all")


def _fetch_batch(batch: list[tuple[str, str]]) -> dict[str, pd.DataFrame]:
    """Download a batch of tickers in one yf.download() call."""
    if not batch:
        return {}
    names_map = {ticker: name for name, ticker in batch}
    tickers   = [t for _, t in batch]
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
            df = _normalise(raw)
            if not df.empty:
                result[names_map[tickers[0]]] = df
        else:
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
        print(f"[live_data] Batch error: {exc}")
        return {}


def _run_full_download() -> dict[str, pd.DataFrame]:
    """Download all NSE stocks. Blocks until every batch is done — no timeout."""
    symbols = _get_nse_symbols()
    total   = len(symbols)
    batches = [symbols[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    print(f"[live_data] Downloading {total} stocks in {len(batches)} batches "
          f"({MAX_WORKERS} workers) …")
    t0 = time.time()

    fresh: dict[str, pd.DataFrame] = {}
    done_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_batch, b): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            done_count += 1
            try:
                fresh.update(future.result())
            except Exception:
                pass
            if done_count % 5 == 0 or done_count == len(batches):
                print(f"  … {done_count}/{len(batches)} batches done, "
                      f"{len(fresh)} stocks so far ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    print(f"[live_data] Download complete: {len(fresh)}/{total} stocks in {elapsed:.1f}s.")
    return fresh


def get_all_live_data(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """
    Return OHLCV data for all NSE stocks.

    Thread-safe: if two requests arrive simultaneously while the cache is
    cold, only ONE download runs; the second waits for the lock and then
    returns the freshly-populated cache.
    """
    global _cache, _cache_ts

    # Fast path — cache is warm
    if _cache and not force_refresh and (time.time() - _cache_ts) < CACHE_TTL_SECONDS:
        return _cache

    # Slow path — acquire lock so only one thread downloads at a time
    with _load_lock:
        # Re-check inside the lock (another thread may have just finished)
        if _cache and not force_refresh and (time.time() - _cache_ts) < CACHE_TTL_SECONDS:
            return _cache

        fresh = _run_full_download()
        if fresh:
            _cache    = fresh
            _cache_ts = time.time()
        else:
            print("[live_data] No data fetched — keeping previous cache.")

    return _cache


def preload_background() -> None:
    """
    Kick off a background thread to warm the cache at server startup.
    Returns immediately; download continues in the background.
    """
    def _load():
        try:
            get_all_live_data()
        except Exception as exc:
            print(f"[live_data] Background preload failed: {exc}")

    age = time.time() - _cache_ts
    if _cache and age < CACHE_TTL_SECONDS:
        print("[live_data] Cache already warm — skipping preload.")
        return

    t = threading.Thread(target=_load, daemon=True, name="live-data-preload")
    t.start()
    print("[live_data] Background preload started.")
