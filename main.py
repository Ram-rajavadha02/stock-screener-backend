"""
Stock Screener — FastAPI entry point.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Data modes
----------
Every endpoint accepts an optional query parameter:
    ?live=true   →  fetch real NSE data from Yahoo Finance (yfinance)
    ?live=false  →  use deterministic mock data (default, works offline)

Example:
    POST /api/screen?live=true
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from data.mock_data import get_all_stock_data, STOCKS
from data.live_data import get_all_live_data
from models.schemas import ScreenRequest, ScreenResponse
from screener.engine import ScreenerEngine
from indicators.calculator import get_series

app = FastAPI(
    title="Stock Screener API",
    description="Chartink-style screening engine — supports mock & live NSE data",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = ScreenerEngine()


def _get_data(live: bool) -> dict:
    return get_all_live_data() if live else get_all_stock_data()


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Stock Screener API is running 🚀"}


# ── Stocks ────────────────────────────────────────────────────────────────────

@app.get("/api/stocks", tags=["Data"])
def list_stocks(live: bool = Query(default=False, description="Use live NSE data")):
    """Return available stock symbols."""
    data = _get_data(live)
    symbols = list(data.keys())
    return {"stocks": symbols, "total": len(symbols), "source": "live" if live else "mock"}


@app.get("/api/stocks/{symbol}", tags=["Data"])
def get_stock(
    symbol: str,
    live: bool = Query(default=False),
):
    """Return OHLCV history for a single symbol (last 30 rows)."""
    all_data = _get_data(live)
    symbol = symbol.upper()
    if symbol not in all_data:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found.")
    df = all_data[symbol].tail(30).reset_index()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return {"symbol": symbol, "source": "live" if live else "mock", "data": df.to_dict(orient="records")}


# ── Screener ──────────────────────────────────────────────────────────────────

@app.post("/api/screen", response_model=ScreenResponse, tags=["Screener"])
def screen_stocks(
    request: ScreenRequest,
    live: bool = Query(default=False, description="Use live NSE data from Yahoo Finance"),
):
    """
    Run a scan against mock data (default) or live NSE data.

    **Mock** (offline, instant):
    ```
    POST /api/screen
    ```

    **Live NSE data** (requires internet, ~2–5 s first call):
    ```
    POST /api/screen?live=true
    ```

    Example payload:
    ```json
    {
      "conditions": [
        {"indicator": "close",  "operator": ">",  "value": "sma_20"},
        {"indicator": "rsi_14", "operator": "<",  "value": "70"},
        {"indicator": "volume", "operator": ">",  "value": "1000000"}
      ],
      "match": "all"
    }
    ```
    """
    try:
        all_data = _get_data(live)
        return _engine.run(request, all_data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ── Debug / Inspect ───────────────────────────────────────────────────────────

@app.get("/api/snapshot", tags=["Debug"])
def snapshot(live: bool = Query(default=False)):
    """
    Returns the latest Close, EMA5, EMA20, SMA20, RSI14, Volume, VolEMA20
    for every stock so you can cross-check values against Chartink manually.

    GET /api/snapshot?live=true
    """
    all_data = _get_data(live)
    result = []
    for symbol, df in all_data.items():
        try:
            last_date = df.index[-1].strftime("%Y-%m-%d")
            close      = round(float(get_series(df, "close").iloc[-1]),    2)
            ema_5      = round(float(get_series(df, "ema_5").dropna().iloc[-1]),    2)
            ema_20     = round(float(get_series(df, "ema_20").dropna().iloc[-1]),   2)
            sma_20     = round(float(get_series(df, "sma_20").dropna().iloc[-1]),   2)
            rsi        = round(float(get_series(df, "rsi_14").dropna().iloc[-1]),   2)
            volume     = int(get_series(df, "volume").iloc[-1])
            vol_ema_20 = round(float(get_series(df, "vol_ema_20").dropna().iloc[-1]), 0)
            result.append({
                "symbol":      symbol,
                "date":        last_date,
                "close":       close,
                "ema_5":       ema_5,
                "close>=ema5": close >= ema_5,
                "ema_20":      ema_20,
                "sma_20":      sma_20,
                "rsi_14":      rsi,
                "volume":      volume,
                "vol_ema_20":  vol_ema_20,
                "vol>=vol_ema20": volume >= vol_ema_20,
            })
        except Exception as e:
            result.append({"symbol": symbol, "error": str(e)})
    return {"source": "live" if live else "mock", "stocks": result}


@app.get("/api/snapshot/breakout", tags=["Debug"])
def snapshot_breakout(live: bool = Query(default=False)):
    """
    Shows pass/fail for each of the 8 breakout conditions per stock.
    Use this to diagnose why no stocks matched the custom scan.

    GET /api/snapshot/breakout?live=true
    """
    all_data = _get_data(live)
    result = []
    for symbol, df in all_data.items():
        try:
            def last(key):
                return float(get_series(df, key).dropna().iloc[-1])

            close           = round(last("close"), 2)
            sma_20          = round(last("sma_20"), 2)
            sma_200         = round(last("sma_200"), 2)
            high_10d        = round(last("high_10d"), 2)
            high_26w        = round(last("high_26w"), 2)
            sma_200_low     = round(last("sma_200_band_low"), 2)
            sma_200_high    = round(last("sma_200_band_high"), 2)
            sma_20_high     = round(last("sma_20_band_high"), 2)

            c1 = close > sma_200
            c2 = close > high_10d
            c3 = sma_20 >= sma_200_low
            c4 = sma_20 <= sma_200_high
            c5 = close >= sma_20
            c6 = close <= sma_20_high
            c7 = close > 80
            c8 = close > high_26w

            result.append({
                "symbol":   symbol,
                "date":     df.index[-1].strftime("%Y-%m-%d"),
                "close":    close,
                "sma_20":   sma_20,
                "sma_200":  sma_200,
                "high_10d": high_10d,
                "high_26w": high_26w,
                "sma_200+0.5%": sma_200_low,
                "sma_200+3%":   sma_200_high,
                "sma_20+2%":    sma_20_high,
                "C1_close>sma200":        c1,
                "C2_close>high10d":       c2,
                "C3_sma20>=sma200+0.5%":  c3,
                "C4_sma20<=sma200+3%":    c4,
                "C5_close>=sma20":        c5,
                "C6_close<=sma20+2%":     c6,
                "C7_close>80":            c7,
                "C8_close>high26w":       c8,
                "all_pass": all([c1, c2, c3, c4, c5, c6, c7, c8]),
                "pass_count": sum([c1, c2, c3, c4, c5, c6, c7, c8]),
            })
        except Exception as e:
            result.append({"symbol": symbol, "error": str(e)})

    result.sort(key=lambda x: x.get("pass_count", 0), reverse=True)
    return {"source": "live" if live else "mock", "stocks": result}
