"""
polygon_client.py
-----------------
Thin Polygon.io REST client for the Agentic AlgoTrading System dashboard.

Endpoints used (all REST v2/v3, work with a standard API key):
  - /v2/aggs/ticker/{ticker}/range/{mult}/{timespan}/{from}/{to}   -> OHLCV bars
  - /v3/reference/tickers/{ticker}                                  -> ticker details (sector/SIC)
  - /v2/snapshot/locale/us/markets/stocks/tickers/{ticker}          -> latest snapshot
  - /v2/reference/news                                              -> recent news headlines

Design notes (per the Financial Behavior doc):
  - Polygon data is the SUBSTRATE, not an alpha source. This module only
    fetches and normalizes; all signal logic lives in features.py.
  - Free-tier keys are limited to ~5 requests/min. We retry on 429 with
    exponential backoff and cache aggressively via st.cache_data.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import requests

BASE_URL = "https://api.polygon.io"


class PolygonError(RuntimeError):
    pass


def _request(path: str, api_key: str, params: dict | None = None,
             max_retries: int = 4) -> dict:
    """GET with 429 backoff. Raises PolygonError on hard failures."""
    params = dict(params or {})
    params["apiKey"] = api_key
    url = f"{BASE_URL}{path}"

    for attempt in range(max_retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            # Free tier rate limit — back off (5 req/min => ~12s spacing)
            wait = 15 * (attempt + 1)
            time.sleep(wait)
            continue
        if resp.status_code in (401, 403):
            raise PolygonError(
                f"Auth error ({resp.status_code}). Check your Polygon API key "
                f"and plan entitlements. Body: {resp.text[:200]}"
            )
        raise PolygonError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    raise PolygonError("Rate-limited repeatedly (429). Free tier allows ~5 req/min — "
                       "reduce the universe size or wait a minute.")


# ----------------------------------------------------------------------
# Aggregates (OHLCV bars) — the system's three horizons: 4H / 1D / 1W
# ----------------------------------------------------------------------

HORIZONS = {
    "4H": {"multiplier": 4, "timespan": "hour",  "lookback_days": 120},
    "1D": {"multiplier": 1, "timespan": "day",   "lookback_days": 400},
    "1W": {"multiplier": 1, "timespan": "week",  "lookback_days": 365 * 3},
}


def get_aggregates(ticker: str, horizon: str, api_key: str,
                   end: date | None = None) -> pd.DataFrame:
    """
    Fetch OHLCV bars for one ticker at one of the system's horizons.

    Returns DataFrame indexed by timestamp with columns:
      open, high, low, close, volume, vwap, transactions
    """
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {list(HORIZONS)}")

    cfg = HORIZONS[horizon]
    end = end or date.today()
    start = end - timedelta(days=cfg["lookback_days"])

    path = (f"/v2/aggs/ticker/{ticker.upper()}/range/"
            f"{cfg['multiplier']}/{cfg['timespan']}/{start}/{end}")
    data = _request(path, api_key, params={"adjusted": "true",
                                           "sort": "asc", "limit": 50000})

    results = data.get("results") or []
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).rename(columns={
        "o": "open", "h": "high", "l": "low", "c": "close",
        "v": "volume", "vw": "vwap", "n": "transactions", "t": "ts",
    })
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    df = df.set_index("ts").sort_index()

    if horizon == "4H":
        # Doc warning: the 4H sleeve is "most contaminated by microstructure
        # noise" — drop bars from illiquid pre/post-market windows.
        df = df[(df.index.hour >= 8) & (df.index.hour <= 20)]

    return df[["open", "high", "low", "close", "volume", "vwap", "transactions"]]


# ----------------------------------------------------------------------
# Reference / snapshot / news
# ----------------------------------------------------------------------

def get_ticker_details(ticker: str, api_key: str) -> dict:
    """Ticker metadata: name, market cap, SIC description (used as sector proxy)."""
    data = _request(f"/v3/reference/tickers/{ticker.upper()}", api_key)
    r = data.get("results", {}) or {}
    return {
        "ticker": ticker.upper(),
        "name": r.get("name"),
        "market_cap": r.get("market_cap"),
        "sector": r.get("sic_description") or "Unknown",
        "primary_exchange": r.get("primary_exchange"),
    }


def get_snapshot(ticker: str, api_key: str) -> dict:
    """Latest snapshot: last price, day change, prev close."""
    data = _request(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}",
                    api_key)
    t = data.get("ticker", {}) or {}
    day = t.get("day", {}) or {}
    prev = t.get("prevDay", {}) or {}
    return {
        "ticker": ticker.upper(),
        "last": day.get("c") or prev.get("c"),
        "day_change_pct": t.get("todaysChangePerc"),
        "prev_close": prev.get("c"),
        "volume": day.get("v"),
    }


def get_news(ticker: str, api_key: str, limit: int = 8) -> pd.DataFrame:
    """
    Recent headlines. Per the doc: news at 4H+ horizons is an INTERPRETATION
    edge, not a speed edge — we surface headlines for context only and make
    no claim of standalone predictive power.
    """
    data = _request("/v2/reference/news", api_key,
                    params={"ticker": ticker.upper(), "limit": limit,
                            "order": "desc", "sort": "published_utc"})
    results = data.get("results") or []
    if not results:
        return pd.DataFrame()
    rows = [{
        "published": r.get("published_utc", "")[:16].replace("T", " "),
        "title": r.get("title"),
        "publisher": (r.get("publisher") or {}).get("name"),
        "url": r.get("article_url"),
    } for r in results]
    return pd.DataFrame(rows)
