from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd
import yfinance as yf


def get_price_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV price history for a single ticker.

    Parameters
    ----------
    ticker:
        Symbol, e.g. "RELIANCE.NS" for NSE or "AAPL" for US.
    period:
        yfinance period string (e.g. "6mo", "1y"). Ignored if explicit dates are used.
    interval:
        Bar interval, e.g. "1d".

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by datetime with columns like
        ["Open", "High", "Low", "Close", "Adj Close", "Volume"].
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'.")
    return df


def get_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Fetch a lightweight set of fundamental fields for a single ticker.

    This is intentionally minimal and best-effort, since availability of
    fields varies across instruments and data providers.
    """
    t = yf.Ticker(ticker)

    info: Dict[str, Any] = {}
    # yfinance has been evolving its API; try both .get_info() and .info.
    try:
        info = t.get_info()
    except Exception:
        try:
            # type: ignore[attr-defined]
            info = t.info  # pragma: no cover - fallback path
        except Exception:
            info = {}

    price = None
    try:
        price = float(t.fast_info["last_price"])
    except Exception:
        price = info.get("currentPrice")

    return {
        "price": price,
        # Common fundamental-style fields; may be None for some tickers.
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "returnOnEquity": info.get("returnOnEquity"),
        "debtToEquity": info.get("debtToEquity"),
        "marketCap": info.get("marketCap"),
        "longName": info.get("longName") or info.get("shortName"),
        "currency": info.get("currency"),
    }


def default_date_range(days: int = 180) -> tuple[datetime, datetime]:
    """
    Utility: compute a (start, end) date range for convenience.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    return start, end

