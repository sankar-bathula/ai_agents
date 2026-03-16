from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class CandlestickPattern:
    """
    Simple representation of a detected candlestick pattern.
    """

    date: Any
    pattern: str
    direction: Optional[str]


def _ensure_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have standard OHLC columns, trying to be tolerant of
    differing capitalisation or presence of only an adjusted close.
    """
    if df is None or df.empty:
        raise ValueError("Price DataFrame is empty; cannot analyze candlestick patterns.")

    data = df.copy()

    # yfinance can sometimes return MultiIndex columns like ('TICKER', 'Open').
    # In that case, collapse to the last element (e.g. 'Open', 'High', ...).
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(levels[-1]) for levels in data.columns]

    # First, try a case-insensitive normalisation of common OHLC names.
    normalised: dict[str, str] = {}
    for col in data.columns:
        key = str(col).strip().lower()
        if key == "open":
            normalised[col] = "Open"
        elif key == "high":
            normalised[col] = "High"
        elif key == "low":
            normalised[col] = "Low"
        elif key in ("close", "closing"):
            normalised[col] = "Close"
        elif key in ("adj close", "adj_close", "adjusted close"):
            # Fall back to adjusted close as Close if nothing else is available.
            normalised[col] = "Adj Close"

    if normalised:
        data = data.rename(columns=normalised)

    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(data.columns)

    # If we are missing some fields but have an adjusted close, synthesise OHLC
    # from it as a flat bar so that the agent can still run.
    if missing and "Adj Close" in data.columns:
        adj = data["Adj Close"]
        for name in required:
            if name not in data.columns:
                data[name] = adj
        missing = required.difference(data.columns)

    if missing:
        raise ValueError(f"Expected OHLC columns {sorted(required)}, missing: {sorted(missing)}")

    return data


def _body(df: pd.DataFrame) -> pd.Series:
    return (df["Close"] - df["Open"]).abs()


def _range(df: pd.DataFrame) -> pd.Series:
    return df["High"] - df["Low"]


def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["High"] - df[["Open", "Close"]].max(axis=1)


def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    return df[["Open", "Close"]].min(axis=1) - df["Low"]


def _is_bull(df: pd.DataFrame) -> pd.Series:
    return df["Close"] > df["Open"]


def _is_bear(df: pd.DataFrame) -> pd.Series:
    return df["Close"] < df["Open"]


def detect_basic_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect a handful of common single / double candle patterns:

    - Hammer / Inverted hammer
    - Shooting star
    - Doji
    - Bullish / Bearish engulfing

    Returns a copy of the input DataFrame with an extra 'pattern' column
    describing the strongest detected pattern per row (or empty string if none).
    """
    data = _ensure_ohlc_columns(df)

    body = _body(data)
    rng = _range(data)
    upper = _upper_shadow(data)
    lower = _lower_shadow(data)

    # Avoid division by zero on very flat bars.
    eps = 1e-9
    body_ratio = body / (rng.replace(0, pd.NA).fillna(eps))
    upper_ratio = upper / (rng.replace(0, pd.NA).fillna(eps))
    lower_ratio = lower / (rng.replace(0, pd.NA).fillna(eps))

    is_small_body = body_ratio < 0.25
    is_very_small_body = body_ratio < 0.1

    # Doji: very small body relative to range.
    doji = is_very_small_body

    # Hammer: small body near top of range with long lower shadow.
    hammer = (
        is_small_body
        & (lower_ratio > 0.6)
        & (upper_ratio < 0.2)
    )

    # Inverted hammer / shooting star: small body near bottom with long upper shadow.
    inverted_hammer = (
        is_small_body
        & (upper_ratio > 0.6)
        & (lower_ratio < 0.2)
        & _is_bull(data)
    )
    shooting_star = (
        is_small_body
        & (upper_ratio > 0.6)
        & (lower_ratio < 0.2)
        & _is_bear(data)
    )

    # Engulfing patterns need previous candle.
    prev_open = data["Open"].shift(1)
    prev_close = data["Close"].shift(1)

    prev_bear = prev_close < prev_open
    prev_bull = prev_close > prev_open

    bull_engulf = (
        prev_bear
        & _is_bull(data)
        & (data["Open"] <= prev_close)
        & (data["Close"] >= prev_open)
    )
    bear_engulf = (
        prev_bull
        & _is_bear(data)
        & (data["Open"] >= prev_close)
        & (data["Close"] <= prev_open)
    )

    # Assign a single label per row based on a simple priority order.
    pattern = pd.Series("", index=data.index, dtype="string")

    pattern = pattern.mask(bull_engulf, "Bullish Engulfing")
    pattern = pattern.mask(bear_engulf, "Bearish Engulfing")
    pattern = pattern.mask(hammer, "Hammer")
    pattern = pattern.mask(inverted_hammer, "Inverted Hammer")
    pattern = pattern.mask(shooting_star, "Shooting Star")
    pattern = pattern.mask(doji, "Doji")

    data = data.copy()
    data["pattern"] = pattern
    return data


def analyze_candlestick_patterns(
    df: pd.DataFrame,
    *,
    lookback: int = 80,
) -> Dict[str, Any]:
    """
    High-level API used by the Streamlit app:

    - Enrich the OHLC DataFrame with a 'pattern' column.
    - Keep only the last `lookback` rows where a pattern is present.
    - Return a small summary + table for display.
    """
    detected = detect_basic_patterns(df)

    # Only keep rows where a non-empty pattern was found.
    mask = detected["pattern"].astype(str).str.len() > 0
    subset = detected.loc[mask].tail(lookback)

    if subset.empty:
        return {
            "last_pattern": None,
            "last_pattern_date": None,
            "recent_patterns": [],
            "table": pd.DataFrame(
                columns=["date", "Open", "High", "Low", "Close", "pattern"]
            ),
        }

    # Ensure we have a materialized date column for display purposes.
    subset = subset.copy()
    subset["date"] = subset.index

    last_row = subset.iloc[-1]
    last_pattern = str(last_row["pattern"])
    last_date = last_row["date"]

    recent: List[Dict[str, Any]] = [
        {
            "date": row["date"],
            "pattern": row["pattern"],
            "Open": row["Open"],
            "High": row["High"],
            "Low": row["Low"],
            "Close": row["Close"],
        }
        for _, row in subset.iterrows()
    ]

    table = subset[["date", "Open", "High", "Low", "Close", "pattern"]]

    return {
        "last_pattern": last_pattern,
        "last_pattern_date": last_date,
        "recent_patterns": recent,
        "table": table,
    }

