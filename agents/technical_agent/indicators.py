from __future__ import annotations

from typing import Iterable, List

import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute a simple RSI (Relative Strength Index) on a price series.
    """
    # Guard against accidentally receiving a DataFrame (e.g. from double-bracket
    # column selection or unexpected upstream behavior). We always want a 1D Series
    # here so that the subsequent `.rename("RSI")` is applied to a Series, not a
    # DataFrame, which would cause pandas to treat the string as a callable.
    if isinstance(series, pd.DataFrame):
        # If upstream has duplicate column names (e.g. multiple "Close" columns),
        # `df["Close"]` returns a DataFrame. Take the first column to proceed.
        series = series.iloc[:, 0]

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))

    # Ensure we are returning a Series with a stable name.
    if isinstance(rsi_val, pd.DataFrame):
        if rsi_val.shape[1] != 1:
            raise ValueError("Internal RSI computation produced multi-column result.")
        rsi_val = rsi_val.iloc[:, 0]

    return rsi_val.rename("RSI")


def moving_averages(
    series: pd.Series,
    windows: Iterable[int] = (20, 50),
) -> pd.DataFrame:
    """
    Compute simple moving averages for a collection of window lengths.
    """
    # As with `rsi`, ensure we are working with a 1D Series, not a DataFrame,
    # so that `.rename(f"MA_{w}")` is applied to a Series and not a DataFrame.
    if isinstance(series, pd.DataFrame):
        # Same rationale as in `rsi`: tolerate duplicate column names by taking
        # the first column.
        series = series.iloc[:, 0]

    ma_frames: List[pd.Series] = []
    for w in windows:
        ma_raw = series.rolling(window=w, min_periods=w).mean()

        if isinstance(ma_raw, pd.DataFrame):
            if ma_raw.shape[1] != 1:
                raise ValueError(
                    "Internal moving_averages computation produced multi-column result."
                )
            ma_raw = ma_raw.iloc[:, 0]

        ma = ma_raw.rename(f"MA_{w}")
        ma_frames.append(ma)
    if not ma_frames:
        return pd.DataFrame(index=series.index)
    return pd.concat(ma_frames, axis=1)


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the OHLCV DataFrame enriched with a few core indicators:

    - RSI(14) on the Close column
    - Simple moving averages (20, 50) on the Close column
    """
    if "Close" not in df.columns:
        raise ValueError("Expected a 'Close' column in price DataFrame.")

    enriched = df.copy()
    enriched["RSI_14"] = rsi(enriched["Close"], period=14)
    ma_df = moving_averages(enriched["Close"], windows=(20, 50))
    for col in ma_df.columns:
        enriched[col] = ma_df[col]
    return enriched

