from __future__ import annotations

from typing import Iterable, List

import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute a simple RSI (Relative Strength Index) on a price series.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.rename("RSI")


def moving_averages(
    series: pd.Series,
    windows: Iterable[int] = (20, 50),
) -> pd.DataFrame:
    """
    Compute simple moving averages for a collection of window lengths.
    """
    ma_frames: List[pd.Series] = []
    for w in windows:
        ma = series.rolling(window=w, min_periods=w).mean().rename(f"MA_{w}")
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

