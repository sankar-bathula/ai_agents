from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class SRLevel:
    price: float
    kind: str  # "support" or "resistance"
    touches: int
    first_touched: Any
    last_touched: Any


def _ensure_close(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Expected a 'Close' column in price DataFrame for support/resistance.")
    close = df["Close"]
    # Guard against the case where upstream returns a DataFrame (e.g. duplicate
    # column names) instead of a 1D Series. We always want a simple Series.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close


def _find_local_extrema(series: pd.Series, window: int = 2) -> Dict[str, pd.Series]:
    """
    Very simple local minima / maxima detector using a rolling window.
    """
    s = series
    # Shifted versions for neighbourhood comparison.
    prev = s.shift(1)
    nxt = s.shift(-1)

    # Local minima / maxima (1-bar lookback/forward).
    is_local_min = (s < prev) & (s <= nxt)
    is_local_max = (s > prev) & (s >= nxt)

    return {"min": is_local_min, "max": is_local_max}


def _cluster_levels(
    prices: pd.Series,
    mask: pd.Series,
    *,
    kind: str,
    tolerance: float = 0.005,
) -> List[SRLevel]:
    """
    Cluster close-by extrema into consolidated levels.

    tolerance is a relative band (e.g. 0.005 = 0.5%) used to group nearby prices.
    """
    idx = prices.index[mask]
    if len(idx) == 0:
        return []

    pts = prices.loc[idx]
    levels: List[SRLevel] = []

    for dt, price in pts.items():
        if not levels:
            levels.append(SRLevel(price=float(price), kind=kind, touches=1, first_touched=dt, last_touched=dt))
            continue

        # Look for an existing level within the tolerance band.
        assigned = False
        for lvl in levels:
            band = lvl.price * tolerance
            if abs(price - lvl.price) <= band:
                # Merge into this level by averaging and updating metadata.
                lvl.price = float((lvl.price * lvl.touches + float(price)) / (lvl.touches + 1))
                lvl.touches += 1
                lvl.last_touched = dt
                assigned = True
                break

        if not assigned:
            levels.append(SRLevel(price=float(price), kind=kind, touches=1, first_touched=dt, last_touched=dt))

    # Sort by proximity to the latest close (most relevant levels first).
    last_close = float(prices.iloc[-1])
    levels.sort(key=lambda lvl: abs(lvl.price - last_close))
    return levels


def analyze_support_resistance(
    df: pd.DataFrame,
    *,
    max_levels: int = 6,
    tolerance: float = 0.005,
) -> Dict[str, Any]:
    """
    High-level API:

    - Detect local minima (support) and maxima (resistance) on the Close series.
    - Cluster nearby prices into consolidated levels using a relative band.
    - Return a small summary & table that the Streamlit app can display.
    """
    close = _ensure_close(df)
    extrema = _find_local_extrema(close)

    supports = _cluster_levels(close, extrema["min"], kind="support", tolerance=tolerance)
    resistances = _cluster_levels(close, extrema["max"], kind="resistance", tolerance=tolerance)

    # Take the strongest / nearest levels first.
    supports = supports[: max_levels // 2]
    resistances = resistances[: max_levels // 2]

    all_levels = supports + resistances

    if not all_levels:
        return {
            "levels": [],
            "table": pd.DataFrame(
                columns=["kind", "price", "touches", "first_touched", "last_touched"]
            ),
            "nearest_support": None,
            "nearest_resistance": None,
        }

    # Build DataFrame for display.
    rows: List[Dict[str, Any]] = []
    for lvl in all_levels:
        rows.append(
            {
                "kind": lvl.kind,
                "price": lvl.price,
                "touches": lvl.touches,
                "first_touched": lvl.first_touched,
                "last_touched": lvl.last_touched,
            }
        )

    table = pd.DataFrame(rows)

    last_close = float(close.iloc[-1])
    # Nearest support: level below or equal to price with minimum distance.
    supports_below = [lvl for lvl in supports if lvl.price <= last_close]
    nearest_support = (
        min(supports_below, key=lambda lvl: last_close - lvl.price) if supports_below else None
    )

    # Nearest resistance: level above or equal to price with minimum distance.
    resistances_above = [lvl for lvl in resistances if lvl.price >= last_close]
    nearest_resistance = (
        min(resistances_above, key=lambda lvl: lvl.price - last_close) if resistances_above else None
    )

    return {
        "levels": all_levels,
        "table": table,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "last_close": last_close,
    }

