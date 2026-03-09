from __future__ import annotations

from typing import Any, Dict


def compute_basic_ratios(
    last_price: float | None,
    fundamentals: Dict[str, Any],
) -> Dict[str, float | None]:
    """
    Compute a small set of core ratios from price and fundamental fields.

    Inputs are intentionally permissive and may contain missing values;
    the function returns None for ratios that cannot be computed safely.
    """
    trailing_pe = fundamentals.get("trailingPE")
    roe = fundamentals.get("returnOnEquity")
    de_raw = fundamentals.get("debtToEquity")

    # Attempt to compute PE from explicit trailingPE or fallback to price / EPS style.
    pe_value: float | None = None
    if isinstance(trailing_pe, (int, float)):
        pe_value = float(trailing_pe)

    # ROE and D/E are typically already ratio values from the data source.
    roe_value: float | None = float(roe) if isinstance(roe, (int, float)) else None
    de_value: float | None = float(de_raw) if isinstance(de_raw, (int, float)) else None

    return {
        "PE": pe_value,
        "ROE": roe_value,
        "DE": de_value,
    }

