"""
Placeholder for stock screening logic.

You can use this module to define:
- Purely rule-based filters (e.g. RSI between 30 and 60, price above MA50,
  PE below sector median, etc.).
- ML-augmented scores using outputs from ML models.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd


def example_rule_based_screener(
    universe: Iterable[str],
    data_loader,
    technical_builder,
    fundamental_builder,
) -> pd.DataFrame:
    """
    Example (non-performant) skeleton for a screener function.

    Parameters
    ----------
    universe:
        Iterable of ticker symbols.
    data_loader:
        Callable(ticker) -> price_df, fundamentals_dict.
    technical_builder:
        Callable(price_df) -> price_df_with_indicators.
    fundamental_builder:
        Callable(last_price, fundamentals_dict) -> ratios_dict.
    """
    rows: List[Dict[str, Any]] = []
    for ticker in universe:
        price_df, fundamentals = data_loader(ticker)
        enriched = technical_builder(price_df)
        ratios = fundamental_builder(enriched["Close"].iloc[-1], fundamentals)

        latest = enriched.iloc[-1]
        row: Dict[str, Any] = {
            "ticker": ticker,
            "close": float(latest["Close"]),
            "RSI_14": float(latest.get("RSI_14", float("nan"))),
        }
        row.update(ratios)
        rows.append(row)

    return pd.DataFrame(rows)

