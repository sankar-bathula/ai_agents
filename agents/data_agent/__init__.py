"""
Data Agent
----------

Responsible for fetching market data (OHLCV) and basic fundamentals from
NSE / Yahoo-style sources.

This package currently exposes high-level convenience functions for:
- get_price_history: historical OHLCV data
- get_fundamentals: basic fundamental snapshot
"""

from .nse_yahoo import get_price_history, get_fundamentals  # noqa: F401

