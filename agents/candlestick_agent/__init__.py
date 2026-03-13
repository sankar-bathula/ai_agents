"""
Candlestick Pattern Agent
-------------------------

Provides simple pattern detection on OHLC price data.

Exports:
- analyze_candlestick_patterns: high-level helper returning a summary and table of
  recent patterns suitable for UI display.
"""

from .patterns import analyze_candlestick_patterns  # noqa: F401

