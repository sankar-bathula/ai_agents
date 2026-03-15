"""
Support & Resistance Agent
--------------------------

Provides simple support and resistance level detection from OHLC price data.

Exports:
- analyze_support_resistance: high-level helper returning clustered levels and a
  summary suitable for UI display.
"""

from .levels import analyze_support_resistance  # noqa: F401

