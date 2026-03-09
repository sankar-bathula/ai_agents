"""
Technical Analysis Agent
------------------------

Provides technical indicators on top of OHLCV price data.

Exports:
- add_basic_indicators: add RSI and moving averages to a price DataFrame.
"""

from .indicators import add_basic_indicators  # noqa: F401

