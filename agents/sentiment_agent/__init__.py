"""
Sentiment Analysis Agent
------------------------

Provides lightweight, free-news-based sentiment analysis for a single ticker.

Exports:
- analyze_recent_sentiment: fetch Yahoo! Finance headlines via yfinance and
  compute a simple lexicon-based sentiment score.
"""

from .news_sentiment import analyze_recent_sentiment  # noqa: F401

