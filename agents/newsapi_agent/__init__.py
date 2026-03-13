"""
NewsAPI Agent
-------------

Fetches recent articles from NewsAPI.org and performs lightweight sentiment
analysis using the existing lexicon-based scorer from `agents.sentiment_agent`.

Configuration
-------------
Set the environment variable `NEWSAPI_KEY` (or place it in a `.env` file if you
use python-dotenv).

Exports:
- analyze_newsapi_sentiment: fetch + score recent articles for a query/ticker.
"""

from .newsapi_sentiment import analyze_newsapi_sentiment  # noqa: F401

