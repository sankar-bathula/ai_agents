from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yfinance as yf


@dataclass
class NewsItem:
    """
    Lightweight representation of a single news headline for a ticker.
    """

    ticker: str
    title: str
    publisher: Optional[str]
    link: Optional[str]
    published_at: Optional[datetime]
    sentiment_score: float


def _safe_to_datetime(ts: Any) -> Optional[datetime]:
    """
    Convert various timestamp formats from yfinance to a UTC datetime.
    """
    if ts is None:
        return None
    try:
        # yfinance typically returns Unix timestamps for providerPublishTime
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        # Fallback for ISO-style strings
        return datetime.fromisoformat(str(ts)).astimezone(timezone.utc)
    except Exception:
        return None


def _build_lexicon() -> Dict[str, float]:
    """
    Very small, hand-crafted sentiment lexicon for finance news headlines.

    Values are in approximately [-1.0, 1.0].
    This avoids any external ML dependencies and keeps the agent "free".
    """
    positive_words = {
        "beat": 0.8,
        "beats": 0.8,
        "surge": 0.9,
        "surges": 0.9,
        "rally": 0.8,
        "rallies": 0.8,
        "record": 0.7,
        "upgrade": 0.7,
        "upgrades": 0.7,
        "strong": 0.6,
        "growth": 0.6,
        "bullish": 0.9,
        "outperform": 0.8,
        "profit": 0.7,
        "profits": 0.7,
        "gain": 0.6,
        "gains": 0.6,
    }
    negative_words = {
        "miss": -0.8,
        "misses": -0.8,
        "plunge": -0.9,
        "plunges": -0.9,
        "slump": -0.9,
        "slumps": -0.9,
        "downgrade": -0.7,
        "downgrades": -0.7,
        "weak": -0.6,
        "loss": -0.7,
        "losses": -0.7,
        "bearish": -0.9,
        "warning": -0.6,
        "fraud": -1.0,
        "scandal": -0.9,
        "risk": -0.5,
    }
    lexicon: Dict[str, float] = {}
    lexicon.update(positive_words)
    lexicon.update(negative_words)
    return lexicon


_LEXICON = _build_lexicon()


def score_headline_sentiment(text: str) -> float:
    """
    Compute a simple sentiment score for a headline using a small lexicon.

    Returns a value in approximately [-1, 1]. Values near 0 are neutral.
    """
    if not text:
        return 0.0

    tokens = [t.strip(".,!?():;\"'").lower() for t in text.split()]
    scores: List[float] = []
    for tok in tokens:
        if tok in _LEXICON:
            scores.append(_LEXICON[tok])

    if not scores:
        return 0.0

    # Simple average of matched word scores.
    return float(sum(scores) / len(scores))


def fetch_ticker_news(
    ticker: str,
    max_items: int = 30,
) -> List[Dict[str, Any]]:
    """
    Fetch recent news items for a ticker using yfinance's free Yahoo! feed.

    This avoids any paid APIs and keeps the agent self-contained.
    Returns a list of dictionaries as provided by yfinance (possibly trimmed).
    """
    t = yf.Ticker(ticker)
    try:
        raw_news: Iterable[Dict[str, Any]] = getattr(t, "news", []) or []
    except Exception:
        raw_news = []

    news_list: List[Dict[str, Any]] = list(raw_news)
    if max_items > 0:
        news_list = news_list[:max_items]
    return news_list


def analyze_recent_sentiment(
    ticker: str,
    max_items: int = 30,
) -> Dict[str, Any]:
    """
    High-level API:
    - Fetch recent headlines for the ticker.
    - Score each headline for sentiment.
    - Return both per-headline details and an aggregate summary.

    Returns
    -------
    Dict with keys:
    - "items": List[NewsItem]
    - "summary": Dict[str, Any] with aggregate metrics
    """
    raw_items = fetch_ticker_news(ticker, max_items=max_items)

    items: List[NewsItem] = []
    for entry in raw_items:
        title = str(entry.get("title") or "").strip()
        if not title:
            continue

        score = score_headline_sentiment(title)
        item = NewsItem(
            ticker=ticker,
            title=title,
            publisher=entry.get("publisher"),
            link=entry.get("link"),
            published_at=_safe_to_datetime(entry.get("providerPublishTime")),
            sentiment_score=score,
        )
        items.append(item)

    if not items:
        return {
            "items": [],
            "summary": {
                "mean_score": 0.0,
                "headline_count": 0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 1.0,
            },
        }

    df = pd.DataFrame(
        [
            {
                "title": i.title,
                "publisher": i.publisher,
                "link": i.link,
                "published_at": i.published_at,
                "sentiment_score": i.sentiment_score,
            }
            for i in items
        ]
    )

    scores = df["sentiment_score"]
    mean_score = float(scores.mean())
    pos = (scores > 0.1).mean()
    neg = (scores < -0.1).mean()
    neu = 1.0 - pos - neg

    summary = {
        "mean_score": mean_score,
        "headline_count": int(len(df)),
        "positive_ratio": float(pos),
        "negative_ratio": float(neg),
        "neutral_ratio": float(neu),
    }

    return {
        "items": items,
        "summary": summary,
        "table": df,
    }

