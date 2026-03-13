from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from agents.sentiment_agent.news_sentiment import score_headline_sentiment


NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"


@dataclass
class NewsAPIArticle:
    query: str
    title: str
    source: Optional[str]
    url: Optional[str]
    published_at: Optional[datetime]
    description: Optional[str]
    sentiment_score: float


def _parse_published_at(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        # NewsAPI uses ISO 8601 like "2026-03-11T12:34:56Z"
        s = str(value).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


def _get_api_key(explicit_key: Optional[str] = None) -> str:
    key = (explicit_key or os.getenv("NEWSAPI_KEY") or "").strip()
    if not key:
        raise ValueError(
            "Missing NewsAPI key. Set environment variable NEWSAPI_KEY (or pass api_key=...)."
        )
    return key


def fetch_newsapi_articles(
    query: str,
    *,
    api_key: Optional[str] = None,
    language: str = "en",
    sort_by: str = "publishedAt",
    max_items: int = 30,
) -> List[Dict[str, Any]]:
    """
    Fetch recent articles from NewsAPI 'everything' endpoint for the given query.
    """
    key = _get_api_key(api_key)
    page_size = max(1, min(int(max_items), 100))

    params = {
        "q": query,
        "language": language,
        "sortBy": sort_by,
        "pageSize": page_size,
    }
    headers = {"X-Api-Key": key}

    resp = requests.get(NEWSAPI_EVERYTHING_URL, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("status") != "ok":
        message = payload.get("message") or "Unknown error from NewsAPI."
        raise ValueError(f"NewsAPI error: {message}")

    articles = payload.get("articles") or []
    if not isinstance(articles, list):
        return []
    return articles[:page_size]


def analyze_newsapi_sentiment(
    query: str,
    *,
    api_key: Optional[str] = None,
    max_items: int = 30,
    language: str = "en",
) -> Dict[str, Any]:
    """
    High-level API:
    - Fetch recent NewsAPI articles for the query.
    - Score each item using a simple lexicon-based sentiment.
    - Return per-article details, a summary, and a DataFrame.
    """
    raw_items = fetch_newsapi_articles(
        query,
        api_key=api_key,
        max_items=max_items,
        language=language,
    )

    items: List[NewsAPIArticle] = []
    for a in raw_items:
        title = str(a.get("title") or "").strip()
        if not title:
            continue

        desc = a.get("description")
        desc_str = str(desc).strip() if desc is not None else None

        # Combine title + description for slightly richer signal.
        combined = title if not desc_str else f"{title}. {desc_str}"
        score = score_headline_sentiment(combined)

        source = None
        src = a.get("source") or {}
        if isinstance(src, dict):
            source = src.get("name") or src.get("id")

        items.append(
            NewsAPIArticle(
                query=query,
                title=title,
                source=str(source).strip() if source else None,
                url=a.get("url"),
                published_at=_parse_published_at(a.get("publishedAt")),
                description=desc_str,
                sentiment_score=float(score),
            )
        )

    if not items:
        return {
            "items": [],
            "summary": {
                "mean_score": 0.0,
                "article_count": 0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 1.0,
            },
            "table": pd.DataFrame(
                columns=["published_at", "source", "title", "description", "url", "sentiment_score"]
            ),
        }

    df = pd.DataFrame(
        [
            {
                "published_at": i.published_at,
                "source": i.source,
                "title": i.title,
                "description": i.description,
                "url": i.url,
                "sentiment_score": i.sentiment_score,
            }
            for i in items
        ]
    )

    scores = df["sentiment_score"]
    mean_score = float(scores.mean())
    pos = float((scores > 0.1).mean())
    neg = float((scores < -0.1).mean())
    neu = float(1.0 - pos - neg)

    summary = {
        "mean_score": mean_score,
        "article_count": int(len(df)),
        "positive_ratio": pos,
        "negative_ratio": neg,
        "neutral_ratio": neu,
    }

    return {
        "items": items,
        "summary": summary,
        "table": df,
    }

