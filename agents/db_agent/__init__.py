from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


DB_PATH = Path(__file__).resolve().parents[1] / "ai_agents.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """
    Create the analysis_snapshots table if it does not exist.
    """
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                ticker TEXT NOT NULL,
                name TEXT,
                currency TEXT,
                last_close REAL,
                pe_source REAL,
                pe_computed REAL,
                roe_source REAL,
                roe_computed REAL,
                de_source REAL,
                de_computed REAL,
                entry_price REAL,
                stop_loss REAL,
                target_price REAL,
                capital REAL,
                risk_ratio TEXT,
                risk_max_amount REAL,
                risk_position_size REAL,
                sentiment_mean_score REAL,
                sentiment_headline_count INTEGER,
                newsapi_mean_score REAL,
                newsapi_article_count INTEGER,
                last_candle_pattern TEXT,
                nearest_support REAL,
                nearest_resistance REAL,
                raw_sentiment_json TEXT,
                raw_newsapi_json TEXT
            );
            """
        )


def _safe_json(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if is_dataclass(value):
            value = asdict(value)
        return json.dumps(value, default=str)
    except Exception:
        return None


def insert_analysis_snapshot(
    *,
    ticker: str,
    fundamentals: Dict[str, Any],
    ratios: Dict[str, Any],
    risk_result: Optional[Dict[str, Any]] = None,
    sentiment_result: Optional[Dict[str, Any]] = None,
    newsapi_result: Optional[Dict[str, Any]] = None,
    last_candle_pattern: Optional[str] = None,
    nearest_support_price: Optional[float] = None,
    nearest_resistance_price: Optional[float] = None,
    last_close: Optional[float] = None,
) -> None:
    """
    Persist a single snapshot of the current analysis into SQLite.
    """
    created_at = datetime.now(timezone.utc).isoformat()

    name = fundamentals.get("longName") or fundamentals.get("shortName")
    currency = fundamentals.get("currency")

    pe_source = fundamentals.get("trailingPE")
    roe_source = fundamentals.get("returnOnEquity")
    de_source = fundamentals.get("debtToEquity")

    pe_computed = ratios.get("PE") if ratios is not None else None
    roe_computed = ratios.get("ROE") if ratios is not None else None
    de_computed = ratios.get("DE") if ratios is not None else None

    entry_price = risk_result.get("entry_price") if risk_result else None
    stop_loss = risk_result.get("stop_loss") if risk_result else None
    target_price = risk_result.get("target_price") if risk_result else None
    capital = risk_result.get("capital") if risk_result else None
    risk_ratio = risk_result.get("ratio_display") if risk_result else None
    risk_max_amount = risk_result.get("max_risk_amount") if risk_result else None
    risk_position_size = risk_result.get("recommended_position_size") if risk_result else None

    sentiment_mean_score = None
    sentiment_headline_count = None
    if sentiment_result:
        s_summary = sentiment_result.get("summary") or {}
        sentiment_mean_score = s_summary.get("mean_score")
        sentiment_headline_count = s_summary.get("headline_count")

    newsapi_mean_score = None
    newsapi_article_count = None
    if newsapi_result:
        n_summary = newsapi_result.get("summary") or {}
        newsapi_mean_score = n_summary.get("mean_score")
        newsapi_article_count = n_summary.get("article_count")

    raw_sentiment_json = _safe_json(sentiment_result)
    raw_newsapi_json = _safe_json(newsapi_result)

    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO analysis_snapshots (
                created_at,
                ticker,
                name,
                currency,
                last_close,
                pe_source,
                pe_computed,
                roe_source,
                roe_computed,
                de_source,
                de_computed,
                entry_price,
                stop_loss,
                target_price,
                capital,
                risk_ratio,
                risk_max_amount,
                risk_position_size,
                sentiment_mean_score,
                sentiment_headline_count,
                newsapi_mean_score,
                newsapi_article_count,
                last_candle_pattern,
                nearest_support,
                nearest_resistance,
                raw_sentiment_json,
                raw_newsapi_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                created_at,
                ticker,
                name,
                currency,
                last_close,
                pe_source,
                pe_computed,
                roe_source,
                roe_computed,
                de_source,
                de_computed,
                entry_price,
                stop_loss,
                target_price,
                capital,
                risk_ratio,
                risk_max_amount,
                risk_position_size,
                sentiment_mean_score,
                sentiment_headline_count,
                newsapi_mean_score,
                newsapi_article_count,
                last_candle_pattern,
                nearest_support_price,
                nearest_resistance_price,
                raw_sentiment_json,
                raw_newsapi_json,
            ),
        )

