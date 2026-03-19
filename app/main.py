from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure the repository root is on sys.path so `import agents.*` works
# even when running `streamlit run app/main.py` from inside `app/`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.data_agent import get_fundamentals, get_price_history
from agents.fundamental_agent import compute_basic_ratios
from agents.newsapi_agent import analyze_newsapi_sentiment
from agents.risk_agent import analyze_position_risk
from agents.sentiment_agent import analyze_recent_sentiment
from agents.technical_agent import add_basic_indicators
from agents.candlestick_agent import analyze_candlestick_patterns
from agents.support_resistance_agent import analyze_support_resistance
from agents.db_agent import init_db, insert_analysis_snapshot


load_dotenv()

st.set_page_config(
    page_title="AI Stock Analyzer ",
    layout="wide",
)

st.title("AI Stock Analyzer")
st.caption("Minimal demoS: Data, Technical, and Fundamental agents for a single ticker")

# Ensure database is ready.
init_db()


@st.cache_data(show_spinner=False)
def load_price_data(ticker: str) -> pd.DataFrame:
    return get_price_history(ticker, period="6mo", interval="1d")


@st.cache_data(show_spinner=False)
def load_fundamentals(ticker: str):
    return get_fundamentals(ticker)


@st.cache_data(show_spinner=False)
def load_sentiment(ticker: str):
    """
    Use the Sentiment Agent to fetch recent news and compute a simple score.
    """
    return analyze_recent_sentiment(ticker)


@st.cache_data(show_spinner=False)
def load_newsapi(query: str, max_items: int = 30, language: str = "en"):
    """
    Fetch NewsAPI articles and compute a simple sentiment score.
    Requires env var NEWSAPI_KEY (loaded from .env if present).
    """
    return analyze_newsapi_sentiment(query, max_items=max_items, language=language)


def main() -> None:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        ticker = st.text_input(
            "Ticker symbol",
            value="RELIANCE.NS",
            help="Use NSE-style symbols like 'RELIANCE.NS' or US symbols like 'AAPL'.",
        )

    with col_right:
        show_raw = st.checkbox("Show raw data table", value=False)
        show_sentiment = st.checkbox("Show news sentiment", value=True)
        show_newsapi = st.checkbox("Show NewsAPI sentiment", value=False)
        show_candles = st.checkbox("Show candlestick patterns", value=False)
        show_sr = st.checkbox("Show support / resistance", value=False)

    if not ticker:
        st.info("Enter a ticker symbol to begin.")
        return

    try:
        prices = load_price_data(ticker)
    except Exception as exc:  # pragma: no cover - UI path
        st.error(f"Failed to load price data for '{ticker}': {exc}")
        return

    if prices.empty:
        st.warning(f"No price data available for '{ticker}'.")
        return

    enriched = add_basic_indicators(prices)

    fundamentals = load_fundamentals(ticker)
    # Use .iloc[-1].item() to avoid FutureWarning on Series -> float.
    last_close = enriched["Close"].iloc[-1].item()
    ratios = compute_basic_ratios(last_close, fundamentals)

    # ---------- Layout ----------
    st.subheader(f"Price overview – {ticker}")
    st.line_chart(enriched["Close"], height=300)

    ma_cols = [c for c in enriched.columns if isinstance(c, str) and c.startswith("MA_")]
    if ma_cols:
        st.line_chart(enriched[ma_cols], height=200)

    st.subheader("RSI (14)")
    if "RSI_14" in enriched.columns:
        st.line_chart(enriched["RSI_14"], height=200)
    else:
        st.write("RSI indicator not available.")

    candle_result = None
    if show_candles:
        st.subheader("Candlestick pattern scan")
        try:
            candle_result = analyze_candlestick_patterns(prices)
            last_pattern = candle_result.get("last_pattern")
            last_date = candle_result.get("last_pattern_date")
            table = candle_result.get("table")

            if last_pattern:
                if last_date is not None:
                    st.markdown(
                        f"**Most recent pattern:** {last_pattern} on {last_date}"
                    )
                else:
                    st.markdown(f"**Most recent pattern:** {last_pattern}")
            else:
                st.info("No recent candlestick patterns detected.")

            if table is not None and not table.empty:
                table = table.copy()
                # Ensure index / date is stringified for Streamlit/Arrow safety.
                table["date"] = table["date"].astype(str)
                st.dataframe(table.tail(60))
        except Exception as exc:  # pragma: no cover - UI path
            st.info(f"Candlestick analysis unavailable: {exc}")

    sr_result = None
    if show_sr:
        st.subheader("Support & resistance levels")
        try:
            sr_result = analyze_support_resistance(prices)
            nearest_support = sr_result.get("nearest_support")
            nearest_resistance = sr_result.get("nearest_resistance")
            last_close = sr_result.get("last_close")
            table = sr_result.get("table")

            if last_close is not None:
                st.markdown(f"**Last close:** {last_close:.2f}")

            if nearest_support is not None:
                st.markdown(
                    f"**Nearest support:** {nearest_support.price:.2f} "
                    f"(touches: {nearest_support.touches})"
                )
            if nearest_resistance is not None:
                st.markdown(
                    f"**Nearest resistance:** {nearest_resistance.price:.2f} "
                    f"(touches: {nearest_resistance.touches})"
                )
            if nearest_support is None and nearest_resistance is None:
                st.info("No clear support or resistance levels detected yet.")

            if table is not None and not table.empty:
                table = table.copy()
                # Stringify dates for Streamlit/Arrow.
                if "first_touched" in table.columns:
                    table["first_touched"] = table["first_touched"].astype(str)
                if "last_touched" in table.columns:
                    table["last_touched"] = table["last_touched"].astype(str)
                st.dataframe(table)
        except Exception as exc:  # pragma: no cover - UI path
            st.info(f"Support / resistance analysis unavailable: {exc}")

    st.subheader("Fundamentals & basic ratios")
    fundamentals_display = {
        "Name": fundamentals.get("longName") or fundamentals.get("shortName"),
        "Currency": fundamentals.get("currency"),
        "Last price": fundamentals.get("price") or last_close,
        "Market cap": fundamentals.get("marketCap"),
        "Trailing PE": fundamentals.get("trailingPE"),
        "Forward PE": fundamentals.get("forwardPE"),
        "ROE (source)": fundamentals.get("returnOnEquity"),
        "Debt/Equity (source)": fundamentals.get("debtToEquity"),
        "PE (computed)": ratios.get("PE"),
        "ROE (computed)": ratios.get("ROE"),
        "D/E (computed)": ratios.get("DE"),
    }
    df_fund = pd.DataFrame.from_dict(
        fundamentals_display, orient="index", columns=["Value"]
    )
    # Ensure Arrow compatibility by converting the Value column to strings.
    df_fund["Value"] = df_fund["Value"].astype(str)
    st.table(df_fund)

    st.subheader("Risk / Reward analysis")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        entry_price = st.number_input(
            "Entry price",
            value=float(last_close),
            min_value=0.0,
            format="%.2f",
        )
    with col_r2:
        stop_loss = st.number_input(
            "Stop loss",
            value=float(last_close * 0.95),
            min_value=0.0,
            format="%.2f",
        )
    with col_r3:
        target_price = st.number_input(
            "Target price",
            value=float(last_close * 1.10),
            min_value=0.0,
            format="%.2f",
        )
    with col_r4:
        capital_input = st.number_input(
            "Account capital (optional)",
            value=0.0,
            min_value=0.0,
            format="%.2f",
            help="Used to suggest a position size based on max risk per trade.",
        )

    max_risk_pct_ui = st.slider(
        "Max risk % per trade",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5,
        help="Capital percentage you are willing to risk on a single trade.",
    )

    try:
        capital = capital_input if capital_input > 0 else None
        risk_result = analyze_position_risk(
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            capital=capital,
            max_risk_pct=max_risk_pct_ui / 100.0,
        )

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Risk / share", f"{risk_result['risk_per_share']:.2f}")
        with col_m2:
            st.metric("Reward / share", f"{risk_result['reward_per_share']:.2f}")
        with col_m3:
            st.metric("Risk-Reward", risk_result["ratio_display"])
        with col_m4:
            st.metric(
                "% to stop / target",
                f"{risk_result['percent_to_stop']:.1f}% / {risk_result['percent_to_target']:.1f}%",
            )

        if capital is not None and risk_result.get("max_risk_amount") is not None:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.metric(
                    "Max risk amount",
                    f"{risk_result['max_risk_amount']:.2f}",
                    help="Maximum risk per trade based on your capital and max risk %.",
                )
            with col_c2:
                size = risk_result.get("recommended_position_size")
                size_display = str(size) if size is not None else "N/A"
                st.metric(
                    "Suggested position size (shares)",
                    size_display,
                )
    except Exception as exc:  # pragma: no cover - UI path
        st.info(f"Unable to compute risk/reward: {exc}")

    #if show_sentiment:
        st.subheader("News sentiment (free Yahoo! Finance feed)")
        try:
            sentiment_result = load_sentiment(ticker)
            summary = sentiment_result.get("summary", {})
            mean_score = summary.get("mean_score", 0.0)
            headline_count = summary.get("headline_count", 0)

            # Simple textual interpretation of the score.
            if mean_score > 0.15:
                sentiment_label = "Bullish"
            elif mean_score < -0.15:
                sentiment_label = "Bearish"
            else:
                sentiment_label = "Neutral"

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Average headline score", f"{mean_score:.2f}")
            with col_s2:
                st.metric("Interpretation", sentiment_label)
            with col_s3:
                st.metric("Headline count", str(headline_count))

            table = sentiment_result.get("table")
            if table is not None and not table.empty:
                # Convert datetimes to string for display safety.
                table = table.copy()
                if "published_at" in table.columns:
                    table["published_at"] = table["published_at"].astype(str)
                st.caption("Recent headlines with sentiment scores")
                st.dataframe(table[["published_at", "publisher", "title", "sentiment_score"]])
            else:
                st.info("No recent news headlines available for this ticker.")
        except Exception as exc:  # pragma: no cover - UI path
            st.warning(f"Sentiment analysis unavailable: {exc}")

    newsapi_result = None
    if show_newsapi:
        st.subheader("News sentiment (NewsAPI.org)")
        col_n1, col_n2, col_n3 = st.columns([2, 1, 1])
        with col_n1:
            default_news_query = fundamentals.get("longName") or fundamentals.get("shortName") or ticker
            # NewsAPI works better with company names than exchange-suffixed tickers (e.g. RELIANCE.NS).
            if isinstance(default_news_query, str) and "." in default_news_query:
                default_news_query = default_news_query.split(".", 1)[0]
            news_query = st.text_input(
                "NewsAPI query",
                value=str(default_news_query),
                help="You can use a ticker, company name, or phrase (e.g. 'Reliance Industries').",
            )
        with col_n2:
            news_lang = st.selectbox("Language", options=["en"], index=0)
        with col_n3:
            news_max = st.number_input(
                "Max articles",
                min_value=5,
                max_value=100,
                value=30,
                step=5,
            )

        try:
            newsapi_result = load_newsapi(news_query, max_items=int(news_max), language=news_lang)
            summary = newsapi_result.get("summary", {})
            mean_score = float(summary.get("mean_score", 0.0))
            article_count = int(summary.get("article_count", 0))

            if mean_score > 0.15:
                sentiment_label = "Bullish"
            elif mean_score < -0.15:
                sentiment_label = "Bearish"
            else:
                sentiment_label = "Neutral"

            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                st.metric("Average article score", f"{mean_score:.2f}")
            with col_a2:
                st.metric("Interpretation", sentiment_label)
            with col_a3:
                st.metric("Article count", str(article_count))

            table = newsapi_result.get("table")
            if table is not None and not table.empty:
                table = table.copy()
                if "published_at" in table.columns:
                    table["published_at"] = table["published_at"].astype(str)
                st.caption("Recent articles with sentiment scores")
                st.dataframe(
                    table[
                        ["published_at", "source", "title", "description", "sentiment_score", "url"]
                    ]
                )
            else:
                st.info("No NewsAPI articles returned for this query.")
        except Exception as exc:  # pragma: no cover - UI path
            st.warning(
                "NewsAPI sentiment unavailable. Ensure NEWSAPI_KEY is set and your plan allows the request. "
                f"Error: {exc}"
            )

    if show_raw:
        st.subheader("Raw enriched data (tail)")
        st.dataframe(enriched.tail(100))

    # Persist a snapshot of this analysis into SQLite.
    try:
        nearest_support_price = None
        nearest_resistance_price = None
        if sr_result is not None:
            ns = sr_result.get("nearest_support")
            nr = sr_result.get("nearest_resistance")
            nearest_support_price = float(ns.price) if ns is not None else None
            nearest_resistance_price = float(nr.price) if nr is not None else None

        last_candle_pattern = None
        if candle_result is not None:
            last_candle_pattern = candle_result.get("last_pattern")

        insert_analysis_snapshot(
            ticker=ticker,
            fundamentals=fundamentals,
            ratios=ratios,
            risk_result=risk_result if "risk_result" in locals() else None,
            sentiment_result=sentiment_result if "sentiment_result" in locals() else None,
            newsapi_result=newsapi_result,
            last_candle_pattern=last_candle_pattern,
            nearest_support_price=nearest_support_price,
            nearest_resistance_price=nearest_resistance_price,
            last_close=last_close,
        )
    except Exception as exc:  # pragma: no cover - UI path
        st.info(f"Could not save analysis snapshot to DB: {exc}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

