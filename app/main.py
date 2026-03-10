from __future__ import annotations

import pandas as pd
import streamlit as st

from agents.data_agent import get_fundamentals, get_price_history
from agents.fundamental_agent import compute_basic_ratios
from agents.sentiment_agent import analyze_recent_sentiment
from agents.technical_agent import add_basic_indicators


st.set_page_config(
    page_title="AI Stock Analyzer (Minimal Demo)",
    layout="wide",
)

st.title("AI Stock Analyzer")
st.caption("Minimal demo: Data, Technical, and Fundamental agents for a single ticker")


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

    if show_sentiment:
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

    if show_raw:
        st.subheader("Raw enriched data (tail)")
        st.dataframe(enriched.tail(100))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

