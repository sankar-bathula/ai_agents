# AI Stock Analyzer

An experimental multi-agent stock analysis toolkit with a Streamlit dashboard.  
The goal is to combine market data, technical indicators, fundamentals, and ML models into a single, interactive interface.

## Features (planned)

- **Data Agent**: Fetches OHLCV price data and basic fundamentals from NSE / Yahoo.
- **Technical Analysis Agent**: Computes indicators like RSI, MACD, and moving averages.
- **Fundamental Analysis Agent**: Computes valuation and quality ratios (PE, ROE, Debt/Equity).
- **ML Prediction Agent**: Uses models such as XGBoost, LightGBM, and LSTM for return/direction forecasts.
- **Screener Agent**: Filters and ranks stocks based on technical, fundamental, and ML signals.
- **Dashboard**: Streamlit UI for exploratory analysis and screening.

This repository currently includes a **minimal working demo** that:

- Fetches price data for a single ticker.
- Computes simple technical indicators (RSI and moving averages).
- Fetches a few basic fundamentals and ratios.
- Displays everything in a small Streamlit app.

## Getting started

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows PowerShell
# source .venv/bin/activate  # macOS / Linux
```

### 2. Install dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

> Note: `tensorflow`, `xgboost`, and `lightgbm` can be heavy installs.  
> If you only want to run the minimal demo, you can comment them out in `requirements.txt` and install later.

### 3. Run the Streamlit dashboard

From the repository root:

```bash
streamlit run app/main.py
```

The app will open in your browser (typically at `http://localhost:8501`).

## Project structure

Planned layout (some modules are still stubs):

```text
ai_agents/
  app/
    main.py                 # Streamlit entry point
  agents/
    data_agent/
      __init__.py
      nse_yahoo.py          # price & basic fundamentals from NSE/Yahoo
      financials.py         # deeper financial statements (stub)
    technical_agent/
      __init__.py
      indicators.py         # RSI, MACD, moving averages, etc.
    fundamental_agent/
      __init__.py
      ratios.py             # PE, ROE, Debt/Equity
    ml_agent/
      __init__.py           # XGBoost, LightGBM, LSTM stubs
    screener_agent/
      __init__.py
      rules.py              # rule-based and ML-augmented screeners (stub)
  requirements.txt
  README.md
```

## Minimal demo behavior

The initial Streamlit app does the following for a single ticker:

- Lets you enter a ticker symbol (e.g. `RELIANCE.NS` or `AAPL`).
- Loads daily price data for the last 6 months via the Data Agent.
- Computes RSI and a couple of moving averages via the Technical Agent.
- Pulls a few fundamental fields and computes basic ratios via the Fundamental Agent.
- Displays:
  - Price chart
  - RSI chart
  - Moving averages overlay
  - A small fundamentals/ratio table

As the project grows, you can:

- Expand ML models inside `agents/ml_agent/`.
- Implement richer screeners in `agents/screener_agent/`.
- Add more Streamlit pages (technical view, fundamental view, screener, ML predictions, etc.).

