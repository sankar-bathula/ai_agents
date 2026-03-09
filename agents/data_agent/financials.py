"""
Stubs for fetching richer financial statements (income statement, balance sheet,
cash flow) for use by fundamental and ML agents.

These implementations are intentionally left as placeholders so you can wire
them to the data provider of your choice (paid API, web-scraper, etc.).
"""

from __future__ import annotations

from typing import Any, Dict


def get_financial_statements(ticker: str) -> Dict[str, Any]:
    """
    Placeholder for fetching detailed financial statements for a ticker.

    Expected structure could include keys such as:
    - "income_statement"
    - "balance_sheet"
    - "cash_flow"
    each mapped to a tabular structure (e.g. pandas.DataFrame or dict of dicts).
    """
    # TODO: Implement using your preferred data source.
    raise NotImplementedError("get_financial_statements is not implemented yet.")

