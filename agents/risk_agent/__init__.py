"""
Risk Analysis Agent
-------------------

Provides simple position-level risk/reward calculations.

Exports:
- analyze_position_risk: compute risk per share, reward per share,
  risk/reward ratio, and optional position sizing suggestions.
"""

from .risk_metrics import analyze_position_risk  # noqa: F401

