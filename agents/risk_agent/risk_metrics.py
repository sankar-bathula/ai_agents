from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Any, Dict, Optional


@dataclass
class PositionRisk:
    entry_price: float
    stop_loss: float
    target_price: float
    risk_per_share: float
    reward_per_share: float
    ratio: float
    ratio_display: str
    risk_amount: Optional[float]
    reward_amount: Optional[float]
    percent_to_stop: float
    percent_to_target: float
    max_risk_amount: Optional[float]
    recommended_position_size: Optional[int]


def _validate_prices(entry_price: float, stop_loss: float, target_price: float) -> None:
    if entry_price <= 0:
        raise ValueError("Entry price must be positive.")
    if stop_loss <= 0:
        raise ValueError("Stop loss must be positive.")
    if target_price <= 0:
        raise ValueError("Target price must be positive.")
    if stop_loss >= entry_price:
        raise ValueError("Stop loss must be below entry price.")
    if target_price <= entry_price:
        raise ValueError("Target price must be above entry price.")


def _format_ratio(ratio: float) -> str:
    """
    Represent a numeric risk/reward ratio as a human-readable string like '1:2.50'.
    """
    if ratio <= 0:
        return "N/A"
    reward_multiple = 1.0 / ratio
    return f"1:{reward_multiple:.2f}"


def analyze_position_risk(
    entry_price: float,
    stop_loss: float,
    target_price: float,
    *,
    capital: Optional[float] = None,
    max_risk_pct: float = 0.01,
    position_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute basic risk / reward metrics for a single position.

    Parameters
    ----------
    entry_price:
        Proposed entry price for the trade.
    stop_loss:
        Price at which you would exit if the trade goes against you.
    target_price:
        Price at which you would take profit.
    capital:
        Optional account equity. If provided (and positive), the function will
        also compute a recommended position size based on `max_risk_pct`
        unless an explicit `position_size` is given.
    max_risk_pct:
        Maximum fraction of capital you are willing to risk on this trade,
        e.g. 0.01 for 1%. Ignored if `capital` is not provided.
    position_size:
        Optional existing or planned number of shares. When provided, the
        function will compute absolute risk/reward in currency terms.

    Returns
    -------
    Dict[str, Any]
        Keys include:
        - risk_per_share
        - reward_per_share
        - ratio (risk / reward as a float)
        - ratio_display (string like '1:2.50')
        - risk_amount (absolute risk for the given position_size, if any)
        - reward_amount (absolute reward for the given position_size, if any)
        - percent_to_stop
        - percent_to_target
        - max_risk_amount (based on capital and max_risk_pct, if any)
        - recommended_position_size (integer, if computable)
    """
    _validate_prices(entry_price, stop_loss, target_price)

    risk_per_share = entry_price - stop_loss
    reward_per_share = target_price - entry_price
    ratio = risk_per_share / reward_per_share
    ratio_display = _format_ratio(ratio)

    percent_to_stop = (risk_per_share / entry_price) * 100.0
    percent_to_target = (reward_per_share / entry_price) * 100.0

    risk_amount: Optional[float] = None
    reward_amount: Optional[float] = None
    if position_size is not None and position_size > 0:
        risk_amount = risk_per_share * position_size
        reward_amount = reward_per_share * position_size

    max_risk_amount: Optional[float] = None
    recommended_position_size: Optional[int] = None
    if capital is not None and capital > 0 and risk_per_share > 0:
        if max_risk_pct <= 0:
            raise ValueError("max_risk_pct must be positive when capital is provided.")
        max_risk_amount = capital * max_risk_pct
        # Use floor to ensure we do not exceed the specified risk.
        size = floor(max_risk_amount / risk_per_share)
        if size > 0:
            recommended_position_size = int(size)

    result = PositionRisk(
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        risk_per_share=risk_per_share,
        reward_per_share=reward_per_share,
        ratio=ratio,
        ratio_display=ratio_display,
        risk_amount=risk_amount,
        reward_amount=reward_amount,
        percent_to_stop=percent_to_stop,
        percent_to_target=percent_to_target,
        max_risk_amount=max_risk_amount,
        recommended_position_size=recommended_position_size,
    )

    return result.__dict__.copy()

