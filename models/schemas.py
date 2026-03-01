"""
Pydantic v2 request / response models.
"""
from typing import Literal
from pydantic import BaseModel, Field


class Condition(BaseModel):
    """A single screening rule."""
    lhs_timeframe: str = Field(
        default="daily",
        description="Timeframe for the left-hand side: 'daily' | 'weekly' | 'monthly'",
    )
    indicator: str = Field(
        ...,
        description="Left-hand side indicator key, e.g. 'close', 'rsi_14', 'volume'",
        examples=["close"],
    )
    operator: str = Field(
        ...,
        description="Comparison operator: '>', '<', '>=', '<=', '=', 'crosses_above', 'crosses_below'",
        examples=[">"],
    )
    rhs_timeframe: str = Field(
        default="daily",
        description="Timeframe for the right-hand side: 'daily' | 'weekly' | 'monthly'",
    )
    value: str = Field(
        ...,
        description="Right-hand side: another indicator key OR a numeric string, e.g. 'sma_20' or '50'",
        examples=["sma_20"],
    )


class ScreenRequest(BaseModel):
    """Payload sent by the mobile app to run a scan."""
    conditions: list[Condition] = Field(..., min_length=1)
    match: Literal["all", "any"] = Field(
        default="all",
        description="'all' = every condition must be true (AND), 'any' = at least one (OR)",
    )


class StockMatch(BaseModel):
    """One stock that passed the screening criteria."""
    symbol: str
    last_close: float
    last_volume: int
    change_pct: float = Field(description="1-day % change")


class ScreenResponse(BaseModel):
    """Full response returned to the mobile app."""
    matched_stocks: list[StockMatch]
    total_matched: int
    total_screened: int
    conditions_used: list[Condition]
    match_mode: str
