"""Pydantic models for portfolio-level metrics used by Position Manager."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from pydantic import BaseModel, Field


class PortfolioByAsset(BaseModel):
    """Breakdown of portfolio metrics by asset."""

    exposure_usdt: Decimal = Field(..., description="Exposure in USDT for the asset")
    unrealized_pnl_usdt: Decimal = Field(
        ...,
        description="Unrealized profit/loss in USDT for the asset",
    )
    realized_pnl_usdt: Decimal = Field(
        ...,
        description="Realized profit/loss in USDT for the asset",
    )
    size: Decimal = Field(..., description="Net position size for the asset")


class PortfolioMetrics(BaseModel):
    """Aggregate portfolio metrics calculated on-demand."""

    total_exposure_usdt: Decimal = Field(
        ...,
        description="Sum of ABS(size) * current_price for all positions with current_price",
    )
    total_unrealized_pnl_usdt: Decimal = Field(
        ...,
        description="Sum of unrealized_pnl across all positions",
    )
    total_realized_pnl_usdt: Decimal = Field(
        ...,
        description="Sum of realized_pnl across all positions",
    )
    portfolio_value_usdt: Decimal = Field(
        ...,
        description="Sum of size * current_price for all positions with current_price",
    )
    open_positions_count: int = Field(
        ...,
        description="Number of positions where size != 0",
    )
    long_positions_count: int = Field(
        ...,
        description="Number of long positions (size > 0)",
    )
    short_positions_count: int = Field(
        ...,
        description="Number of short positions (size < 0)",
    )
    net_exposure_usdt: Decimal = Field(
        ...,
        description="Net exposure in USDT (long minus short exposure)",
    )
    by_asset: Dict[str, PortfolioByAsset] = Field(
        default_factory=dict,
        description="Per-asset breakdown of portfolio metrics",
    )
    calculated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when metrics were calculated",
    )


class PortfolioExposure(BaseModel):
    """Slim response model with only exposure-related fields."""

    total_exposure_usdt: Decimal = Field(..., description="Total portfolio exposure in USDT")
    calculated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when exposure was calculated",
    )


class PortfolioPnL(BaseModel):
    """Slim response model with only PnL-related fields."""

    total_unrealized_pnl_usdt: Decimal = Field(
        ...,
        description="Total unrealized PnL in USDT",
    )
    total_realized_pnl_usdt: Decimal = Field(
        ...,
        description="Total realized PnL in USDT",
    )
    total_pnl_usdt: Decimal = Field(
        ...,
        description="Combined PnL (realized + unrealized) in USDT",
    )
    calculated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when PnL was calculated",
    )



