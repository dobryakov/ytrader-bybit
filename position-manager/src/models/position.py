"""Pydantic models for positions and position snapshots used by Position Manager."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, computed_field
from ..config.logging import get_logger

logger = get_logger(__name__)


class Position(BaseModel):
    """Unified position entity for active and historical positions.

    This model follows the unified architecture from position-manager-2.0.md:
    - One table `positions` for all positions (active and closed)
    - Active position: closed_at IS NULL and size != 0
    - Closed position: closed_at IS NOT NULL and size = 0
    - Supports optimistic locking via `version`
    - Includes all fields required for portfolio calculations and ML features
    """

    # Identification
    id: UUID = Field(default_factory=uuid4, description="Unique position identifier")
    asset: str = Field(
        ...,
        description="Trading pair symbol (e.g., 'BTCUSDT')",
        max_length=20,
    )
    mode: str = Field(
        default="one-way",
        description="Trading mode: 'one-way' or 'hedge'",
        max_length=20,
    )
    
    # Position state
    size: Decimal = Field(
        ...,
        description="Net position size (positive = long, negative = short, zero = closed)",
    )
    average_entry_price: Optional[Decimal] = Field(
        None,
        description="Average entry price for position",
        gt=0,
    )
    current_price: Optional[Decimal] = Field(
        None,
        description="Latest markPrice used for portfolio calculations",
    )
    
    # PnL
    unrealized_pnl: Decimal = Field(
        default=Decimal("0"),
        description="Current unrealized profit/loss",
    )
    realized_pnl: Decimal = Field(
        default=Decimal("0"),
        description="Cumulative realized profit/loss",
    )
    
    # Hedge mode
    long_size: Optional[Decimal] = Field(
        None,
        description="Long position size (for hedge mode)",
    )
    short_size: Optional[Decimal] = Field(
        None,
        description="Short position size (for hedge mode)",
    )
    long_avg_price: Optional[Decimal] = Field(
        None,
        description="Long average entry price (for hedge mode)",
        gt=0,
    )
    short_avg_price: Optional[Decimal] = Field(
        None,
        description="Short average entry price (for hedge mode)",
        gt=0,
    )
    
    # Financial indicators from Bybit
    leverage: Optional[Decimal] = Field(None, description="Position leverage")
    position_value: Optional[Decimal] = Field(None, description="Position value")
    liq_price: Optional[Decimal] = Field(None, description="Liquidation price")
    bust_price: Optional[Decimal] = Field(None, description="Bust price")
    take_profit: Optional[Decimal] = Field(None, description="Take profit price")
    stop_loss: Optional[Decimal] = Field(None, description="Stop loss price")
    cum_realised_pnl: Optional[Decimal] = Field(None, description="Cumulative realized PnL from Bybit")
    cum_unrealised_pnl: Optional[Decimal] = Field(None, description="Cumulative unrealized PnL from Bybit")
    
    # Fees and margins
    total_fees: Decimal = Field(
        default=Decimal("0"),
        description="Total fees for this position",
    )
    opening_fees: Optional[Decimal] = Field(None, description="Opening fees")
    closing_fees: Optional[Decimal] = Field(None, description="Closing fees")
    margin_used: Optional[Decimal] = Field(None, description="Margin used")
    available_margin: Optional[Decimal] = Field(None, description="Available margin")
    maintenance_margin: Optional[Decimal] = Field(None, description="Maintenance margin")
    
    # Volume tracking
    max_size: Optional[Decimal] = Field(None, description="Maximum size during position lifetime")
    min_size: Optional[Decimal] = Field(None, description="Minimum size during position lifetime")
    total_volume_traded: Decimal = Field(
        default=Decimal("0"),
        description="Total volume traded for this position",
    )
    
    # Price tracking
    first_entry_price: Optional[Decimal] = Field(None, description="First entry price", gt=0)
    last_entry_price: Optional[Decimal] = Field(None, description="Last entry price", gt=0)
    exit_price: Optional[Decimal] = Field(None, description="Exit price (for closed positions)", gt=0)
    
    # PnL metrics
    peak_unrealized_pnl: Optional[Decimal] = Field(None, description="Peak unrealized PnL")
    peak_unrealized_pnl_at: Optional[datetime] = Field(None, description="Timestamp of peak unrealized PnL")
    worst_unrealized_pnl: Optional[Decimal] = Field(None, description="Worst unrealized PnL")
    worst_unrealized_pnl_at: Optional[datetime] = Field(None, description="Timestamp of worst unrealized PnL")
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Position record creation timestamp (also represents opening time since each record is created once)",
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp",
    )
    closed_at: Optional[datetime] = Field(
        None,
        description="Timestamp when position was closed (NULL = active, NOT NULL = closed)",
    )
    
    # Metadata
    version: int = Field(
        default=1,
        description="Optimistic locking version, incremented on each update",
        ge=1,
    )
    source: Optional[str] = Field(None, description="Update source", max_length=50)
    last_sync_with_bybit: Optional[datetime] = Field(None, description="Last sync with Bybit API")
    bybit_position_data: Optional[Dict[str, Any]] = Field(None, description="Full Bybit position data (JSONB)")

    @field_validator("asset")
    @classmethod
    def validate_asset(cls, v: str) -> str:
        """Validate asset format (e.g., BTCUSDT)."""
        v = v.upper()
        # Keep validation light here; full regex is enforced at API layer.
        if len(v) < 3 or len(v) > 20:
            raise ValueError("Asset must be between 3 and 20 characters")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate trading mode is 'one-way' or 'hedge'."""
        v_lower = v.lower()
        if v_lower not in {"one-way", "hedge"}:
            raise ValueError("Mode must be 'one-way' or 'hedge'")
        return v_lower

    @field_validator("average_entry_price")
    @classmethod
    def validate_average_entry_price(
        cls,
        v: Optional[Decimal],
        info,
    ) -> Optional[Decimal]:
        """average_entry_price must be positive when size != 0."""
        size: Optional[Decimal] = info.data.get("size")
        if size is not None and size != 0 and v is None:
            raise ValueError("average_entry_price must be set when position size is non-zero")
        return v

    @computed_field
    @property
    def unrealized_pnl_pct(self) -> Optional[Decimal]:
        """Computed unrealized PnL percentage.

        (unrealized_pnl / (abs(size) * average_entry_price)) * 100
        """
        if (
            self.unrealized_pnl is None
            or self.average_entry_price is None
            or self.size == 0
        ):
            return None
        try:
            denom = abs(self.size) * self.average_entry_price
            if denom == 0:
                return None
            return (self.unrealized_pnl / denom) * Decimal("100")
        except Exception:
            return None

    @computed_field
    @property
    def is_active(self) -> bool:
        """Check if position is active."""
        return self.closed_at is None
    
    @computed_field
    @property
    def is_closed(self) -> bool:
        """Check if position is closed."""
        return self.closed_at is not None
    
    @computed_field
    @property
    def total_pnl(self) -> Decimal:
        """Total PnL (unrealized + realized)."""
        return self.unrealized_pnl + self.realized_pnl
    
    @computed_field
    @property
    def holding_time_minutes(self) -> Optional[int]:
        """Time position was held in minutes.
        
        For closed positions: from created_at to closed_at
        For active positions: from created_at to now
        """
        if self.is_closed and self.closed_at and self.created_at:
            delta = self.closed_at - self.created_at
            return int(delta.total_seconds() // 60)
        elif self.created_at:
            delta = datetime.utcnow() - self.created_at
            return int(delta.total_seconds() // 60)
        return None
    
    @computed_field
    @property
    def unrealized_pnl_pct(self) -> Optional[Decimal]:
        """Computed unrealized PnL percentage.

        (unrealized_pnl / (abs(size) * average_entry_price)) * 100
        """
        if (
            self.unrealized_pnl is None
            or self.average_entry_price is None
            or self.size == 0
        ):
            return None
        try:
            denom = abs(self.size) * self.average_entry_price
            if denom == 0:
                return None
            return (self.unrealized_pnl / denom) * Decimal("100")
        except Exception:
            return None

    def to_db_dict(self) -> Dict[str, Any]:
        """Serialize to a dict suitable for asyncpg operations."""
        result = {
            "id": str(self.id),
            "asset": self.asset,
            "mode": self.mode,
            "size": str(self.size),
            "average_entry_price": str(self.average_entry_price) if self.average_entry_price is not None else None,
            "current_price": str(self.current_price) if self.current_price is not None else None,
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "long_size": str(self.long_size) if self.long_size is not None else None,
            "short_size": str(self.short_size) if self.short_size is not None else None,
            "long_avg_price": str(self.long_avg_price) if self.long_avg_price is not None else None,
            "short_avg_price": str(self.short_avg_price) if self.short_avg_price is not None else None,
            "leverage": str(self.leverage) if self.leverage is not None else None,
            "position_value": str(self.position_value) if self.position_value is not None else None,
            "liq_price": str(self.liq_price) if self.liq_price is not None else None,
            "bust_price": str(self.bust_price) if self.bust_price is not None else None,
            "take_profit": str(self.take_profit) if self.take_profit is not None else None,
            "stop_loss": str(self.stop_loss) if self.stop_loss is not None else None,
            "cum_realised_pnl": str(self.cum_realised_pnl) if self.cum_realised_pnl is not None else None,
            "cum_unrealised_pnl": str(self.cum_unrealised_pnl) if self.cum_unrealised_pnl is not None else None,
            "total_fees": str(self.total_fees),
            "opening_fees": str(self.opening_fees) if self.opening_fees is not None else None,
            "closing_fees": str(self.closing_fees) if self.closing_fees is not None else None,
            "margin_used": str(self.margin_used) if self.margin_used is not None else None,
            "available_margin": str(self.available_margin) if self.available_margin is not None else None,
            "maintenance_margin": str(self.maintenance_margin) if self.maintenance_margin is not None else None,
            "max_size": str(self.max_size) if self.max_size is not None else None,
            "min_size": str(self.min_size) if self.min_size is not None else None,
            "total_volume_traded": str(self.total_volume_traded),
            "first_entry_price": str(self.first_entry_price) if self.first_entry_price is not None else None,
            "last_entry_price": str(self.last_entry_price) if self.last_entry_price is not None else None,
            "exit_price": str(self.exit_price) if self.exit_price is not None else None,
            "peak_unrealized_pnl": str(self.peak_unrealized_pnl) if self.peak_unrealized_pnl is not None else None,
            "peak_unrealized_pnl_at": self.peak_unrealized_pnl_at,
            "worst_unrealized_pnl": str(self.worst_unrealized_pnl) if self.worst_unrealized_pnl is not None else None,
            "worst_unrealized_pnl_at": self.worst_unrealized_pnl_at,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "closed_at": self.closed_at,
            "version": self.version,
            "source": self.source,
            "last_sync_with_bybit": self.last_sync_with_bybit,
            "bybit_position_data": self.bybit_position_data,
        }
        return result

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> "Position":
        """Create Position from a database row dict, handling type conversion."""
        # UUID conversion - handle both string and UUID objects from asyncpg
        if "id" in data and data["id"] is not None:
            if isinstance(data["id"], str):
                data["id"] = UUID(data["id"])
            elif not isinstance(data["id"], UUID):
                # Convert asyncpg UUID or other UUID-like objects to standard UUID
                data["id"] = UUID(str(data["id"]))

        # Decimal conversion and NULL handling
        # For fields with default values, remove None to let Pydantic use defaults
        decimal_fields = [
            "size",
            "average_entry_price",
            "current_price",
            "unrealized_pnl",
            "realized_pnl",
            "long_size",
            "short_size",
            "long_avg_price",
            "short_avg_price",
            "leverage",
            "position_value",
            "liq_price",
            "bust_price",
            "take_profit",
            "stop_loss",
            "cum_realised_pnl",
            "cum_unrealised_pnl",
            "total_fees",
            "opening_fees",
            "closing_fees",
            "margin_used",
            "available_margin",
            "maintenance_margin",
            "max_size",
            "min_size",
            "total_volume_traded",
            "first_entry_price",
            "last_entry_price",
            "exit_price",
            "peak_unrealized_pnl",
            "worst_unrealized_pnl",
        ]
        
        for field in decimal_fields:
            value = data.get(field)
            if value is not None:
                if not isinstance(value, Decimal):
                    data[field] = Decimal(str(value))
            else:
                # Remove None values for fields with defaults to let Pydantic use defaults
                # Keep None for Optional fields
                if field in ["unrealized_pnl", "realized_pnl", "total_fees", "total_volume_traded"]:
                    data.pop(field, None)  # Remove to use default Decimal("0")
                elif field == "size":
                    # size is required (NOT NULL in DB), but handle None gracefully
                    # Set to 0 if None (should not happen in normal operation)
                    data[field] = Decimal("0")

        # Handle datetime fields with defaults
        if "created_at" in data and data["created_at"] is None:
            data.pop("created_at", None)  # Remove to use default_factory
        
        # Ensure datetime fields are properly converted from asyncpg/timestamp types
        # asyncpg returns datetime objects directly, but handle edge cases
        for datetime_field in [
            "closed_at",
            "last_updated",
            "created_at",
            "peak_unrealized_pnl_at",
            "worst_unrealized_pnl_at",
            "last_sync_with_bybit",
        ]:
            if datetime_field in data and data[datetime_field] is not None:
                if not isinstance(data[datetime_field], datetime):
                    # Try to convert string or other types to datetime
                    try:
                        if isinstance(data[datetime_field], str):
                            # Handle ISO format strings
                            data[datetime_field] = datetime.fromisoformat(
                                data[datetime_field].replace("Z", "+00:00")
                            )
                        else:
                            # For other types, try str conversion first
                            data[datetime_field] = datetime.fromisoformat(
                                str(data[datetime_field]).replace("Z", "+00:00")
                            )
                    except (ValueError, AttributeError):
                        # If conversion fails, keep original value or set to None
                        logger.warning(
                            f"Failed to convert {datetime_field} to datetime",
                            value=data[datetime_field],
                            value_type=type(data[datetime_field]),
                        )
                        # Keep original value - Pydantic will validate

        return cls(**data)


class PositionSnapshot(BaseModel):
    """Historical snapshot of a position state, stored as JSONB in the database."""

    id: UUID = Field(default_factory=uuid4, description="Unique snapshot identifier")
    position_id: UUID = Field(..., description="Reference to positions.id")
    asset: str = Field(..., description="Trading pair", max_length=20)
    mode: str = Field(..., description="Trading mode", max_length=20)
    snapshot_data: Dict[str, Any] = Field(
        ...,
        description="Complete position state at snapshot time (JSONB column)",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Snapshot creation timestamp",
    )

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        v_lower = v.lower()
        if v_lower not in {"one-way", "hedge"}:
            raise ValueError("Mode must be 'one-way' or 'hedge'")
        return v_lower

    def to_db_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "position_id": str(self.position_id),
            "asset": self.asset,
            "mode": self.mode,
            "snapshot_data": self.snapshot_data,
            "created_at": self.created_at,
        }

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> "PositionSnapshot":
        # UUID conversion - handle both string and UUID objects from asyncpg
        for field in ["id", "position_id"]:
            if field in data and data[field] is not None:
                if isinstance(data[field], str):
                    data[field] = UUID(data[field])
                elif not isinstance(data[field], UUID):
                    # Convert asyncpg UUID or other UUID-like objects to standard UUID
                    data[field] = UUID(str(data[field]))
        # Map snapshot_timestamp to created_at if present (for backward compatibility)
        if "snapshot_timestamp" in data and "created_at" not in data:
            data["created_at"] = data.pop("snapshot_timestamp")
        return cls(**data)





