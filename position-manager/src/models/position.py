"""Pydantic models for positions and position snapshots used by Position Manager."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, computed_field


class Position(BaseModel):
    """Position entity mirroring the shared `positions` table plus derived ML features.

    This model intentionally follows the schema defined in `data-model.md`:
    - One logical position per (asset, mode)
    - Supports optimistic locking via `version`
    - Includes fields required for portfolio calculations and ML features
    """

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
    unrealized_pnl: Decimal = Field(
        default=Decimal("0"),
        description="Current unrealized profit/loss",
    )
    realized_pnl: Decimal = Field(
        default=Decimal("0"),
        description="Cumulative realized profit/loss",
    )
    long_size: Optional[Decimal] = Field(
        None,
        description="Long position size (for hedge mode)",
    )
    short_size: Optional[Decimal] = Field(
        None,
        description="Short position size (for hedge mode)",
    )
    version: int = Field(
        default=1,
        description="Optimistic locking version, incremented on each update",
        ge=1,
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp",
    )
    closed_at: Optional[datetime] = Field(
        None,
        description="Timestamp when position was closed (size == 0)",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Position creation timestamp",
    )

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
    def time_held_minutes(self) -> Optional[int]:
        """Approximate time held in minutes based on last_updated vs created_at."""
        try:
            delta = self.last_updated - self.created_at
            return int(delta.total_seconds() // 60)
        except Exception:
            return None

    def to_db_dict(self) -> Dict[str, Any]:
        """Serialize to a dict suitable for asyncpg operations."""
        return {
            "id": str(self.id),
            "asset": self.asset,
            "mode": self.mode,
            "size": str(self.size),
            "average_entry_price": str(self.average_entry_price)
            if self.average_entry_price is not None
            else None,
            "current_price": str(self.current_price) if self.current_price is not None else None,
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "long_size": str(self.long_size) if self.long_size is not None else None,
            "short_size": str(self.short_size) if self.short_size is not None else None,
            "version": self.version,
            "last_updated": self.last_updated,
            "closed_at": self.closed_at,
            "created_at": self.created_at,
        }

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
        for field in [
            "size",
            "average_entry_price",
            "current_price",
            "unrealized_pnl",
            "realized_pnl",
            "long_size",
            "short_size",
        ]:
            value = data.get(field)
            if value is not None:
                if not isinstance(value, Decimal):
                    data[field] = Decimal(str(value))
            else:
                # Remove None values for fields with defaults to let Pydantic use defaults
                # Keep None for Optional fields (long_size, short_size, average_entry_price, current_price)
                if field in ["unrealized_pnl", "realized_pnl"]:
                    data.pop(field, None)  # Remove to use default Decimal("0")

        # Handle datetime fields with defaults
        if "created_at" in data and data["created_at"] is None:
            data.pop("created_at", None)  # Remove to use default_factory

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


class ClosedPosition(BaseModel):
    """Historical record of a closed position from closed_positions table."""

    id: UUID = Field(default_factory=uuid4, description="Unique closed position identifier")
    original_position_id: UUID = Field(..., description="ID from positions table at closure")
    asset: str = Field(..., description="Trading pair symbol", max_length=20)
    mode: str = Field(..., description="Trading mode", max_length=20)
    
    # Position state at closure
    final_size: Decimal = Field(default=Decimal("0"), description="Position size at closure (always 0)")
    average_entry_price: Optional[Decimal] = Field(None, description="Average entry price", gt=0)
    exit_price: Optional[Decimal] = Field(None, description="Price at closure", gt=0)
    current_price: Optional[Decimal] = Field(None, description="Current price at closure (same as exit_price)")
    
    # PnL at closure
    realized_pnl: Decimal = Field(default=Decimal("0"), description="Realized profit/loss at closure")
    unrealized_pnl_at_close: Decimal = Field(default=Decimal("0"), description="Unrealized PnL at closure")
    
    # Hedge mode fields
    long_size: Optional[Decimal] = Field(None, description="Long position size (for hedge mode)")
    short_size: Optional[Decimal] = Field(None, description="Short position size (for hedge mode)")
    long_avg_price: Optional[Decimal] = Field(None, description="Long average price (for hedge mode)", gt=0)
    short_avg_price: Optional[Decimal] = Field(None, description="Short average price (for hedge mode)", gt=0)
    
    # Fees
    total_fees: Optional[Decimal] = Field(None, description="Total fees for this position")
    
    # Metadata
    opened_at: datetime = Field(..., description="When position was opened (created_at from original position)")
    closed_at: datetime = Field(..., description="When position was closed")
    version: int = Field(..., description="Version number at closure", ge=1)

    @field_validator("asset")
    @classmethod
    def validate_asset(cls, v: str) -> str:
        """Validate asset format (e.g., BTCUSDT)."""
        v = v.upper()
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

    @computed_field
    @property
    def holding_time_minutes(self) -> Optional[int]:
        """Time position was held in minutes."""
        try:
            delta = self.closed_at - self.opened_at
            return int(delta.total_seconds() // 60)
        except Exception:
            return None

    @computed_field
    @property
    def total_pnl(self) -> Decimal:
        """Total PnL for a closed position.
        
        For a fully closed position, realized_pnl already includes all PnL
        (since unrealized_pnl_at_close became realized when position was closed).
        Therefore, total_pnl = realized_pnl.
        """
        return self.realized_pnl

    def to_db_dict(self) -> Dict[str, Any]:
        """Serialize to a dict suitable for asyncpg operations."""
        return {
            "id": str(self.id),
            "original_position_id": str(self.original_position_id),
            "asset": self.asset,
            "mode": self.mode,
            "final_size": str(self.final_size),
            "average_entry_price": str(self.average_entry_price) if self.average_entry_price is not None else None,
            "exit_price": str(self.exit_price) if self.exit_price is not None else None,
            "current_price": str(self.current_price) if self.current_price is not None else None,
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl_at_close": str(self.unrealized_pnl_at_close),
            "long_size": str(self.long_size) if self.long_size is not None else None,
            "short_size": str(self.short_size) if self.short_size is not None else None,
            "long_avg_price": str(self.long_avg_price) if self.long_avg_price is not None else None,
            "short_avg_price": str(self.short_avg_price) if self.short_avg_price is not None else None,
            "total_fees": str(self.total_fees) if self.total_fees is not None else None,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "version": self.version,
        }

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> "ClosedPosition":
        """Create ClosedPosition from a database row dict, handling type conversion."""
        # UUID conversion
        for field in ["id", "original_position_id"]:
            if field in data and data[field] is not None:
                if isinstance(data[field], str):
                    data[field] = UUID(data[field])
                elif not isinstance(data[field], UUID):
                    data[field] = UUID(str(data[field]))

        # Decimal conversion
        for field in [
            "final_size",
            "average_entry_price",
            "exit_price",
            "current_price",
            "realized_pnl",
            "unrealized_pnl_at_close",
            "long_size",
            "short_size",
            "long_avg_price",
            "short_avg_price",
            "total_fees",
        ]:
            value = data.get(field)
            if value is not None:
                if not isinstance(value, Decimal):
                    data[field] = Decimal(str(value))
            else:
                # Remove None for fields with defaults to let Pydantic use defaults
                if field in ["realized_pnl", "unrealized_pnl_at_close", "final_size"]:
                    data.pop(field, None)

        return cls(**data)



