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
        # UUID conversion
        if "id" in data and isinstance(data["id"], str):
            data["id"] = UUID(data["id"])

        # Decimal conversion
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
            if value is not None and not isinstance(value, Decimal):
                data[field] = Decimal(str(value))

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
        for field in ["id", "position_id"]:
            if field in data and isinstance(data[field], str):
                data[field] = UUID(data[field])
        return cls(**data)



