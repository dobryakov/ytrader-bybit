"""Position model representing current trading positions."""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class Position(BaseModel):
    """Position entity representing current trading position for an asset.

    Supports both one-way and hedge-mode trading. Position state is stored
    and updated based on order executions.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique position identifier")
    asset: str = Field(..., description="Trading pair symbol (e.g., 'BTCUSDT')", max_length=20)
    size: Decimal = Field(
        ...,
        description="Position size (positive = long, negative = short, zero = no position)",
    )
    average_entry_price: Optional[Decimal] = Field(None, description="Average entry price for position", gt=0)
    unrealized_pnl: Optional[Decimal] = Field(None, description="Unrealized profit and loss")
    realized_pnl: Optional[Decimal] = Field(None, description="Realized profit and loss (cumulative)")
    mode: str = Field(
        default="one-way",
        description="Trading mode: 'one-way' or 'hedge'",
        max_length=20,
    )
    long_size: Optional[Decimal] = Field(None, description="Long position size (for hedge-mode, separate from short)")
    short_size: Optional[Decimal] = Field(None, description="Short position size (for hedge-mode, separate from long)")
    long_avg_price: Optional[Decimal] = Field(None, description="Average entry price for long position (hedge-mode)", gt=0)
    short_avg_price: Optional[Decimal] = Field(None, description="Average entry price for short position (hedge-mode)", gt=0)
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    last_snapshot_at: Optional[datetime] = Field(None, description="Last snapshot timestamp")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate trading mode is 'one-way' or 'hedge'."""
        v_lower = v.lower()
        if v_lower not in {"one-way", "hedge"}:
            raise ValueError("Mode must be 'one-way' or 'hedge'")
        return v_lower

    @field_validator("size", "long_size", "short_size")
    @classmethod
    def validate_size(cls, v: Optional[Decimal], info) -> Optional[Decimal]:
        """Validate position size based on mode."""
        mode = info.data.get("mode", "one-way")
        if mode == "one-way":
            # In one-way mode, size represents net position
            # Can be positive (long), negative (short), or zero
            pass
        elif mode == "hedge":
            # In hedge-mode, long_size and short_size track separate positions
            # Size may be 0 if long and short positions offset
            pass
        return v

    def to_dict(self) -> dict:
        """Convert position to dictionary for database operations."""
        return {
            "id": str(self.id),
            "asset": self.asset,
            "size": str(self.size),
            "average_entry_price": str(self.average_entry_price) if self.average_entry_price is not None else None,
            "unrealized_pnl": str(self.unrealized_pnl) if self.unrealized_pnl is not None else None,
            "realized_pnl": str(self.realized_pnl) if self.realized_pnl is not None else None,
            "mode": self.mode,
            "long_size": str(self.long_size) if self.long_size is not None else None,
            "short_size": str(self.short_size) if self.short_size is not None else None,
            "long_avg_price": str(self.long_avg_price) if self.long_avg_price is not None else None,
            "short_avg_price": str(self.short_avg_price) if self.short_avg_price is not None else None,
            "last_updated": self.last_updated,
            "last_snapshot_at": self.last_snapshot_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        """Create position from dictionary (e.g., from database)."""
        # Convert string UUID to UUID object
        if isinstance(data.get("id"), str):
            data["id"] = UUID(data["id"])

        # Convert string decimals to Decimal
        for field in [
            "size",
            "average_entry_price",
            "unrealized_pnl",
            "realized_pnl",
            "long_size",
            "short_size",
            "long_avg_price",
            "short_avg_price",
        ]:
            if field in data and data[field] is not None:
                if isinstance(data[field], str):
                    data[field] = Decimal(data[field])

        return cls(**data)

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            UUID: str,
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }


class PositionSnapshot(BaseModel):
    """Position snapshot for historical tracking and validation."""

    id: UUID = Field(default_factory=uuid4, description="Unique snapshot identifier")
    position_id: UUID = Field(..., description="Reference to positions.id")
    asset: str = Field(..., description="Trading pair", max_length=20)
    size: Decimal = Field(..., description="Position size at snapshot time")
    average_entry_price: Optional[Decimal] = Field(None, description="Average entry price at snapshot time", gt=0)
    unrealized_pnl: Optional[Decimal] = Field(None, description="Unrealized PnL at snapshot time")
    realized_pnl: Optional[Decimal] = Field(None, description="Realized PnL at snapshot time")
    mode: str = Field(..., description="Trading mode", max_length=20)
    long_size: Optional[Decimal] = Field(None, description="Long size (hedge-mode)")
    short_size: Optional[Decimal] = Field(None, description="Short size (hedge-mode)")
    snapshot_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When snapshot was created")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate trading mode is 'one-way' or 'hedge'."""
        v_lower = v.lower()
        if v_lower not in {"one-way", "hedge"}:
            raise ValueError("Mode must be 'one-way' or 'hedge'")
        return v_lower

    def to_dict(self) -> dict:
        """Convert snapshot to dictionary for database operations."""
        return {
            "id": str(self.id),
            "position_id": str(self.position_id),
            "asset": self.asset,
            "size": str(self.size),
            "average_entry_price": str(self.average_entry_price) if self.average_entry_price is not None else None,
            "unrealized_pnl": str(self.unrealized_pnl) if self.unrealized_pnl is not None else None,
            "realized_pnl": str(self.realized_pnl) if self.realized_pnl is not None else None,
            "mode": self.mode,
            "long_size": str(self.long_size) if self.long_size is not None else None,
            "short_size": str(self.short_size) if self.short_size is not None else None,
            "snapshot_timestamp": self.snapshot_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PositionSnapshot":
        """Create snapshot from dictionary (e.g., from database)."""
        # Convert string UUIDs to UUID objects
        for field in ["id", "position_id"]:
            if field in data and isinstance(data[field], str):
                data[field] = UUID(data[field])

        # Convert string decimals to Decimal
        for field in [
            "size",
            "average_entry_price",
            "unrealized_pnl",
            "realized_pnl",
            "long_size",
            "short_size",
        ]:
            if field in data and data[field] is not None:
                if isinstance(data[field], str):
                    data[field] = Decimal(data[field])

        return cls(**data)

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            UUID: str,
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }

