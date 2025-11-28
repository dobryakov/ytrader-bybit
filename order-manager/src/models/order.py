"""Order model representing trading orders placed on Bybit exchange."""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class Order(BaseModel):
    """Order entity representing a trading order on Bybit exchange."""

    id: UUID = Field(default_factory=uuid4, description="Unique order identifier (internal)")
    order_id: str = Field(..., description="Bybit order ID (returned by exchange)", max_length=100)
    signal_id: UUID = Field(..., description="Trading signal identifier that created this order")
    asset: str = Field(..., description="Trading pair symbol (e.g., 'BTCUSDT')", max_length=20)
    side: str = Field(..., description="Order side: 'Buy' or 'Sell'", max_length=10)
    order_type: str = Field(..., description="Order type: 'Market' or 'Limit'", max_length=20)
    quantity: Decimal = Field(..., description="Order quantity in base currency", gt=0)
    price: Optional[Decimal] = Field(None, description="Limit order price (NULL for market orders)")
    status: str = Field(
        default="pending",
        description="Order status: 'pending', 'partially_filled', 'filled', 'cancelled', 'rejected', 'dry_run'",
        max_length=50,
    )
    filled_quantity: Decimal = Field(default=Decimal("0"), description="Quantity that has been filled", ge=0)
    average_price: Optional[Decimal] = Field(None, description="Average execution price (if partially or fully filled)")
    fees: Optional[Decimal] = Field(None, description="Total fees paid for this order")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When order was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    executed_at: Optional[datetime] = Field(None, description="When order was fully executed (filled)")
    trace_id: Optional[str] = Field(None, description="Trace ID for request flow tracking", max_length=100)
    is_dry_run: bool = Field(default=False, description="Whether order was created in dry-run mode")
    rejection_reason: Optional[str] = Field(None, description="Reason for order rejection (for rejected orders)")

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: str) -> str:
        """Validate order side is 'Buy' or 'SELL' (database constraint format)."""
        v_normalized = v.strip()
        if v_normalized.upper() == "BUY":
            return "Buy"
        elif v_normalized.upper() == "SELL":
            return "SELL"
        else:
            raise ValueError("Side must be 'Buy' or 'SELL'")

    @field_validator("order_type")
    @classmethod
    def validate_order_type(cls, v: str) -> str:
        """Validate order type is 'Market' or 'Limit'."""
        v_title = v.title()
        if v_title not in {"Market", "Limit"}:
            raise ValueError("Order type must be 'Market' or 'Limit'")
        return v_title

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate order status."""
        valid_statuses = {"pending", "partially_filled", "filled", "cancelled", "rejected", "dry_run"}
        v_lower = v.lower()
        if v_lower not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        return v_lower

    @field_validator("price")
    @classmethod
    def validate_price(cls, v: Optional[Decimal], info) -> Optional[Decimal]:
        """Validate price is set for limit orders."""
        if info.data.get("order_type") == "Limit" and v is None:
            raise ValueError("Price must be set for limit orders")
        if info.data.get("order_type") == "Market" and v is not None:
            # Allow price for market orders (may be set for reference)
            pass
        return v

    @field_validator("filled_quantity")
    @classmethod
    def validate_filled_quantity(cls, v: Decimal, info) -> Decimal:
        """Validate filled quantity does not exceed order quantity."""
        quantity = info.data.get("quantity")
        if quantity and v > quantity:
            raise ValueError("Filled quantity cannot exceed order quantity")
        return v

    def to_dict(self) -> dict:
        """Convert order to dictionary for database operations."""
        return {
            "id": str(self.id),
            "order_id": self.order_id,
            "signal_id": str(self.signal_id),
            "asset": self.asset,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": str(self.quantity),
            "price": str(self.price) if self.price is not None else None,
            "status": self.status,
            "filled_quantity": str(self.filled_quantity),
            "average_price": str(self.average_price) if self.average_price is not None else None,
            "fees": str(self.fees) if self.fees is not None else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "executed_at": self.executed_at,
            "trace_id": self.trace_id,
            "is_dry_run": self.is_dry_run,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Order":
        """Create order from dictionary (e.g., from database)."""
        # Convert string UUIDs to UUID objects
        if isinstance(data.get("id"), str):
            data["id"] = UUID(data["id"])
        if isinstance(data.get("signal_id"), str):
            data["signal_id"] = UUID(data["signal_id"])

        # Convert string decimals to Decimal
        for field in ["quantity", "filled_quantity", "price", "average_price", "fees"]:
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

