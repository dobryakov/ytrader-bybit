"""
Order and Position State data model.

Represents a read-only snapshot of current orders and positions from the shared database.
Used as input for intelligent signal generation.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field, field_validator


class OrderState(BaseModel):
    """Represents a single open order."""

    id: str = Field(..., description="Order UUID")
    order_id: str = Field(..., description="Order identifier from order manager")
    signal_id: str = Field(..., description="Original trading signal identifier")
    asset: str = Field(..., description="Trading pair symbol (e.g., 'BTCUSDT')")
    side: str = Field(..., description="Order side: 'Buy' or 'Sell'")
    order_type: str = Field(..., description="Order type: 'Market' or 'Limit'")
    quantity: Decimal = Field(..., description="Order quantity", gt=0)
    price: Optional[Decimal] = Field(default=None, description="Order price (for limit orders)")
    status: str = Field(..., description="Order status")
    filled_quantity: Decimal = Field(default=Decimal("0"), description="Filled quantity", ge=0)
    average_price: Optional[Decimal] = Field(default=None, description="Average execution price")
    fees: Optional[Decimal] = Field(default=None, description="Fees paid")
    created_at: datetime = Field(..., description="Order creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    executed_at: Optional[datetime] = Field(default=None, description="Execution timestamp")

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: str) -> str:
        """Validate side is 'Buy' or 'Sell'."""
        if v.upper() not in ("BUY", "SELL"):
            raise ValueError("side must be 'Buy' or 'Sell'")
        return v.upper()

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status is a valid order status."""
        valid_statuses = {"pending", "partially_filled", "filled", "cancelled", "rejected", "dry_run"}
        if v.lower() not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")
        return v.lower()


class PositionState(BaseModel):
    """Represents a single open position."""

    id: str = Field(..., description="Position UUID")
    asset: str = Field(..., description="Trading pair symbol (e.g., 'BTCUSDT')")
    size: Decimal = Field(..., description="Position size (positive for long, negative for short)")
    average_entry_price: Optional[Decimal] = Field(default=None, description="Average entry price")
    unrealized_pnl: Decimal = Field(default=Decimal("0"), description="Unrealized profit/loss")
    realized_pnl: Decimal = Field(default=Decimal("0"), description="Realized profit/loss")
    mode: str = Field(default="one-way", description="Position mode: 'one-way' or 'hedge'")
    long_size: Optional[Decimal] = Field(default=None, description="Long position size (hedge mode)")
    short_size: Optional[Decimal] = Field(default=None, description="Short position size (hedge mode)")
    long_avg_price: Optional[Decimal] = Field(default=None, description="Long average price (hedge mode)")
    short_avg_price: Optional[Decimal] = Field(default=None, description="Short average price (hedge mode)")
    last_updated: datetime = Field(..., description="Last update timestamp")
    last_snapshot_at: Optional[datetime] = Field(default=None, description="Last snapshot timestamp")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate mode is 'one-way' or 'hedge'."""
        if v.lower() not in ("one-way", "hedge"):
            raise ValueError("mode must be 'one-way' or 'hedge'")
        return v.lower()


class OrderPositionState(BaseModel):
    """
    Complete order and position state snapshot.

    Represents current snapshot of open orders and positions for a strategy.
    Used as input for intelligent signal generation.
    """

    strategy_id: str = Field(..., description="Trading strategy identifier")
    orders: List[OrderState] = Field(default_factory=list, description="List of open orders")
    positions: List[PositionState] = Field(default_factory=list, description="List of open positions")
    snapshot_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Snapshot timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

    def get_total_exposure(self, asset: Optional[str] = None) -> Decimal:
        """
        Calculate total exposure for an asset or all assets.

        Args:
            asset: Asset to calculate exposure for (None for all assets)

        Returns:
            Total exposure (sum of position sizes)
        """
        positions = self.positions if asset is None else [p for p in self.positions if p.asset == asset]
        return sum(p.size for p in positions)

    def get_unrealized_pnl(self, asset: Optional[str] = None) -> Decimal:
        """
        Calculate total unrealized P&L for an asset or all assets.

        Args:
            asset: Asset to calculate P&L for (None for all assets)

        Returns:
            Total unrealized P&L
        """
        positions = self.positions if asset is None else [p for p in self.positions if p.asset == asset]
        return sum(p.unrealized_pnl for p in positions)

    def get_open_orders_count(self, asset: Optional[str] = None) -> int:
        """
        Get count of open orders for an asset or all assets.

        Args:
            asset: Asset to count orders for (None for all assets)

        Returns:
            Number of open orders
        """
        orders = self.orders if asset is None else [o for o in self.orders if o.asset == asset]
        # Count only pending or partially filled orders
        return len([o for o in orders if o.status in ("pending", "partially_filled")])

    def has_position(self, asset: str) -> bool:
        """
        Check if there is an open position for an asset.

        Args:
            asset: Trading pair symbol

        Returns:
            True if position exists, False otherwise
        """
        return any(p.asset == asset and p.size != 0 for p in self.positions)

    def get_position(self, asset: str) -> Optional[PositionState]:
        """
        Get position for a specific asset.

        Args:
            asset: Trading pair symbol

        Returns:
            PositionState or None if not found
        """
        for position in self.positions:
            if position.asset == asset:
                return position
        return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the state
        """
        return {
            "strategy_id": self.strategy_id,
            "orders": [
                {
                    "id": order.id,
                    "order_id": order.order_id,
                    "signal_id": order.signal_id,
                    "asset": order.asset,
                    "side": order.side,
                    "order_type": order.order_type,
                    "quantity": str(order.quantity),
                    "price": str(order.price) if order.price else None,
                    "status": order.status,
                    "filled_quantity": str(order.filled_quantity),
                    "average_price": str(order.average_price) if order.average_price else None,
                    "fees": str(order.fees) if order.fees else None,
                    "created_at": order.created_at.isoformat() + "Z",
                    "updated_at": order.updated_at.isoformat() + "Z",
                    "executed_at": order.executed_at.isoformat() + "Z" if order.executed_at else None,
                }
                for order in self.orders
            ],
            "positions": [
                {
                    "id": position.id,
                    "asset": position.asset,
                    "size": str(position.size),
                    "average_entry_price": str(position.average_entry_price) if position.average_entry_price else None,
                    "unrealized_pnl": str(position.unrealized_pnl),
                    "realized_pnl": str(position.realized_pnl),
                    "mode": position.mode,
                    "long_size": str(position.long_size) if position.long_size else None,
                    "short_size": str(position.short_size) if position.short_size else None,
                    "long_avg_price": str(position.long_avg_price) if position.long_avg_price else None,
                    "short_avg_price": str(position.short_avg_price) if position.short_avg_price else None,
                    "last_updated": position.last_updated.isoformat() + "Z",
                    "last_snapshot_at": position.last_snapshot_at.isoformat() + "Z" if position.last_snapshot_at else None,
                }
                for position in self.positions
            ],
            "snapshot_timestamp": self.snapshot_timestamp.isoformat() + "Z",
            "metadata": self.metadata,
        }

