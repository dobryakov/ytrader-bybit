"""
Order Execution Event data model.

Represents an enriched event from the order manager microservice containing
details about executed trades for model training purposes.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator


class MarketConditions(BaseModel):
    """Market conditions at execution time."""

    spread: float = Field(..., description="Bid-ask spread", ge=0)
    volume_24h: float = Field(..., description="24-hour trading volume", ge=0)
    volatility: float = Field(..., description="Current volatility measure", ge=0)


class PerformanceMetrics(BaseModel):
    """Trade performance metrics."""

    slippage: float = Field(..., description="Price difference (execution - signal)")
    slippage_percent: float = Field(..., description="Slippage as percentage")
    realized_pnl: Optional[float] = Field(default=None, description="Realized profit/loss (if closed)")
    return_percent: Optional[float] = Field(default=None, description="Return percentage")


class OrderExecutionEvent(BaseModel):
    """
    Order execution event data model.

    Represents an enriched event from the order manager microservice containing
    details about executed trades. Consumed from RabbitMQ for training purposes.
    """

    event_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique event identifier")
    order_id: str = Field(..., description="Order identifier from order manager")
    signal_id: str = Field(..., description="Original trading signal identifier")
    strategy_id: str = Field(..., description="Trading strategy identifier")
    asset: str = Field(..., description="Trading pair (e.g., 'BTCUSDT')")
    side: str = Field(..., description="Order side: 'buy' or 'sell'")
    execution_price: float = Field(..., description="Actual execution price", gt=0)
    execution_quantity: float = Field(..., description="Executed quantity", gt=0)
    execution_fees: float = Field(..., description="Total fees paid", ge=0)
    executed_at: datetime = Field(..., description="Execution timestamp")
    signal_price: float = Field(..., description="Original signal price (for slippage calculation)", gt=0)
    signal_timestamp: datetime = Field(..., description="Original signal timestamp")
    market_conditions: MarketConditions = Field(
        ..., description="Market data at execution time"
    )
    performance: PerformanceMetrics = Field(..., description="Trade performance metrics")
    trace_id: Optional[str] = Field(default=None, description="Trace ID for request flow tracking")

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: str) -> str:
        """Validate side is 'buy' or 'sell'."""
        if v.lower() not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        return v.lower()

    @field_validator("asset")
    @classmethod
    def validate_asset(cls, v: str) -> str:
        """Validate asset is a valid trading pair format."""
        if not v or len(v) < 3:
            raise ValueError("asset must be a valid trading pair (e.g., 'BTCUSDT')")
        return v.upper()

    @field_validator("executed_at")
    @classmethod
    def validate_executed_at(cls, v: datetime) -> datetime:
        """Validate executed_at is not in the future."""
        if v > datetime.utcnow():
            raise ValueError("executed_at cannot be in the future")
        return v

    @field_validator("signal_timestamp")
    @classmethod
    def validate_signal_timestamp(cls, v: datetime) -> datetime:
        """Validate signal_timestamp is not in the future."""
        if v > datetime.utcnow():
            raise ValueError("signal_timestamp cannot be in the future")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert execution event to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the execution event
        """
        return {
            "event_id": self.event_id,
            "order_id": self.order_id,
            "signal_id": self.signal_id,
            "strategy_id": self.strategy_id,
            "asset": self.asset,
            "side": self.side,
            "execution_price": self.execution_price,
            "execution_quantity": self.execution_quantity,
            "execution_fees": self.execution_fees,
            "executed_at": self.executed_at.isoformat() + "Z",
            "signal_price": self.signal_price,
            "signal_timestamp": self.signal_timestamp.isoformat() + "Z",
            "market_conditions": {
                "spread": self.market_conditions.spread,
                "volume_24h": self.market_conditions.volume_24h,
                "volatility": self.market_conditions.volatility,
            },
            "performance": {
                "slippage": self.performance.slippage,
                "slippage_percent": self.performance.slippage_percent,
                "realized_pnl": self.performance.realized_pnl,
                "return_percent": self.performance.return_percent,
            },
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderExecutionEvent":
        """
        Create OrderExecutionEvent from dictionary.

        Args:
            data: Dictionary containing execution event data

        Returns:
            OrderExecutionEvent instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Parse datetime strings if present
        if isinstance(data.get("executed_at"), str):
            data["executed_at"] = datetime.fromisoformat(data["executed_at"].replace("Z", "+00:00"))
        if isinstance(data.get("signal_timestamp"), str):
            data["signal_timestamp"] = datetime.fromisoformat(data["signal_timestamp"].replace("Z", "+00:00"))

        return cls(**data)

