"""Trading signal model for in-memory processing of signals from RabbitMQ."""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class MarketDataSnapshot(BaseModel):
    """Market data snapshot at signal generation time."""

    price: Decimal = Field(..., description="Current market price", gt=0)
    spread: Optional[Decimal] = Field(None, description="Bid-ask spread (percentage)", ge=0)
    volume_24h: Optional[Decimal] = Field(None, description="24-hour trading volume", ge=0)
    volatility: Optional[Decimal] = Field(None, description="Market volatility metric", ge=0)
    orderbook_depth: Optional[Dict[str, Any]] = Field(None, description="Orderbook depth data")
    technical_indicators: Optional[Dict[str, Any]] = Field(None, description="Technical indicator values")


class TradingSignal(BaseModel):
    """Trading signal received from model service via RabbitMQ.

    This is an in-memory structure for processing signals. Signals are consumed
    from RabbitMQ queue and may not be persisted (for audit trail, optional
    persistence can be added).
    """

    signal_id: UUID = Field(..., description="Unique signal identifier")
    signal_type: str = Field(..., description="Trading signal type: 'buy' or 'sell'", max_length=10)
    asset: str = Field(..., description="Trading pair (e.g., 'BTCUSDT')", max_length=20)
    amount: Decimal = Field(..., description="Amount in quote currency (USDT)", gt=0)
    confidence: Decimal = Field(..., description="Confidence score (0.0-1.0)", ge=0, le=1)
    timestamp: datetime = Field(..., description="Signal generation timestamp")
    strategy_id: str = Field(..., description="Trading strategy identifier", max_length=100)
    model_version: Optional[str] = Field(None, description="Model version (null for warm-up)", max_length=100)
    is_warmup: bool = Field(default=False, description="Whether warm-up signal")
    market_data_snapshot: MarketDataSnapshot = Field(..., description="Market data at signal time")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    trace_id: Optional[str] = Field(None, description="Trace ID for request tracking", max_length=100)

    @field_validator("signal_type")
    @classmethod
    def validate_signal_type(cls, v: str) -> str:
        """Validate signal type is 'buy' or 'sell'."""
        v_lower = v.lower()
        if v_lower not in {"buy", "sell"}:
            raise ValueError("Signal type must be 'buy' or 'sell'")
        return v_lower

    @field_validator("asset")
    @classmethod
    def validate_asset(cls, v: str) -> str:
        """Validate asset format (uppercase, e.g., 'BTCUSDT')."""
        v_upper = v.upper()
        # Basic validation - should match Bybit trading pair format
        if len(v_upper) < 4:
            raise ValueError("Asset must be a valid trading pair (e.g., 'BTCUSDT')")
        return v_upper

    def get_target_timestamp(self) -> Optional[datetime]:
        """Extract target timestamp from signal metadata.
        
        The target timestamp can be:
        - Directly in metadata as 'target_timestamp' (ISO string)
        - Computed from 'prediction_horizon_seconds' + signal timestamp
        
        Returns:
            Target timestamp if available, None otherwise
        """
        if not self.metadata:
            return None
        
        # Try to get target_timestamp directly
        target_ts = self.metadata.get("target_timestamp")
        if target_ts:
            if isinstance(target_ts, str):
                try:
                    # Parse ISO format string, handling 'Z' suffix
                    parsed_ts = datetime.fromisoformat(target_ts.replace("Z", "+00:00"))
                    return parsed_ts
                except (ValueError, AttributeError):
                    pass
            elif isinstance(target_ts, datetime):
                return target_ts
        
        # Try to compute from prediction_horizon_seconds
        horizon_seconds = self.metadata.get("prediction_horizon_seconds")
        if horizon_seconds is not None:
            try:
                # Handle both int and string types
                if isinstance(horizon_seconds, str):
                    horizon = int(horizon_seconds)
                elif isinstance(horizon_seconds, (int, float)):
                    horizon = int(horizon_seconds)
                else:
                    horizon = None
                
                if horizon and horizon > 0:
                    from datetime import timedelta
                    # Ensure timestamp is timezone-aware for calculation
                    timestamp = self.timestamp
                    if timestamp.tzinfo is None:
                        # If naive, assume UTC
                        from datetime import timezone
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    result = timestamp + timedelta(seconds=horizon)
                    return result
            except (ValueError, TypeError, AttributeError):
                pass
        
        return None

    def to_dict(self) -> dict:
        """Convert signal to dictionary for serialization."""
        return {
            "signal_id": str(self.signal_id),
            "signal_type": self.signal_type,
            "asset": self.asset,
            "amount": str(self.amount),
            "confidence": str(self.confidence),
            "timestamp": self.timestamp.isoformat(),
            "strategy_id": self.strategy_id,
            "model_version": self.model_version,
            "is_warmup": self.is_warmup,
            "market_data_snapshot": self.market_data_snapshot.model_dump(),
            "metadata": self.metadata,
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TradingSignal":
        """Create signal from dictionary (e.g., from RabbitMQ message)."""
        # Convert string UUID to UUID object
        if isinstance(data.get("signal_id"), str):
            data["signal_id"] = UUID(data["signal_id"])

        # Convert string decimals to Decimal
        for field in ["amount", "confidence"]:
            if field in data and isinstance(data[field], (str, float, int)):
                data[field] = Decimal(str(data[field]))

        # Convert timestamp string to datetime
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))

        # Convert market_data_snapshot dict to MarketDataSnapshot object
        if isinstance(data.get("market_data_snapshot"), dict):
            market_data = data["market_data_snapshot"]
            # Convert price to Decimal if needed
            if "price" in market_data and isinstance(market_data["price"], (str, float)):
                market_data["price"] = Decimal(str(market_data["price"]))
            # Convert other decimal fields
            for field in ["spread", "volume_24h", "volatility"]:
                if field in market_data and market_data[field] is not None:
                    if isinstance(market_data[field], (str, float)):
                        market_data[field] = Decimal(str(market_data[field]))
            data["market_data_snapshot"] = MarketDataSnapshot(**market_data)

        return cls(**data)

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            UUID: str,
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }

