"""
Signal validation service.

Validates trading signals for required fields, value ranges, and format compliance.
"""

from typing import List, Tuple
from datetime import datetime, timedelta

from ..models.signal import TradingSignal
from ..config.logging import get_logger
from ..config.exceptions import ModelServiceError

logger = get_logger(__name__)


class SignalValidationError(ModelServiceError):
    """Error during signal validation."""

    pass


class SignalValidator:
    """Validates trading signals before publishing."""

    def __init__(
        self,
        min_amount: float = 10.0,
        max_amount: float = 100000.0,
        max_timestamp_age_seconds: int = 300,
    ):
        """
        Initialize signal validator.

        Args:
            min_amount: Minimum allowed order amount
            max_amount: Maximum allowed order amount
            max_timestamp_age_seconds: Maximum age of signal timestamp in seconds
        """
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.max_timestamp_age_seconds = max_timestamp_age_seconds

    def validate(self, signal: TradingSignal) -> Tuple[bool, List[str]]:
        """
        Validate a trading signal.

        Args:
            signal: TradingSignal to validate

        Returns:
            Tuple of (is_valid: bool, errors: List[str])
        """
        errors = []

        # Validate signal_type
        if signal.signal_type not in ("buy", "sell"):
            errors.append(f"Invalid signal_type: {signal.signal_type}. Must be 'buy' or 'sell'")

        # Validate asset
        if not signal.asset or len(signal.asset) < 3:
            errors.append(f"Invalid asset: {signal.asset}. Must be a valid trading pair")

        # Validate amount
        if signal.amount <= 0:
            errors.append(f"Invalid amount: {signal.amount}. Must be positive")
        elif signal.amount < self.min_amount:
            errors.append(f"Amount {signal.amount} below minimum {self.min_amount}")
        elif signal.amount > self.max_amount:
            errors.append(f"Amount {signal.amount} above maximum {self.max_amount}")

        # Validate confidence
        if not 0.0 <= signal.confidence <= 1.0:
            errors.append(f"Invalid confidence: {signal.confidence}. Must be between 0.0 and 1.0")

        # Validate timestamp
        now = datetime.utcnow()
        if signal.timestamp > now:
            errors.append(f"Timestamp {signal.timestamp} is in the future")
        elif (now - signal.timestamp).total_seconds() > self.max_timestamp_age_seconds:
            errors.append(
                f"Timestamp {signal.timestamp} is too old (>{self.max_timestamp_age_seconds}s)"
            )

        # Validate strategy_id
        if not signal.strategy_id or not signal.strategy_id.strip():
            errors.append("strategy_id is required and cannot be empty")

        # Validate market_data_snapshot
        if not signal.market_data_snapshot:
            errors.append("market_data_snapshot is required")
        else:
            snapshot = signal.market_data_snapshot
            if snapshot.price <= 0:
                errors.append(f"Invalid market_data_snapshot.price: {snapshot.price}. Must be positive")
            if snapshot.spread < 0:
                errors.append(f"Invalid market_data_snapshot.spread: {snapshot.spread}. Must be non-negative")
            if snapshot.volume_24h < 0:
                errors.append(
                    f"Invalid market_data_snapshot.volume_24h: {snapshot.volume_24h}. Must be non-negative"
                )
            if snapshot.volatility < 0:
                errors.append(
                    f"Invalid market_data_snapshot.volatility: {snapshot.volatility}. Must be non-negative"
                )

        # Validate signal_id
        if not signal.signal_id:
            errors.append("signal_id is required")

        is_valid = len(errors) == 0

        if not is_valid:
            logger.warning(
                "Signal validation failed",
                signal_id=signal.signal_id,
                errors=errors,
                signal_type=signal.signal_type,
                asset=signal.asset,
            )

        return is_valid, errors

    def validate_and_raise(self, signal: TradingSignal) -> None:
        """
        Validate a trading signal and raise exception if invalid.

        Args:
            signal: TradingSignal to validate

        Raises:
            SignalValidationError: If validation fails
        """
        is_valid, errors = self.validate(signal)
        if not is_valid:
            error_message = f"Signal validation failed: {', '.join(errors)}"
            raise SignalValidationError(error_message)


# Global signal validator instance
signal_validator = SignalValidator()

