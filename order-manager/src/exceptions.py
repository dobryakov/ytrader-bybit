"""Error handling and exception classes."""

from typing import Optional


class OrderManagerError(Exception):
    """Base exception for Order Manager service."""

    def __init__(self, message: str, trace_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.trace_id = trace_id


class ConfigurationError(OrderManagerError):
    """Raised when configuration is invalid or missing."""

    pass


class DatabaseError(OrderManagerError):
    """Raised when database operations fail."""

    pass


class QueueError(OrderManagerError):
    """Raised when queue operations fail."""

    pass


class BybitAPIError(OrderManagerError):
    """Raised when Bybit API operations fail."""

    pass


class OrderExecutionError(OrderManagerError):
    """Raised when order execution fails."""

    pass


class RiskLimitError(OrderManagerError):
    """Raised when risk limits are exceeded."""

    pass

