"""Error handling and exception classes."""

from typing import Optional


class WebSocketGatewayError(Exception):
    """Base exception for WebSocket Gateway service."""

    def __init__(self, message: str, trace_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.trace_id = trace_id


class ConfigurationError(WebSocketGatewayError):
    """Raised when configuration is invalid or missing."""

    pass


class DatabaseError(WebSocketGatewayError):
    """Raised when database operations fail."""

    pass


class QueueError(WebSocketGatewayError):
    """Raised when queue operations fail."""

    pass


class WebSocketConnectionError(WebSocketGatewayError):
    """Raised when WebSocket connection fails."""

    pass


class AuthenticationError(WebSocketGatewayError):
    """Raised when authentication fails."""

    pass


class SubscriptionError(WebSocketGatewayError):
    """Raised when subscription operations fail."""

    pass


class ValidationError(WebSocketGatewayError):
    """Raised when data validation fails."""

    pass

