"""Custom exceptions for dashboard-api service."""


class DashboardAPIError(Exception):
    """Base exception for dashboard-api errors."""

    def __init__(self, message: str, trace_id: str = None):
        """
        Initialize exception.

        Args:
            message: Error message
            trace_id: Optional trace ID for request tracking
        """
        super().__init__(message)
        self.message = message
        self.trace_id = trace_id


class DatabaseError(DashboardAPIError):
    """Database operation error."""

    pass


class ConfigurationError(DashboardAPIError):
    """Configuration error."""

    pass

