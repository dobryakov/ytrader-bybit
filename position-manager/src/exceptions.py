"""Custom exceptions for Position Manager service."""


class PositionManagerError(Exception):
    """Base exception for Position Manager errors."""

    pass


class DatabaseError(PositionManagerError):
    """Database operation error."""

    pass


class QueueError(PositionManagerError):
    """Message queue operation error."""

    pass


class ValidationError(PositionManagerError):
    """Position validation error."""

    pass


class NotFoundError(PositionManagerError):
    """Resource not found error."""

    pass

