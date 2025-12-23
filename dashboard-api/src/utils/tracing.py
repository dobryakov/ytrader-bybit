"""Trace ID generation and propagation utilities.

This module provides utilities for generating and propagating trace IDs
across the application for request flow tracking and observability.
"""

import contextvars
import uuid
from typing import Optional

import structlog

# Context variable to store trace ID for the current async context
_trace_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "trace_id", default=None
)


def generate_trace_id() -> str:
    """Generate a new unique trace ID.

    Returns:
        A UUID4 string formatted as a trace ID.
    """
    return str(uuid.uuid4())


def set_trace_id(trace_id: Optional[str]) -> None:
    """Set trace ID in the current async context.

    This makes the trace ID available throughout the async call chain
    via contextvars, which are automatically propagated through async/await.

    Args:
        trace_id: Trace ID to set, or None to clear it.
    """
    _trace_id_context.set(trace_id)
    # Also bind to structlog context for logging
    if trace_id:
        structlog.contextvars.bind_contextvars(trace_id=trace_id)
    else:
        structlog.contextvars.clear_contextvars()


def get_trace_id() -> Optional[str]:
    """Get current trace ID from the async context.

    Returns:
        Current trace ID if available, None otherwise.
    """
    return _trace_id_context.get(None)


def get_or_create_trace_id() -> str:
    """Get current trace ID or create a new one if none exists.

    This is useful for ensuring trace IDs exist in contexts where they
    might not have been explicitly set (e.g., background tasks).

    Returns:
        Existing trace ID if available, otherwise a newly generated one.
    """
    trace_id = get_trace_id()
    if not trace_id:
        trace_id = generate_trace_id()
        set_trace_id(trace_id)
    return trace_id


def clear_trace_id() -> None:
    """Clear trace ID from the current async context."""
    _trace_id_context.set(None)
    structlog.contextvars.clear_contextvars()

