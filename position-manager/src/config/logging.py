"""
Structured logging setup using structlog.

Provides consistent logging with trace IDs for request flow tracking.
"""

import sys
import structlog
from typing import Any

from .settings import settings


def configure_logging() -> None:
    """
    Configure structured logging with trace IDs.

    Sets up structlog with JSON output for production and console output
    for development, including trace ID support.
    """
    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,  # Merge context variables (trace_id, etc.)
        structlog.stdlib.add_log_level,  # Add log level
        structlog.stdlib.add_logger_name,  # Add logger name
        structlog.processors.TimeStamper(fmt="iso"),  # Add ISO timestamp
        structlog.processors.StackInfoRenderer(),  # Add stack info for exceptions
        structlog.processors.format_exc_info,  # Format exceptions
    ]

    # Add JSON renderer for production, console for development
    if settings.position_manager_log_level == "DEBUG":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    import logging

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.position_manager_log_level),
    )


def get_logger(name: str = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def set_trace_id(trace_id: str) -> None:
    """
    Set trace ID in logging context.

    Args:
        trace_id: Unique trace identifier for request flow tracking
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(trace_id=trace_id)


def bind_context(**kwargs: Any) -> None:
    """
    Bind additional context variables to logging.

    Args:
        **kwargs: Context variables to bind (e.g., position_id, asset, mode)
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all context variables from logging."""
    structlog.contextvars.clear_contextvars()

