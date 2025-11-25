"""Structured logging infrastructure with structlog."""

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from .settings import settings


def add_trace_id(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add trace_id to log events if available in context."""
    # Trace ID will be added by middleware/context vars
    return event_dict


def setup_logging() -> None:
    """Configure structured logging with structlog."""
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.ws_gateway_log_level.upper(), logging.INFO),
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # Merge context variables (trace_id, etc.)
            structlog.processors.add_log_level,  # Add log level
            structlog.processors.TimeStamper(fmt="iso"),  # Add ISO timestamp
            structlog.processors.StackInfoRenderer(),  # Add stack info for exceptions
            structlog.processors.format_exc_info,  # Format exceptions
            add_trace_id,  # Custom processor for trace IDs
            structlog.processors.JSONRenderer()  # Output as JSON
            if settings.ws_gateway_log_level.upper() == "DEBUG"
            else structlog.dev.ConsoleRenderer(),  # Pretty console output for non-DEBUG
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.ws_gateway_log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)

