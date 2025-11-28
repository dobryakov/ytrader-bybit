"""
Retry utility for handling transient failures in RabbitMQ and database operations.

Provides exponential backoff retry logic with configurable attempts and delays.
"""

import asyncio
from typing import Callable, TypeVar, Optional, List
from functools import wraps

from .logging import get_logger
from .exceptions import MessageQueueError, DatabaseError

logger = get_logger(__name__)

T = TypeVar("T")

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 30.0  # seconds
DEFAULT_BACKOFF_MULTIPLIER = 2.0

# Retryable exceptions
RETRYABLE_EXCEPTIONS = (
    MessageQueueError,
    DatabaseError,
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)


async def retry_async(
    func: Callable[..., T],
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
    retryable_exceptions: tuple = RETRYABLE_EXCEPTIONS,
    operation_name: Optional[str] = None,
    *args,
    **kwargs,
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 30.0)
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
        retryable_exceptions: Tuple of exceptions that should trigger retry
        operation_name: Name of the operation for logging (optional)
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result of the function call

    Raises:
        Last exception if all retries fail
    """
    op_name = operation_name or func.__name__
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e

            if attempt < max_retries:
                logger.warning(
                    f"Retryable error in {op_name}",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    delay=delay,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                await asyncio.sleep(delay)
                delay = min(delay * backoff_multiplier, max_delay)
            else:
                logger.error(
                    f"All retries exhausted for {op_name}",
                    max_retries=max_retries,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
        except Exception as e:
            # Check if this is a "queue not found" error (expected when publisher not started)
            error_str = str(e)
            is_queue_not_found = (
                "no queue" in error_str.lower()
                or "not_found" in error_str.upper()
                or "ChannelNotFoundEntity" in error_str
            )

            if is_queue_not_found:
                # Queue not found is expected when publisher service isn't running
                # Don't log as error, just pass through
                logger.debug(
                    f"Queue not found in {op_name} (expected if publisher not started)",
                    error_type=type(e).__name__,
                )
            else:
                # Non-retryable exception - log as error
                logger.error(
                    f"Non-retryable error in {op_name}",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
            raise

    # If we get here, all retries were exhausted
    if last_exception:
        raise last_exception
    raise RuntimeError(f"Unexpected error in retry logic for {op_name}")


def retry_decorator(
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
    retryable_exceptions: tuple = RETRYABLE_EXCEPTIONS,
):
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 30.0)
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
        retryable_exceptions: Tuple of exceptions that should trigger retry

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_async(
                func,
                max_retries=max_retries,
                initial_delay=initial_delay,
                max_delay=max_delay,
                backoff_multiplier=backoff_multiplier,
                retryable_exceptions=retryable_exceptions,
                operation_name=func.__name__,
                *args,
                **kwargs,
            )

        return wrapper

    return decorator

