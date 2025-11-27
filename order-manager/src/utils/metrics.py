"""Performance monitoring and metrics utilities."""

import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from functools import wraps

from ..config.logging import get_logger

logger = get_logger(__name__)

# Simple in-memory metrics storage (can be replaced with Prometheus/StatsD in production)
# IMPORTANT: api_response_times must always be a list, never a dict
_metrics: Dict[str, Any] = {
    "order_processing_latency": [],
    "api_response_times": [],  # Must be list for latency tracking
    "queue_depths": {},
    "error_counts": {},
}

def _ensure_api_response_times_is_list():
    """Ensure api_response_times is always a list - call this before any operation."""
    global _metrics
    if "api_response_times" not in _metrics:
        _metrics["api_response_times"] = []
    elif not isinstance(_metrics["api_response_times"], list):
        # Force reset to list if it somehow became a dict
        _metrics["api_response_times"] = []


def record_latency(metric_name: str, latency_ms: float, **tags):
    """Record latency metric.

    Args:
        metric_name: Name of the metric
        latency_ms: Latency in milliseconds
        tags: Additional tags for the metric
    """
    # Always ensure api_response_times is a list first (critical fix)
    if metric_name == "api_response_times":
        if metric_name not in _metrics or not isinstance(_metrics[metric_name], list):
            _metrics[metric_name] = []
    
    # Ensure metric is a list, not a dict
    if metric_name not in _metrics:
        _metrics[metric_name] = []
    elif not isinstance(_metrics[metric_name], list):
        # If it was initialized as something else, reset it
        _metrics[metric_name] = []

    # Double-check before append (safety net)
    if not isinstance(_metrics[metric_name], list):
        _metrics[metric_name] = []

    _metrics[metric_name].append({
        "value": latency_ms,
        "tags": tags,
        "timestamp": time.time(),
    })

    # Keep only last 1000 measurements
    if len(_metrics[metric_name]) > 1000:
        _metrics[metric_name] = _metrics[metric_name][-1000:]

    logger.debug(
        "metric_recorded",
        metric=metric_name,
        latency_ms=latency_ms,
        **tags,
    )


def increment_counter(metric_name: str, value: int = 1, **tags):
    """Increment a counter metric.

    Args:
        metric_name: Name of the metric
        value: Value to increment by
        tags: Additional tags for the metric
    """
    if "counters" not in _metrics:
        _metrics["counters"] = {}
    key = f"{metric_name}:{':'.join(f'{k}={v}' for k, v in sorted(tags.items()))}"
    _metrics["counters"][key] = _metrics["counters"].get(key, 0) + value

    logger.debug(
        "counter_incremented",
        metric=metric_name,
        value=value,
        **tags,
    )


@asynccontextmanager
async def measure_latency(metric_name: str, **tags):
    """Context manager to measure operation latency.

    Usage:
        async with measure_latency("order_processing", asset="BTCUSDT"):
            # operation to measure
            await process_order()
    """
    start_time = time.time()
    try:
        yield
    finally:
        latency_ms = (time.time() - start_time) * 1000
        record_latency(metric_name, latency_ms, **tags)


def track_latency(metric_name: str, **default_tags):
    """Decorator to track function execution latency.

    Usage:
        @track_latency("api_request", endpoint="orders")
        async def get_orders():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tags = default_tags.copy()
            async with measure_latency(metric_name, **tags):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tags = default_tags.copy()
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                latency_ms = (time.time() - start_time) * 1000
                record_latency(metric_name, latency_ms, **tags)
            return result

        if hasattr(func, "__code__") and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        return sync_wrapper

    return decorator


def get_metrics_summary() -> Dict[str, Any]:
    """Get summary of all metrics.

    Returns:
        Dictionary with metric summaries
    """
    # Ensure api_response_times is always a list before processing
    _ensure_api_response_times_is_list()
    
    summary: Dict[str, Any] = {}

    # Calculate statistics for latency metrics
    for metric_name, values in _metrics.items():
        if isinstance(values, list) and values:
            latencies = [v["value"] for v in values]
            summary[metric_name] = {
                "count": len(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "avg": sum(latencies) / len(latencies),
                "p50": sorted(latencies)[len(latencies) // 2],
                "p95": sorted(latencies)[int(len(latencies) * 0.95)],
                "p99": sorted(latencies)[int(len(latencies) * 0.99)],
            }
        elif isinstance(values, list) and not values:
            # Empty list - return empty summary
            summary[metric_name] = {
                "count": 0,
                "min": 0,
                "max": 0,
                "avg": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0,
            }
        elif isinstance(values, dict):
            summary[metric_name] = values.copy()

    return summary


def reset_metrics():
    """Reset all metrics (useful for testing)."""
    global _metrics
    _metrics = {
        "order_processing_latency": [],
        "api_response_times": [],  # Changed from {} to [] for latency tracking
        "queue_depths": {},
        "error_counts": {},
    }

