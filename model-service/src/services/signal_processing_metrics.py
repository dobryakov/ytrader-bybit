"""
Signal processing delay metrics.

Tracks delay between signal creation and publication for monitoring purposes.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Dict, Any

from ..config.logging import get_logger

logger = get_logger(__name__)


class SignalProcessingMetrics:
    """In-memory metrics for signal processing delays."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_delay_seconds: float = 0.0
        self._max_delay_seconds: float = 0.0
        self._count: int = 0
        # Simple histogram buckets in seconds
        self._buckets = {
            "<1s": 0,
            "1-5s": 0,
            "5-30s": 0,
            "30-120s": 0,
            ">120s": 0,
        }
        self._last_reset: datetime = datetime.utcnow()

    def _bucket_name(self, delay_seconds: float) -> str:
        if delay_seconds < 1:
            return "<1s"
        if delay_seconds < 5:
            return "1-5s"
        if delay_seconds < 30:
            return "5-30s"
        if delay_seconds < 120:
            return "30-120s"
        return ">120s"

    def record_delay(self, delay_seconds: float) -> None:
        """
        Record a new processing delay measurement.

        Args:
            delay_seconds: Time between signal creation and publication in seconds.
        """
        if delay_seconds < 0:
            # Ignore negative values (clock issues)
            return

        with self._lock:
            self._count += 1
            self._total_delay_seconds += delay_seconds
            if delay_seconds > self._max_delay_seconds:
                self._max_delay_seconds = delay_seconds

            bucket = self._bucket_name(delay_seconds)
            if bucket in self._buckets:
                self._buckets[bucket] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Return current metrics snapshot."""
        with self._lock:
            average_delay = (
                self._total_delay_seconds / self._count if self._count > 0 else 0.0
            )
            return {
                "total_signals": self._count,
                "average_delay_seconds": average_delay,
                "max_delay_seconds": self._max_delay_seconds,
                "bucket_counts": dict(self._buckets),
                "last_reset": self._last_reset.isoformat() + "Z",
            }

    def reset(self) -> None:
        """Reset all metrics (used mainly for testing or manual resets)."""
        with self._lock:
            self._total_delay_seconds = 0.0
            self._max_delay_seconds = 0.0
            self._count = 0
            for key in self._buckets:
                self._buckets[key] = 0
            self._last_reset = datetime.utcnow()
            logger.info("Signal processing delay metrics reset")


# Global metrics instance
signal_processing_metrics = SignalProcessingMetrics()


