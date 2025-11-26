"""
Rate limiting service for signal generation.

Implements configurable rate limiting with burst allowance to prevent resource exhaustion.
"""

import time
from collections import deque
from typing import Optional

from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Rate limiter with burst allowance for signal generation."""

    def __init__(
        self,
        rate_limit: Optional[int] = None,
        burst_allowance: Optional[int] = None,
        window_seconds: int = 60,
    ):
        """
        Initialize rate limiter.

        Args:
            rate_limit: Maximum number of signals per window (default from settings)
            burst_allowance: Additional signals allowed in burst (default from settings)
            window_seconds: Time window in seconds (default: 60 seconds = 1 minute)
        """
        self.rate_limit = rate_limit or settings.signal_generation_rate_limit
        self.burst_allowance = burst_allowance or settings.signal_generation_burst_allowance
        self.window_seconds = window_seconds
        self.max_allowed = self.rate_limit + self.burst_allowance

        # Track request timestamps using sliding window
        self._request_times: deque = deque()

    def check_rate_limit(self) -> tuple[bool, Optional[str]]:
        """
        Check if a signal can be generated within rate limits.

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
            - If allowed=True, reason is None
            - If allowed=False, reason explains why (rate limit exceeded, burst exceeded, etc.)
        """
        current_time = time.time()

        # Remove timestamps outside the current window
        cutoff_time = current_time - self.window_seconds
        while self._request_times and self._request_times[0] < cutoff_time:
            self._request_times.popleft()

        # Check if we're within limits
        current_count = len(self._request_times)

        if current_count < self.rate_limit:
            # Within normal rate limit
            self._request_times.append(current_time)
            logger.debug(
                "Rate limit check passed (normal)",
                current_count=current_count,
                rate_limit=self.rate_limit,
            )
            return True, None

        elif current_count < self.max_allowed:
            # Within burst allowance
            self._request_times.append(current_time)
            logger.debug(
                "Rate limit check passed (burst)",
                current_count=current_count,
                max_allowed=self.max_allowed,
                burst_used=current_count - self.rate_limit,
            )
            return True, None

        else:
            # Rate limit exceeded
            oldest_time = self._request_times[0] if self._request_times else current_time
            wait_seconds = self.window_seconds - (current_time - oldest_time)
            reason = f"Rate limit exceeded: {current_count}/{self.max_allowed} signals in last {self.window_seconds}s. Wait {wait_seconds:.1f}s"
            logger.warning(
                "Rate limit exceeded",
                current_count=current_count,
                max_allowed=self.max_allowed,
                wait_seconds=wait_seconds,
            )
            return False, reason

    def reset(self) -> None:
        """Reset rate limiter (clear all tracked requests)."""
        self._request_times.clear()
        logger.info("Rate limiter reset")

    def get_status(self) -> dict:
        """
        Get current rate limiter status.

        Returns:
            Dictionary with current count, limits, and availability
        """
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds

        # Count requests in current window
        valid_requests = [t for t in self._request_times if t >= cutoff_time]
        current_count = len(valid_requests)

        return {
            "current_count": current_count,
            "rate_limit": self.rate_limit,
            "burst_allowance": self.burst_allowance,
            "max_allowed": self.max_allowed,
            "remaining": max(0, self.max_allowed - current_count),
            "window_seconds": self.window_seconds,
        }


# Global rate limiter instance
rate_limiter = RateLimiter()

