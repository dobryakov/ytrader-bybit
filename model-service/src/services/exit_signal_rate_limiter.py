"""
Exit signal rate limiter service.

Prevents excessive exit signal generation with per-asset rate limiting,
cooldown periods, and maximum signals per time window.
"""

import time
from typing import Dict, Optional, Any
from collections import defaultdict
from datetime import datetime, timedelta

from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)


class ExitSignalRateLimiter:
    """
    Rate limiter for exit signal generation.

    Prevents excessive exit signal generation with:
    - Per-asset rate limiting
    - Cooldown period after exit signal
    - Maximum signals per time window
    """

    def __init__(
        self,
        max_signals_per_minute: Optional[int] = None,
        cooldown_seconds: Optional[int] = None,
    ):
        """
        Initialize exit signal rate limiter.

        Args:
            max_signals_per_minute: Maximum exit signals per asset per minute (default from settings)
            cooldown_seconds: Cooldown period after exit signal in seconds (default: 60)
        """
        self.max_signals_per_minute = max_signals_per_minute if max_signals_per_minute is not None else settings.exit_strategy_rate_limit
        self.cooldown_seconds = cooldown_seconds if cooldown_seconds is not None else 60  # Default 1 minute cooldown

        # Per-asset tracking
        self._signal_times: Dict[str, list] = defaultdict(list)  # Asset -> list of signal timestamps
        self._last_signal_time: Dict[str, float] = {}  # Asset -> last signal timestamp

    def check_rate_limit(self, asset: str) -> tuple[bool, Optional[str]]:
        """
        Check if exit signal can be generated for asset within rate limits.

        Args:
            asset: Trading pair symbol

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
            - If allowed=True, reason is None
            - If allowed=False, reason explains why (cooldown, rate limit, etc.)
        """
        current_time = time.time()

        # Check cooldown period (only if cooldown_seconds > 0)
        if self.cooldown_seconds > 0 and asset in self._last_signal_time:
            time_since_last = current_time - self._last_signal_time[asset]
            if time_since_last < self.cooldown_seconds:
                wait_seconds = self.cooldown_seconds - time_since_last
                reason = f"Cooldown period active: {wait_seconds:.1f}s remaining"
                logger.debug("Exit signal rate limited: cooldown", asset=asset, wait_seconds=wait_seconds)
                return False, reason

        # Check rate limit (signals per minute)
        cutoff_time = current_time - 60.0  # 1 minute window
        signal_times = self._signal_times[asset]
        recent_signals = [t for t in signal_times if t >= cutoff_time]

        if len(recent_signals) >= self.max_signals_per_minute:
            oldest_signal = recent_signals[0] if recent_signals else current_time
            wait_seconds = 60.0 - (current_time - oldest_signal)
            reason = (
                f"Rate limit exceeded: {len(recent_signals)}/{self.max_signals_per_minute} "
                f"signals in last minute. Wait {wait_seconds:.1f}s"
            )
            logger.warning("Exit signal rate limited: per-minute limit", asset=asset, count=len(recent_signals))
            return False, reason

        # Allowed - record signal
        signal_times.append(current_time)
        # Only record last signal time if cooldown is enabled
        if self.cooldown_seconds > 0:
            self._last_signal_time[asset] = current_time

        # Clean up old timestamps (keep only last 5 minutes)
        cutoff_cleanup = current_time - 300.0
        self._signal_times[asset] = [t for t in signal_times if t >= cutoff_cleanup]

        logger.debug("Exit signal rate limit check passed", asset=asset, recent_count=len(recent_signals))
        return True, None

    def reset_asset(self, asset: str) -> None:
        """
        Reset rate limiting for a specific asset.

        Args:
            asset: Trading pair symbol
        """
        if asset in self._signal_times:
            self._signal_times[asset].clear()
        if asset in self._last_signal_time:
            del self._last_signal_time[asset]
        logger.debug("Rate limiter reset for asset", asset=asset)

    def get_status(self, asset: str) -> Dict[str, Any]:
        """
        Get current rate limiter status for asset.

        Args:
            asset: Trading pair symbol

        Returns:
            Dictionary with current count, limits, and availability
        """
        current_time = time.time()
        cutoff_time = current_time - 60.0

        signal_times = self._signal_times.get(asset, [])
        recent_signals = [t for t in signal_times if t >= cutoff_time]
        recent_count = len(recent_signals)

        last_signal_time = self._last_signal_time.get(asset)
        time_since_last = current_time - last_signal_time if last_signal_time else None
        cooldown_remaining = (
            max(0, self.cooldown_seconds - time_since_last) if time_since_last is not None else None
        )

        return {
            "asset": asset,
            "recent_signals_count": recent_count,
            "max_signals_per_minute": self.max_signals_per_minute,
            "remaining": max(0, self.max_signals_per_minute - recent_count),
            "cooldown_seconds": self.cooldown_seconds,
            "cooldown_remaining": cooldown_remaining,
            "last_signal_time": last_signal_time,
        }


# Global exit signal rate limiter instance
exit_signal_rate_limiter = ExitSignalRateLimiter()

