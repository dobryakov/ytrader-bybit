"""
Unit tests for ExitSignalRateLimiter.
"""

import pytest
import time
from src.services.exit_signal_rate_limiter import ExitSignalRateLimiter


def test_rate_limiter_allows_first_signal():
    """Test that first signal is always allowed."""
    limiter = ExitSignalRateLimiter(max_signals_per_minute=10, cooldown_seconds=60)

    allowed, reason = limiter.check_rate_limit("BTCUSDT")

    assert allowed is True
    assert reason is None


def test_rate_limiter_enforces_cooldown():
    """Test that cooldown period is enforced."""
    limiter = ExitSignalRateLimiter(max_signals_per_minute=10, cooldown_seconds=60)

    # First signal allowed
    allowed, _ = limiter.check_rate_limit("BTCUSDT")
    assert allowed is True

    # Second signal immediately after should be blocked by cooldown
    allowed, reason = limiter.check_rate_limit("BTCUSDT")
    assert allowed is False
    assert "cooldown" in reason.lower()


def test_rate_limiter_enforces_per_minute_limit():
    """Test that per-minute limit is enforced."""
    # Create fresh limiter instance to avoid any state issues
    limiter = ExitSignalRateLimiter(max_signals_per_minute=3, cooldown_seconds=0)
    
    asset = "TEST_ASSET_FOR_RATE_LIMIT"
    # Ensure clean state
    if asset in limiter._signal_times:
        limiter._signal_times[asset].clear()
    if asset in limiter._last_signal_time:
        del limiter._last_signal_time[asset]

    # Generate 3 signals (should all be allowed)
    for i in range(3):
        allowed, reason = limiter.check_rate_limit(asset)
        assert allowed is True, f"Signal {i+1} should be allowed, got reason: {reason}"

    # 4th signal should be blocked
    allowed, reason = limiter.check_rate_limit(asset)
    assert allowed is False, f"4th signal should be blocked, got reason: {reason}"
    assert "rate limit" in reason.lower() or "minute" in reason.lower()


def test_rate_limiter_per_asset():
    """Test that rate limiting is per-asset."""
    limiter = ExitSignalRateLimiter(max_signals_per_minute=1, cooldown_seconds=0)

    # Signal for BTCUSDT
    allowed, _ = limiter.check_rate_limit("BTCUSDT")
    assert allowed is True

    # Signal for ETHUSDT (different asset) should still be allowed
    allowed, _ = limiter.check_rate_limit("ETHUSDT")
    assert allowed is True

    # Second signal for BTCUSDT should be blocked
    allowed, _ = limiter.check_rate_limit("BTCUSDT")
    assert allowed is False


def test_rate_limiter_reset_asset():
    """Test resetting rate limiter for specific asset."""
    limiter = ExitSignalRateLimiter(max_signals_per_minute=1, cooldown_seconds=0)

    # Generate signal
    allowed, _ = limiter.check_rate_limit("BTCUSDT")
    assert allowed is True

    # Second signal blocked
    allowed, _ = limiter.check_rate_limit("BTCUSDT")
    assert allowed is False

    # Reset
    limiter.reset_asset("BTCUSDT")

    # Signal should be allowed again
    allowed, _ = limiter.check_rate_limit("BTCUSDT")
    assert allowed is True


def test_rate_limiter_get_status():
    """Test getting rate limiter status."""
    limiter = ExitSignalRateLimiter(max_signals_per_minute=10, cooldown_seconds=60)

    # Generate a signal
    limiter.check_rate_limit("BTCUSDT")

    status = limiter.get_status("BTCUSDT")

    assert status["asset"] == "BTCUSDT"
    assert status["recent_signals_count"] >= 1
    assert status["max_signals_per_minute"] == 10
    assert "cooldown_remaining" in status
    assert "last_signal_time" in status

