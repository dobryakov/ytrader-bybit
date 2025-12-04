"""
Perpetual features computation module.
"""
from typing import Dict, Optional
from datetime import datetime, timezone

from src.models.rolling_windows import RollingWindows


def compute_funding_rate(
    rolling_windows: RollingWindows,
) -> Optional[float]:
    """Compute current funding rate."""
    # Get latest funding rate from rolling windows or separate storage
    # For now, we'll need to track funding rate separately
    # This is a placeholder - actual implementation depends on how funding rate is stored
    return None


def compute_time_to_funding(
    rolling_windows: RollingWindows,
) -> Optional[float]:
    """Compute time to next funding (in seconds)."""
    # Get next funding time from rolling windows or separate storage
    # For now, we'll need to track next funding time separately
    # This is a placeholder - actual implementation depends on how funding time is stored
    return None


def compute_all_perpetual_features(
    rolling_windows: RollingWindows,
    funding_rate: Optional[float] = None,
    next_funding_time: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """Compute all perpetual features."""
    features = {}
    
    # Use provided funding rate if available
    if funding_rate is not None:
        features["funding_rate"] = funding_rate
    else:
        features["funding_rate"] = compute_funding_rate(rolling_windows)
    
    # Compute time to funding
    if next_funding_time is not None:
        now = datetime.now(timezone.utc)
        next_funding_dt = datetime.fromtimestamp(next_funding_time / 1000, tz=timezone.utc)
        time_to_funding = (next_funding_dt - now).total_seconds()
        features["time_to_funding"] = max(0.0, time_to_funding)
    else:
        features["time_to_funding"] = compute_time_to_funding(rolling_windows)
    
    return features

