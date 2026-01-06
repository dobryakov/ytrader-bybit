"""
Temporal features computation module.
"""
from typing import Dict
from datetime import datetime
import math


def compute_time_of_day_sin(timestamp: datetime) -> float:
    """Compute time of day using sine encoding (cyclic)."""
    hour = timestamp.hour
    # sin(2π * hour / 24)
    return math.sin(2 * math.pi * hour / 24)


def compute_time_of_day_cos(timestamp: datetime) -> float:
    """Compute time of day using cosine encoding (cyclic)."""
    hour = timestamp.hour
    # cos(2π * hour / 24)
    return math.cos(2 * math.pi * hour / 24)


def compute_day_of_week_sin(timestamp: datetime) -> float:
    """Compute day of week using sine encoding (cyclic).
    
    Monday = 0, Sunday = 6
    sin(2π * weekday / 7)
    """
    weekday = timestamp.weekday()  # 0 = Monday, 6 = Sunday
    # sin(2π * weekday / 7)
    return math.sin(2 * math.pi * weekday / 7)


def compute_day_of_week_cos(timestamp: datetime) -> float:
    """Compute day of week using cosine encoding (cyclic).
    
    Monday = 0, Sunday = 6
    cos(2π * weekday / 7)
    """
    weekday = timestamp.weekday()  # 0 = Monday, 6 = Sunday
    # cos(2π * weekday / 7)
    return math.cos(2 * math.pi * weekday / 7)


def compute_is_asia_session(timestamp: datetime) -> float:
    """Compute binary flag for Asia trading session.
    
    Asia session: 00:00 – 08:00 UTC
    Returns 1.0 if timestamp is within session, 0.0 otherwise.
    """
    hour = timestamp.hour
    # 00:00 – 08:00 UTC
    if 0 <= hour < 8:
        return 1.0
    return 0.0


def compute_is_london_session(timestamp: datetime) -> float:
    """Compute binary flag for London trading session.
    
    London session: 07:00 – 15:00 UTC
    Returns 1.0 if timestamp is within session, 0.0 otherwise.
    """
    hour = timestamp.hour
    # 07:00 – 15:00 UTC
    if 7 <= hour < 15:
        return 1.0
    return 0.0


def compute_is_ny_session(timestamp: datetime) -> float:
    """Compute binary flag for New York trading session.
    
    NY session: 13:00 – 21:00 UTC
    Returns 1.0 if timestamp is within session, 0.0 otherwise.
    """
    hour = timestamp.hour
    # 13:00 – 21:00 UTC
    if 13 <= hour < 21:
        return 1.0
    return 0.0


def compute_all_temporal_features(
    timestamp: datetime,
) -> Dict[str, float]:
    """Compute all temporal features."""
    features = {}
    
    features["time_of_day_sin"] = compute_time_of_day_sin(timestamp)
    features["time_of_day_cos"] = compute_time_of_day_cos(timestamp)
    features["dow_sin"] = compute_day_of_week_sin(timestamp)
    features["dow_cos"] = compute_day_of_week_cos(timestamp)
    features["is_asia_session"] = compute_is_asia_session(timestamp)
    features["is_london_session"] = compute_is_london_session(timestamp)
    features["is_ny_session"] = compute_is_ny_session(timestamp)
    
    return features

