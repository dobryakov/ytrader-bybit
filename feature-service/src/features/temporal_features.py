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


def compute_all_temporal_features(
    timestamp: datetime,
) -> Dict[str, float]:
    """Compute all temporal features."""
    features = {}
    
    features["time_of_day_sin"] = compute_time_of_day_sin(timestamp)
    features["time_of_day_cos"] = compute_time_of_day_cos(timestamp)
    
    return features

