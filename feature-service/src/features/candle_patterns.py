"""
Candlestick pattern features computation module.

Computes categorical and pattern-based features from the last 3 minutes of kline data.
All features are computed using relative thresholds (averages over 3 candles).
"""
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from src.models.rolling_windows import RollingWindows


def _ensure_datetime(value) -> datetime:
    """Ensure value is a datetime object."""
    if isinstance(value, datetime):
        return value
    elif isinstance(value, str):
        from dateutil.parser import parse
        return parse(value)
    else:
        return datetime.now(timezone.utc)


def _get_candle_components(
    klines: pd.DataFrame,
    index: int,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """
    Extract candle components (open, high, low, close, volume, body_size, etc.) from kline.
    
    Returns:
        Tuple of (open, high, low, close, volume, body_size) or None if insufficient data
    """
    if index < 0 or index >= len(klines):
        return None
    
    candle = klines.iloc[index]
    
    # Extract values, ensuring numeric types
    try:
        open_price = float(pd.to_numeric(candle.get("open", 0), errors='coerce') or 0)
        high = float(pd.to_numeric(candle.get("high", 0), errors='coerce') or 0)
        low = float(pd.to_numeric(candle.get("low", 0), errors='coerce') or 0)
        close = float(pd.to_numeric(candle.get("close", 0), errors='coerce') or 0)
        volume = float(pd.to_numeric(candle.get("volume", 0), errors='coerce') or 0)
    except (ValueError, TypeError, KeyError):
        return None
    
    # Safety check: prices must be positive
    if open_price < 1e-8 or high < 1e-8 or low < 1e-8 or close < 1e-8:
        return None
    
    # Compute body size (absolute, normalized)
    body_size = abs(close - open_price) / open_price  # in decimals (0.01 = 1%)
    
    return (open_price, high, low, close, volume, body_size)


def _compute_candle_features(
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    body_size: float,
) -> Dict[str, float]:
    """
    Compute all features for a single candle.
    
    Returns:
        Dictionary with candle features
    """
    features = {}
    
    # Body components
    body_direction = (close - open_price) / open_price  # positive = green, negative = red
    total_range = (high - low) / open_price if open_price > 0 else 0.0
    
    # Upper and lower shadows (in decimals)
    upper_shadow = (high - max(open_price, close)) / open_price if open_price > 0 else 0.0
    lower_shadow = (min(open_price, close) - low) / open_price if open_price > 0 else 0.0
    
    # Color (binary)
    is_green = 1.0 if close > open_price else 0.0
    is_red = 1.0 if close < open_price else 0.0
    
    # Special patterns (single candle)
    # Doji: very small body (< 0.05% of range)
    is_doji = 1.0 if body_size < (0.05 / 100) * total_range and total_range > 0 else 0.0
    
    # Hammer: long lower shadow, short upper shadow
    is_hammer = 1.0 if (lower_shadow > 2 * body_size and upper_shadow < 0.3 * body_size and body_size > 0) else 0.0
    
    # Inverted hammer: long upper shadow, short lower shadow
    is_inverted_hammer = 1.0 if (upper_shadow > 2 * body_size and lower_shadow < 0.3 * body_size and body_size > 0) else 0.0
    
    # Pin bar: very long one shadow (> 2x range)
    is_pin_bar = 1.0 if ((upper_shadow > 2 * total_range) or (lower_shadow > 2 * total_range)) and total_range > 0 else 0.0
    
    # Shooting star: long upper shadow + red candle
    is_shooting_star = 1.0 if (
        upper_shadow > 2 * body_size and 
        lower_shadow < 0.3 * body_size and 
        close < open_price and
        body_size > 0
    ) else 0.0
    
    features.update({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "body_size": body_size,
        "body_direction": body_direction,
        "total_range": total_range,
        "upper_shadow": upper_shadow,
        "lower_shadow": lower_shadow,
        "is_green": is_green,
        "is_red": is_red,
        "is_doji": is_doji,
        "is_hammer": is_hammer,
        "is_inverted_hammer": is_inverted_hammer,
        "is_pin_bar": is_pin_bar,
        "is_shooting_star": is_shooting_star,
    })
    
    return features


def compute_all_candle_patterns_3m(
    rolling_windows: RollingWindows,
) -> Dict[str, Optional[float]]:
    """
    Compute all candlestick pattern features from the last 3 minutes (3 candles).
    
    Returns dictionary with ~77 features:
    - Basic categorical features for each candle (color, body size, shadows, volume)
    - Special single-candle patterns (doji, hammer, etc.)
    - Multi-candle sequences (color patterns, body trends, etc.)
    - Classic reversal patterns (stars, engulfing, harami, etc.)
    
    Args:
        rolling_windows: RollingWindows instance with kline data
        
    Returns:
        Dictionary of feature name -> feature value (0.0 or 1.0 for binary, float for ratios)
    """
    features = {}
    
    now = _ensure_datetime(rolling_windows.last_update)
    # Get last 3 minutes of kline data (need at least 3 candles)
    start_time = now - timedelta(minutes=4)  # Extra buffer to ensure we have 3 complete candles
    klines = rolling_windows.get_klines_for_window("1m", start_time, now)
    
    if len(klines) < 3:
        # Return all features as None if insufficient data
        return _get_empty_features_dict()
    
    # Sort by timestamp to ensure correct order (0 = oldest, 2 = newest/current)
    klines_sorted = klines.sort_values("timestamp").reset_index(drop=True)
    
    # Get last 3 candles (most recent)
    candle_data = []
    for i in range(-3, 0):  # Last 3 candles
        idx = len(klines_sorted) + i
        candle_comp = _get_candle_components(klines_sorted, idx)
        if candle_comp is None:
            return _get_empty_features_dict()
        candle_data.append(candle_comp)
    
    # Extract candle components (0 = oldest, 1 = middle, 2 = newest)
    open_0, high_0, low_0, close_0, volume_0, body_size_0 = candle_data[0]
    open_1, high_1, low_1, close_1, volume_1, body_size_1 = candle_data[1]
    open_2, high_2, low_2, close_2, volume_2, body_size_2 = candle_data[2]
    
    # Compute features for each candle
    candle_0_features = _compute_candle_features(open_0, high_0, low_0, close_0, volume_0, body_size_0)
    candle_1_features = _compute_candle_features(open_1, high_1, low_1, close_1, volume_1, body_size_1)
    candle_2_features = _compute_candle_features(open_2, high_2, low_2, close_2, volume_2, body_size_2)
    
    # Compute relative thresholds (averages over 3 candles)
    avg_body_size = (body_size_0 + body_size_1 + body_size_2) / 3.0
    avg_upper_shadow = (candle_0_features["upper_shadow"] + candle_1_features["upper_shadow"] + candle_2_features["upper_shadow"]) / 3.0
    avg_lower_shadow = (candle_0_features["lower_shadow"] + candle_1_features["lower_shadow"] + candle_2_features["lower_shadow"]) / 3.0
    avg_volume = (volume_0 + volume_1 + volume_2) / 3.0
    
    # Use minimum threshold if average is zero (all doji case)
    min_body_threshold = 0.01 / 100  # 0.01%
    min_shadow_threshold = 0.01 / 100
    min_volume_threshold = 1e-8
    
    body_threshold = max(avg_body_size, min_body_threshold)
    upper_shadow_threshold = max(avg_upper_shadow, min_shadow_threshold)
    lower_shadow_threshold = max(avg_lower_shadow, min_shadow_threshold)
    volume_threshold = max(avg_volume, min_volume_threshold)
    
    # ============================================================================
    # 1. BASIC CATEGORICAL FEATURES FOR EACH CANDLE (30 features)
    # ============================================================================
    
    # Color (6 features: 2 per candle × 3 candles)
    features["candle_0_is_green"] = candle_0_features["is_green"]
    features["candle_0_is_red"] = candle_0_features["is_red"]
    features["candle_1_is_green"] = candle_1_features["is_green"]
    features["candle_1_is_red"] = candle_1_features["is_red"]
    features["candle_2_is_green"] = candle_2_features["is_green"]
    features["candle_2_is_red"] = candle_2_features["is_red"]
    
    # Body size (6 features: 2 per candle × 3 candles)
    features["candle_0_body_large"] = 1.0 if body_size_0 > body_threshold else 0.0
    features["candle_0_body_small"] = 1.0 if body_size_0 <= body_threshold else 0.0
    features["candle_1_body_large"] = 1.0 if body_size_1 > body_threshold else 0.0
    features["candle_1_body_small"] = 1.0 if body_size_1 <= body_threshold else 0.0
    features["candle_2_body_large"] = 1.0 if body_size_2 > body_threshold else 0.0
    features["candle_2_body_small"] = 1.0 if body_size_2 <= body_threshold else 0.0
    
    # Upper shadow (6 features: 2 per candle × 3 candles)
    upper_shadow_0 = candle_0_features["upper_shadow"]
    upper_shadow_1 = candle_1_features["upper_shadow"]
    upper_shadow_2 = candle_2_features["upper_shadow"]
    
    features["candle_0_upper_shadow_large"] = 1.0 if upper_shadow_0 > upper_shadow_threshold else 0.0
    features["candle_0_upper_shadow_small"] = 1.0 if upper_shadow_0 <= upper_shadow_threshold else 0.0
    features["candle_1_upper_shadow_large"] = 1.0 if upper_shadow_1 > upper_shadow_threshold else 0.0
    features["candle_1_upper_shadow_small"] = 1.0 if upper_shadow_1 <= upper_shadow_threshold else 0.0
    features["candle_2_upper_shadow_large"] = 1.0 if upper_shadow_2 > upper_shadow_threshold else 0.0
    features["candle_2_upper_shadow_small"] = 1.0 if upper_shadow_2 <= upper_shadow_threshold else 0.0
    
    # Lower shadow (6 features: 2 per candle × 3 candles)
    lower_shadow_0 = candle_0_features["lower_shadow"]
    lower_shadow_1 = candle_1_features["lower_shadow"]
    lower_shadow_2 = candle_2_features["lower_shadow"]
    
    features["candle_0_lower_shadow_large"] = 1.0 if lower_shadow_0 > lower_shadow_threshold else 0.0
    features["candle_0_lower_shadow_small"] = 1.0 if lower_shadow_0 <= lower_shadow_threshold else 0.0
    features["candle_1_lower_shadow_large"] = 1.0 if lower_shadow_1 > lower_shadow_threshold else 0.0
    features["candle_1_lower_shadow_small"] = 1.0 if lower_shadow_1 <= lower_shadow_threshold else 0.0
    features["candle_2_lower_shadow_large"] = 1.0 if lower_shadow_2 > lower_shadow_threshold else 0.0
    features["candle_2_lower_shadow_small"] = 1.0 if lower_shadow_2 <= lower_shadow_threshold else 0.0
    
    # Volume (6 features: 2 per candle × 3 candles)
    features["candle_0_volume_large"] = 1.0 if volume_0 > volume_threshold else 0.0
    features["candle_0_volume_small"] = 1.0 if volume_0 <= volume_threshold else 0.0
    features["candle_1_volume_large"] = 1.0 if volume_1 > volume_threshold else 0.0
    features["candle_1_volume_small"] = 1.0 if volume_1 <= volume_threshold else 0.0
    features["candle_2_volume_large"] = 1.0 if volume_2 > volume_threshold else 0.0
    features["candle_2_volume_small"] = 1.0 if volume_2 <= volume_threshold else 0.0
    
    # ============================================================================
    # 2. SPECIAL SINGLE-CANDLE PATTERNS (15 features: 5 per candle × 3 candles)
    # ============================================================================
    
    features["candle_0_is_doji"] = candle_0_features["is_doji"]
    features["candle_0_is_hammer"] = candle_0_features["is_hammer"]
    features["candle_0_is_inverted_hammer"] = candle_0_features["is_inverted_hammer"]
    features["candle_0_is_pin_bar"] = candle_0_features["is_pin_bar"]
    features["candle_0_is_shooting_star"] = candle_0_features["is_shooting_star"]
    
    features["candle_1_is_doji"] = candle_1_features["is_doji"]
    features["candle_1_is_hammer"] = candle_1_features["is_hammer"]
    features["candle_1_is_inverted_hammer"] = candle_1_features["is_inverted_hammer"]
    features["candle_1_is_pin_bar"] = candle_1_features["is_pin_bar"]
    features["candle_1_is_shooting_star"] = candle_1_features["is_shooting_star"]
    
    features["candle_2_is_doji"] = candle_2_features["is_doji"]
    features["candle_2_is_hammer"] = candle_2_features["is_hammer"]
    features["candle_2_is_inverted_hammer"] = candle_2_features["is_inverted_hammer"]
    features["candle_2_is_pin_bar"] = candle_2_features["is_pin_bar"]
    features["candle_2_is_shooting_star"] = candle_2_features["is_shooting_star"]
    
    # ============================================================================
    # 3. COLOR SEQUENCE PATTERNS (6 features)
    # ============================================================================
    
    is_green_0 = candle_0_features["is_green"]
    is_green_1 = candle_1_features["is_green"]
    is_green_2 = candle_2_features["is_green"]
    
    features["pattern_all_green"] = 1.0 if (is_green_0 == 1.0 and is_green_1 == 1.0 and is_green_2 == 1.0) else 0.0
    features["pattern_all_red"] = 1.0 if (is_green_0 == 0.0 and is_green_1 == 0.0 and is_green_2 == 0.0) else 0.0
    features["pattern_green_red_green"] = 1.0 if (is_green_0 == 1.0 and is_green_1 == 0.0 and is_green_2 == 1.0) else 0.0
    features["pattern_red_green_red"] = 1.0 if (is_green_0 == 0.0 and is_green_1 == 1.0 and is_green_2 == 0.0) else 0.0
    features["pattern_green_green_red"] = 1.0 if (is_green_0 == 1.0 and is_green_1 == 1.0 and is_green_2 == 0.0) else 0.0
    features["pattern_red_red_green"] = 1.0 if (is_green_0 == 0.0 and is_green_1 == 0.0 and is_green_2 == 1.0) else 0.0
    
    # ============================================================================
    # 4. BODY SIZE TREND PATTERNS (4 features)
    # ============================================================================
    
    features["pattern_body_increasing"] = 1.0 if (body_size_0 < body_size_1 < body_size_2) else 0.0
    features["pattern_body_decreasing"] = 1.0 if (body_size_0 > body_size_1 > body_size_2) else 0.0
    features["pattern_body_middle_peak"] = 1.0 if (body_size_1 > body_size_0 and body_size_1 > body_size_2) else 0.0
    features["pattern_body_middle_low"] = 1.0 if (body_size_1 < body_size_0 and body_size_1 < body_size_2) else 0.0
    
    # ============================================================================
    # 5. VOLUME TREND PATTERNS (3 features)
    # ============================================================================
    
    features["pattern_volume_increasing"] = 1.0 if (volume_0 < volume_1 < volume_2) else 0.0
    features["pattern_volume_decreasing"] = 1.0 if (volume_0 > volume_1 > volume_2) else 0.0
    features["pattern_volume_middle_peak"] = 1.0 if (volume_1 > volume_0 and volume_1 > volume_2) else 0.0
    
    # ============================================================================
    # 6. PRICE + VOLUME COMBINED PATTERNS (for candle 2 - current) (4 features)
    # ============================================================================
    
    features["pattern_green_large_volume"] = 1.0 if (is_green_2 == 1.0 and volume_2 > volume_threshold) else 0.0
    features["pattern_red_large_volume"] = 1.0 if (is_green_2 == 0.0 and volume_2 > volume_threshold) else 0.0
    features["pattern_green_small_volume"] = 1.0 if (is_green_2 == 1.0 and volume_2 <= volume_threshold) else 0.0
    features["pattern_red_small_volume"] = 1.0 if (is_green_2 == 0.0 and volume_2 <= volume_threshold) else 0.0
    
    # ============================================================================
    # 7. ENGULFING PATTERNS (2 features: candle 2 engulfs candle 1)
    # ============================================================================
    
    # Bullish engulfing: green candle 2 completely covers red candle 1
    bullish_engulfing = (
        is_green_2 == 1.0 and
        is_green_1 == 0.0 and
        open_2 < close_1 and
        close_2 > open_1
    )
    features["pattern_bullish_engulfing"] = 1.0 if bullish_engulfing else 0.0
    
    # Bearish engulfing: red candle 2 completely covers green candle 1
    bearish_engulfing = (
        is_green_2 == 0.0 and
        is_green_1 == 1.0 and
        open_2 > close_1 and
        close_2 < open_1
    )
    features["pattern_bearish_engulfing"] = 1.0 if bearish_engulfing else 0.0
    
    # ============================================================================
    # 8. SHADOW TREND PATTERNS (2 features)
    # ============================================================================
    
    features["pattern_upper_shadows_increasing"] = 1.0 if (upper_shadow_0 < upper_shadow_1 < upper_shadow_2) else 0.0
    features["pattern_lower_shadows_increasing"] = 1.0 if (lower_shadow_0 < lower_shadow_1 < lower_shadow_2) else 0.0
    
    # ============================================================================
    # 9. CLASSIC REVERSAL PATTERNS: STARS (3 features)
    # ============================================================================
    
    # Evening star: green (large) → small → red (large)
    evening_star = (
        is_green_0 == 1.0 and body_size_0 > body_threshold and
        body_size_1 < body_threshold * 0.5 and
        is_green_2 == 0.0 and body_size_2 > body_threshold
    )
    features["pattern_evening_star"] = 1.0 if evening_star else 0.0
    
    # Morning star: red (large) → small → green (large)
    morning_star = (
        is_green_0 == 0.0 and body_size_0 > body_threshold and
        body_size_1 < body_threshold * 0.5 and
        is_green_2 == 1.0 and body_size_2 > body_threshold
    )
    features["pattern_morning_star"] = 1.0 if morning_star else 0.0
    
    # Doji star: doji in middle, large candles on sides
    doji_star = (
        candle_1_features["is_doji"] == 1.0 and
        body_size_0 > body_threshold and
        body_size_2 > body_threshold
    )
    features["pattern_doji_star"] = 1.0 if doji_star else 0.0
    
    # ============================================================================
    # 10. HARAMI PATTERNS (2 features: candle 2 inside candle 1)
    # ============================================================================
    
    # Bullish harami: small green candle 2 inside large red candle 1
    bullish_harami = (
        is_green_1 == 0.0 and body_size_1 > body_threshold and
        is_green_2 == 1.0 and body_size_2 < body_threshold * 0.5 and
        open_2 > close_1 and close_2 < open_1
    )
    features["pattern_bullish_harami"] = 1.0 if bullish_harami else 0.0
    
    # Bearish harami: small red candle 2 inside large green candle 1
    bearish_harami = (
        is_green_1 == 1.0 and body_size_1 > body_threshold and
        is_green_2 == 0.0 and body_size_2 < body_threshold * 0.5 and
        open_2 < close_1 and close_2 > open_1
    )
    features["pattern_bearish_harami"] = 1.0 if bearish_harami else 0.0
    
    # ============================================================================
    # 11. THREE METHODS PATTERNS (2 features)
    # ============================================================================
    
    # Rising three methods: green (large) → small candles in range → continuation
    rising_three = (
        is_green_0 == 1.0 and body_size_0 > body_threshold and
        body_size_1 < body_threshold * 0.5 and
        body_size_2 < body_threshold * 0.5 and
        low_1 >= low_0 * 0.999 and high_1 <= high_0 * 1.001 and  # In range of candle 0
        low_2 >= low_0 * 0.999 and high_2 <= high_0 * 1.001
    )
    features["pattern_rising_three_methods"] = 1.0 if rising_three else 0.0
    
    # Falling three methods: red (large) → small candles in range → continuation
    falling_three = (
        is_green_0 == 0.0 and body_size_0 > body_threshold and
        body_size_1 < body_threshold * 0.5 and
        body_size_2 < body_threshold * 0.5 and
        low_1 >= low_0 * 0.999 and high_1 <= high_0 * 1.001 and
        low_2 >= low_0 * 0.999 and high_2 <= high_0 * 1.001
    )
    features["pattern_falling_three_methods"] = 1.0 if falling_three else 0.0
    
    # ============================================================================
    # 12. INSIDE BAR PATTERNS (3 features: candle 2 inside candle 1)
    # ============================================================================
    
    # Inside bar: candle 2 completely inside candle 1 range
    inside_bar = (high_2 < high_1 and low_2 > low_1)
    features["pattern_inside_bar"] = 1.0 if inside_bar else 0.0
    features["pattern_inside_bar_bullish"] = 1.0 if (inside_bar and is_green_2 == 1.0) else 0.0
    features["pattern_inside_bar_bearish"] = 1.0 if (inside_bar and is_green_2 == 0.0) else 0.0
    
    # ============================================================================
    # 13. ADDITIONAL PATTERNS (3 features)
    # ============================================================================
    
    # Hanging man: red candle with long lower shadow after green candles
    hanging_man = (
        is_green_2 == 0.0 and
        lower_shadow_2 > 2 * body_size_2 and
        upper_shadow_2 < 0.3 * body_size_2 and
        body_size_2 > 0 and
        is_green_0 == 1.0 and
        is_green_1 == 1.0
    )
    features["pattern_hanging_man"] = 1.0 if hanging_man else 0.0
    
    # Tweezers top: two candles with similar highs (within 0.1%)
    tweezers_top = (
        abs(high_1 - high_2) / max(high_1, high_2) < 0.001 and
        upper_shadow_1 > 0 and
        upper_shadow_2 > 0
    )
    features["pattern_tweezers_top"] = 1.0 if tweezers_top else 0.0
    
    # Tweezers bottom: two candles with similar lows (within 0.1%)
    tweezers_bottom = (
        abs(low_1 - low_2) / max(low_1, low_2) < 0.001 and
        lower_shadow_1 > 0 and
        lower_shadow_2 > 0
    )
    features["pattern_tweezers_bottom"] = 1.0 if tweezers_bottom else 0.0
    
    return features


def _get_empty_features_dict() -> Dict[str, Optional[float]]:
    """Return dictionary with all 77 features set to None."""
    feature_names = [
        # Basic categorical (30)
        "candle_0_is_green", "candle_0_is_red", "candle_1_is_green", "candle_1_is_red",
        "candle_2_is_green", "candle_2_is_red",
        "candle_0_body_large", "candle_0_body_small", "candle_1_body_large", "candle_1_body_small",
        "candle_2_body_large", "candle_2_body_small",
        "candle_0_upper_shadow_large", "candle_0_upper_shadow_small",
        "candle_1_upper_shadow_large", "candle_1_upper_shadow_small",
        "candle_2_upper_shadow_large", "candle_2_upper_shadow_small",
        "candle_0_lower_shadow_large", "candle_0_lower_shadow_small",
        "candle_1_lower_shadow_large", "candle_1_lower_shadow_small",
        "candle_2_lower_shadow_large", "candle_2_lower_shadow_small",
        "candle_0_volume_large", "candle_0_volume_small",
        "candle_1_volume_large", "candle_1_volume_small",
        "candle_2_volume_large", "candle_2_volume_small",
        # Special patterns (15)
        "candle_0_is_doji", "candle_0_is_hammer", "candle_0_is_inverted_hammer",
        "candle_0_is_pin_bar", "candle_0_is_shooting_star",
        "candle_1_is_doji", "candle_1_is_hammer", "candle_1_is_inverted_hammer",
        "candle_1_is_pin_bar", "candle_1_is_shooting_star",
        "candle_2_is_doji", "candle_2_is_hammer", "candle_2_is_inverted_hammer",
        "candle_2_is_pin_bar", "candle_2_is_shooting_star",
        # Sequences (32)
        "pattern_all_green", "pattern_all_red", "pattern_green_red_green",
        "pattern_red_green_red", "pattern_green_green_red", "pattern_red_red_green",
        "pattern_body_increasing", "pattern_body_decreasing", "pattern_body_middle_peak",
        "pattern_body_middle_low",
        "pattern_volume_increasing", "pattern_volume_decreasing", "pattern_volume_middle_peak",
        "pattern_green_large_volume", "pattern_red_large_volume",
        "pattern_green_small_volume", "pattern_red_small_volume",
        "pattern_bullish_engulfing", "pattern_bearish_engulfing",
        "pattern_upper_shadows_increasing", "pattern_lower_shadows_increasing",
        "pattern_evening_star", "pattern_morning_star", "pattern_doji_star",
        "pattern_bullish_harami", "pattern_bearish_harami",
        "pattern_rising_three_methods", "pattern_falling_three_methods",
        "pattern_inside_bar", "pattern_inside_bar_bullish", "pattern_inside_bar_bearish",
        "pattern_hanging_man", "pattern_tweezers_top", "pattern_tweezers_bottom",
    ]
    return {name: None for name in feature_names}

