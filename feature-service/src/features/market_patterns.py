"""
Market Pattern abstraction for dynamic feature computation.

Each pattern is represented as an object that can:
- Tell what data it needs (window, segments, input sources)
- Compute its value from prepared data
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime, timedelta, timezone
import pandas as pd

from src.models.feature_registry import FeatureDefinition


@dataclass
class InputRequirement:
    """Requirements for a specific input source."""
    # For klines: base interval (e.g., 1 minute)
    base_interval: Optional[timedelta] = None
    # For klines: number of segments to divide window into
    num_segments: Optional[int] = None
    # For trades/orderflow: aggregation type
    aggregation: Optional[str] = None  # "sum", "mean", "pct_change", etc.
    # For time-based patterns: just needs timestamp
    needs_timestamp: bool = False


@dataclass
class PatternRequirements:
    """Complete requirements for a pattern computation."""
    # Total time window needed
    window: timedelta
    # Requirements per input source
    inputs: Dict[str, InputRequirement]
    # Pattern name (for debugging)
    pattern_name: str


@dataclass
class CandleSegment:
    """A single aggregated candle segment."""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    # Computed properties
    body_size: float  # abs(close - open) / open
    is_green: bool
    is_red: bool
    upper_shadow: float
    lower_shadow: float
    total_range: float


class MarketPattern(ABC):
    """
    Base class for all market patterns.
    
    Each pattern knows:
    - What feature names it supports
    - What data it needs (window, segments, input sources)
    - How to compute its value from prepared data
    """
    
    @abstractmethod
    def supports(self, feature_def: FeatureDefinition) -> bool:
        """
        Check if this pattern can compute the given feature.
        
        Args:
            feature_def: Feature definition from Registry
            
        Returns:
            True if this pattern supports the feature
        """
        pass
    
    @abstractmethod
    def get_requirements(self, feature_def: FeatureDefinition) -> PatternRequirements:
        """
        Get data requirements for computing this pattern.
        
        Args:
            feature_def: Feature definition from Registry
            
        Returns:
            PatternRequirements describing what data is needed
        """
        pass
    
    @abstractmethod
    def compute(self, feature_def: FeatureDefinition, data: Dict[str, Any]) -> Optional[float]:
        """
        Compute pattern value from prepared data.
        
        Args:
            feature_def: Feature definition from Registry
            data: Prepared data dict with keys matching input sources
                  (e.g., "klines" -> list of CandleSegment, "time" -> datetime)
            
        Returns:
            Feature value or None if computation failed
        """
        pass


def _parse_lookback_window_to_timedelta(lookback_window: str) -> timedelta:
    """
    Parse lookback_window string like '45m', '3m', '67s', '0s' into timedelta.
    
    Args:
        lookback_window: String like "45m", "67s", "1h", "2d"
        
    Returns:
        timedelta object
    """
    if not lookback_window:
        return timedelta(0)
    
    try:
        unit = lookback_window[-1]
        value = int(lookback_window[:-1])
    except (ValueError, IndexError):
        return timedelta(0)
    
    if unit == "s":
        return timedelta(seconds=value)
    elif unit == "m":
        return timedelta(minutes=value)
    elif unit == "h":
        return timedelta(hours=value)
    elif unit == "d":
        return timedelta(days=value)
    else:
        return timedelta(0)


def _determine_num_segments(window: timedelta) -> int:
    """
    Determine number of candle segments to use based on window size.
    
    Logic:
    - Very short windows (< 1 minute): 1 segment
    - Short windows (1-5 minutes): 3 segments
    - Medium windows (5-30 minutes): 3 segments (can be extended later)
    - Long windows (> 30 minutes): 3 segments (can be extended later)
    
    Args:
        window: Time window duration
        
    Returns:
        Number of segments to divide window into
    """
    total_seconds = window.total_seconds()
    
    if total_seconds < 60:  # Less than 1 minute
        return 1
    elif total_seconds <= 300:  # 1-5 minutes
        return 3
    elif total_seconds <= 1800:  # 5-30 minutes
        return 3
    else:  # > 30 minutes
        return 3  # Can be extended to 5 or more if needed


class CandlePattern(MarketPattern):
    """
    Candle pattern implementation for all candle/pattern features.
    
    Supports all candle and pattern features from Feature Registry.
    Dynamically determines number of segments based on lookback_window.
    """
    
    # All supported feature names
    SUPPORTED_FEATURES = {
        # Single candle features
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
        "candle_0_is_doji", "candle_0_is_hammer", "candle_0_is_inverted_hammer",
        "candle_0_is_pin_bar", "candle_0_is_shooting_star",
        "candle_1_is_doji", "candle_1_is_hammer", "candle_1_is_inverted_hammer",
        "candle_1_is_pin_bar", "candle_1_is_shooting_star",
        "candle_2_is_doji", "candle_2_is_hammer", "candle_2_is_inverted_hammer",
        "candle_2_is_pin_bar", "candle_2_is_shooting_star",
        # Pattern features
        "pattern_all_green", "pattern_all_red", "pattern_green_red_green",
        "pattern_red_green_red", "pattern_green_green_red", "pattern_red_red_green",
        "pattern_body_increasing", "pattern_body_decreasing", "pattern_body_middle_peak",
        "pattern_body_middle_low",
        "pattern_volume_increasing", "pattern_volume_decreasing", "pattern_volume_middle_peak",
        "pattern_green_large_volume", "pattern_red_large_volume",
        "pattern_green_small_volume", "pattern_red_small_volume",
        "pattern_bullish_engulfing", "pattern_bearish_engulfing",
        "pattern_red_green_large_body", "pattern_green_red_large_body",
        "pattern_candle_color_sequence_ternary",
        "pattern_upper_shadows_increasing", "pattern_lower_shadows_increasing",
        "pattern_evening_star", "pattern_morning_star", "pattern_doji_star",
        "pattern_bullish_harami", "pattern_bearish_harami",
        "pattern_rising_three_methods", "pattern_falling_three_methods",
        "pattern_inside_bar", "pattern_inside_bar_bullish", "pattern_inside_bar_bearish",
        "pattern_hanging_man", "pattern_tweezers_top", "pattern_tweezers_bottom",
        # Compact format features (from 15m/45m versions)
        "candle_0_body_ratio", "candle_1_body_ratio", "candle_2_body_ratio",
        "candle_0_upper_shadow_ratio", "candle_1_upper_shadow_ratio", "candle_2_upper_shadow_ratio",
        "candle_0_lower_shadow_ratio", "candle_1_lower_shadow_ratio", "candle_2_lower_shadow_ratio",
    }
    
    def supports(self, feature_def: FeatureDefinition) -> bool:
        """Check if this pattern supports the feature."""
        if feature_def.name not in self.SUPPORTED_FEATURES:
            return False
        
        # Must have kline in input_sources
        if "kline" not in feature_def.input_sources:
            return False
        
        return True
    
    def get_requirements(self, feature_def: FeatureDefinition) -> PatternRequirements:
        """Get data requirements for candle pattern."""
        window = _parse_lookback_window_to_timedelta(feature_def.lookback_window)
        num_segments = _determine_num_segments(window)
        
        return PatternRequirements(
            window=window,
            inputs={
                "klines": InputRequirement(
                    base_interval=timedelta(minutes=1),  # Always use 1-minute klines as base
                    num_segments=num_segments,
                )
            },
            pattern_name=feature_def.name,
        )
    
    def compute(self, feature_def: FeatureDefinition, data: Dict[str, Any]) -> Optional[float]:
        """
        Compute candle pattern value from prepared segments.
        
        Args:
            feature_def: Feature definition
            data: Dict with "klines" -> list of CandleSegment
            
        Returns:
            Feature value or None
        """
        if "klines" not in data:
            return None
        
        segments: List[CandleSegment] = data["klines"]
        
        # Need at least 3 segments for most patterns (some need only 1-2)
        if len(segments) < 1:
            return None
        
        # For patterns that need specific number of candles, check availability
        # Most patterns use last 3 segments (indices -3, -2, -1)
        # If we have fewer, we'll use what we have (with approximation if needed)
        
        # Get segments (use last N, where N is min(3, len(segments)))
        num_needed = min(3, len(segments))
        candle_segments = segments[-num_needed:] if len(segments) >= num_needed else segments
        
        # If we need 3 but only have fewer, approximate missing ones
        while len(candle_segments) < 3:
            if candle_segments:
                last = candle_segments[-1]
                # Approximate: use last segment's close for all OHLC, zero volume
                approximated = CandleSegment(
                    open=last.close,
                    high=last.close,
                    low=last.close,
                    close=last.close,
                    volume=0.0,
                    timestamp=last.timestamp - timedelta(minutes=1),
                    body_size=0.0,
                    is_green=False,
                    is_red=False,
                    upper_shadow=0.0,
                    lower_shadow=0.0,
                    total_range=0.0,
                )
                candle_segments.insert(0, approximated)
            else:
                return None
        
        # Now we have at least 3 segments: candle_0 (oldest), candle_1 (middle), candle_2 (newest)
        c0 = candle_segments[0]
        c1 = candle_segments[1] if len(candle_segments) > 1 else c0
        c2 = candle_segments[2] if len(candle_segments) > 2 else c1
        
        # Compute thresholds (averages over available segments)
        all_segments = candle_segments
        avg_body_size = sum(s.body_size for s in all_segments) / len(all_segments) if all_segments else 0.0
        avg_upper_shadow = sum(s.upper_shadow for s in all_segments) / len(all_segments) if all_segments else 0.0
        avg_lower_shadow = sum(s.lower_shadow for s in all_segments) / len(all_segments) if all_segments else 0.0
        avg_volume = sum(s.volume for s in all_segments) / len(all_segments) if all_segments else 0.0
        
        min_body_threshold = 0.01 / 100
        min_shadow_threshold = 0.01 / 100
        min_volume_threshold = 1e-8
        
        body_threshold = max(avg_body_size, min_body_threshold)
        upper_shadow_threshold = max(avg_upper_shadow, min_shadow_threshold)
        lower_shadow_threshold = max(avg_lower_shadow, min_shadow_threshold)
        volume_threshold = max(avg_volume, min_volume_threshold)
        
        # Route to specific feature computation
        return self._compute_feature(
            feature_def.name,
            c0, c1, c2,
            body_threshold, upper_shadow_threshold, lower_shadow_threshold, volume_threshold,
        )
    
    def _compute_feature(
        self,
        feature_name: str,
        c0: CandleSegment,
        c1: CandleSegment,
        c2: CandleSegment,
        body_threshold: float,
        upper_shadow_threshold: float,
        lower_shadow_threshold: float,
        volume_threshold: float,
    ) -> Optional[float]:
        """Compute specific feature value."""
        # Single candle features
        if feature_name == "candle_0_is_green":
            return 1.0 if c0.is_green else 0.0
        elif feature_name == "candle_0_is_red":
            return 1.0 if c0.is_red else 0.0
        elif feature_name == "candle_1_is_green":
            return 1.0 if c1.is_green else 0.0
        elif feature_name == "candle_1_is_red":
            return 1.0 if c1.is_red else 0.0
        elif feature_name == "candle_2_is_green":
            return 1.0 if c2.is_green else 0.0
        elif feature_name == "candle_2_is_red":
            return 1.0 if c2.is_red else 0.0
        
        # Body size features
        elif feature_name == "candle_0_body_large":
            return 1.0 if c0.body_size > body_threshold else 0.0
        elif feature_name == "candle_0_body_small":
            return 1.0 if c0.body_size <= body_threshold else 0.0
        elif feature_name == "candle_1_body_large":
            return 1.0 if c1.body_size > body_threshold else 0.0
        elif feature_name == "candle_1_body_small":
            return 1.0 if c1.body_size <= body_threshold else 0.0
        elif feature_name == "candle_2_body_large":
            return 1.0 if c2.body_size > body_threshold else 0.0
        elif feature_name == "candle_2_body_small":
            return 1.0 if c2.body_size <= body_threshold else 0.0
        
        # Body ratio features (compact format)
        elif feature_name == "candle_0_body_ratio":
            return c0.body_size / c0.total_range if c0.total_range > 0 else 0.0
        elif feature_name == "candle_1_body_ratio":
            return c1.body_size / c1.total_range if c1.total_range > 0 else 0.0
        elif feature_name == "candle_2_body_ratio":
            return c2.body_size / c2.total_range if c2.total_range > 0 else 0.0
        
        # Upper shadow features
        elif feature_name == "candle_0_upper_shadow_large":
            return 1.0 if c0.upper_shadow > upper_shadow_threshold else 0.0
        elif feature_name == "candle_0_upper_shadow_small":
            return 1.0 if c0.upper_shadow <= upper_shadow_threshold else 0.0
        elif feature_name == "candle_1_upper_shadow_large":
            return 1.0 if c1.upper_shadow > upper_shadow_threshold else 0.0
        elif feature_name == "candle_1_upper_shadow_small":
            return 1.0 if c1.upper_shadow <= upper_shadow_threshold else 0.0
        elif feature_name == "candle_2_upper_shadow_large":
            return 1.0 if c2.upper_shadow > upper_shadow_threshold else 0.0
        elif feature_name == "candle_2_upper_shadow_small":
            return 1.0 if c2.upper_shadow <= upper_shadow_threshold else 0.0
        
        # Upper shadow ratio features (compact format)
        elif feature_name == "candle_0_upper_shadow_ratio":
            return c0.upper_shadow / c0.total_range if c0.total_range > 0 else 0.0
        elif feature_name == "candle_1_upper_shadow_ratio":
            return c1.upper_shadow / c1.total_range if c1.total_range > 0 else 0.0
        elif feature_name == "candle_2_upper_shadow_ratio":
            return c2.upper_shadow / c2.total_range if c2.total_range > 0 else 0.0
        
        # Lower shadow features
        elif feature_name == "candle_0_lower_shadow_large":
            return 1.0 if c0.lower_shadow > lower_shadow_threshold else 0.0
        elif feature_name == "candle_0_lower_shadow_small":
            return 1.0 if c0.lower_shadow <= lower_shadow_threshold else 0.0
        elif feature_name == "candle_1_lower_shadow_large":
            return 1.0 if c1.lower_shadow > lower_shadow_threshold else 0.0
        elif feature_name == "candle_1_lower_shadow_small":
            return 1.0 if c1.lower_shadow <= lower_shadow_threshold else 0.0
        elif feature_name == "candle_2_lower_shadow_large":
            return 1.0 if c2.lower_shadow > lower_shadow_threshold else 0.0
        elif feature_name == "candle_2_lower_shadow_small":
            return 1.0 if c2.lower_shadow <= lower_shadow_threshold else 0.0
        
        # Lower shadow ratio features (compact format)
        elif feature_name == "candle_0_lower_shadow_ratio":
            return c0.lower_shadow / c0.total_range if c0.total_range > 0 else 0.0
        elif feature_name == "candle_1_lower_shadow_ratio":
            return c1.lower_shadow / c1.total_range if c1.total_range > 0 else 0.0
        elif feature_name == "candle_2_lower_shadow_ratio":
            return c2.lower_shadow / c2.total_range if c2.total_range > 0 else 0.0
        
        # Volume features
        elif feature_name == "candle_0_volume_large":
            return 1.0 if c0.volume > volume_threshold else 0.0
        elif feature_name == "candle_0_volume_small":
            return 1.0 if c0.volume <= volume_threshold else 0.0
        elif feature_name == "candle_1_volume_large":
            return 1.0 if c1.volume > volume_threshold else 0.0
        elif feature_name == "candle_1_volume_small":
            return 1.0 if c1.volume <= volume_threshold else 0.0
        elif feature_name == "candle_2_volume_large":
            return 1.0 if c2.volume > volume_threshold else 0.0
        elif feature_name == "candle_2_volume_small":
            return 1.0 if c2.volume <= volume_threshold else 0.0
        
        # Special single-candle patterns
        elif feature_name == "candle_0_is_doji":
            return 1.0 if c0.body_size < (0.05 / 100) * c0.total_range and c0.total_range > 0 else 0.0
        elif feature_name == "candle_0_is_hammer":
            return 1.0 if (c0.lower_shadow > 2 * c0.body_size and c0.upper_shadow < 0.3 * c0.body_size and c0.body_size > 0) else 0.0
        elif feature_name == "candle_1_is_doji":
            return 1.0 if c1.body_size < (0.05 / 100) * c1.total_range and c1.total_range > 0 else 0.0
        elif feature_name == "candle_1_is_hammer":
            return 1.0 if (c1.lower_shadow > 2 * c1.body_size and c1.upper_shadow < 0.3 * c1.body_size and c1.body_size > 0) else 0.0
        elif feature_name == "candle_2_is_doji":
            return 1.0 if c2.body_size < (0.05 / 100) * c2.total_range and c2.total_range > 0 else 0.0
        elif feature_name == "candle_2_is_hammer":
            return 1.0 if (c2.lower_shadow > 2 * c2.body_size and c2.upper_shadow < 0.3 * c2.body_size and c2.body_size > 0) else 0.0
        
        # Color sequence patterns
        elif feature_name == "pattern_all_green":
            return 1.0 if (c0.is_green and c1.is_green and c2.is_green) else 0.0
        elif feature_name == "pattern_all_red":
            return 1.0 if (not c0.is_green and not c1.is_green and not c2.is_green) else 0.0
        elif feature_name == "pattern_green_red_green":
            return 1.0 if (c0.is_green and not c1.is_green and c2.is_green) else 0.0
        elif feature_name == "pattern_red_green_red":
            return 1.0 if (not c0.is_green and c1.is_green and not c2.is_green) else 0.0
        
        # Body trend patterns
        elif feature_name == "pattern_body_increasing":
            return 1.0 if (c0.body_size < c1.body_size < c2.body_size) else 0.0
        elif feature_name == "pattern_body_decreasing":
            return 1.0 if (c0.body_size > c1.body_size > c2.body_size) else 0.0
        
        # Volume trend patterns
        elif feature_name == "pattern_volume_increasing":
            return 1.0 if (c0.volume < c1.volume < c2.volume) else 0.0
        elif feature_name == "pattern_volume_decreasing":
            return 1.0 if (c0.volume > c1.volume > c2.volume) else 0.0
        elif feature_name == "pattern_green_large_volume":
            return 1.0 if (c2.is_green and c2.volume > volume_threshold * 1.5) else 0.0
        elif feature_name == "pattern_red_large_volume":
            return 1.0 if (not c2.is_green and c2.volume > volume_threshold * 1.5) else 0.0
        
        # Engulfing patterns
        elif feature_name == "pattern_bullish_engulfing":
            return 1.0 if (
                c2.is_green and not c1.is_green and
                c2.open < c1.close and c2.close > c1.open
            ) else 0.0
        elif feature_name == "pattern_bearish_engulfing":
            return 1.0 if (
                not c2.is_green and c1.is_green and
                c2.open > c1.close and c2.close < c1.open
            ) else 0.0
        
        # Large body reversal patterns
        elif feature_name == "pattern_red_green_large_body":
            return 1.0 if (
                not c1.is_green and c2.is_green and
                c1.body_size > 0 and c2.body_size > c1.body_size * 1.5
            ) else 0.0
        elif feature_name == "pattern_green_red_large_body":
            return 1.0 if (
                c1.is_green and not c2.is_green and
                c1.body_size > 0 and c2.body_size > c1.body_size * 1.5
            ) else 0.0
        
        # Ternary color sequence encoding
        elif feature_name == "pattern_candle_color_sequence_ternary":
            color_code_0 = 2.0 if c0.is_green else (1.0 if c0.body_size < (0.05 / 100) * c0.total_range and c0.total_range > 0 else 0.0)
            color_code_1 = 2.0 if c1.is_green else (1.0 if c1.body_size < (0.05 / 100) * c1.total_range and c1.total_range > 0 else 0.0)
            color_code_2 = 2.0 if c2.is_green else (1.0 if c2.body_size < (0.05 / 100) * c2.total_range and c2.total_range > 0 else 0.0)
            return color_code_0 * 9.0 + color_code_1 * 3.0 + color_code_2 * 1.0
        
        # Star patterns
        elif feature_name == "pattern_morning_star":
            return 1.0 if (
                not c0.is_green and c0.body_size > body_threshold and
                c1.body_size < body_threshold * 0.3 and
                c2.is_green and c2.body_size > body_threshold and
                c2.close > (c0.open + c0.close) / 2
            ) else 0.0
        elif feature_name == "pattern_evening_star":
            return 1.0 if (
                c0.is_green and c0.body_size > body_threshold and
                c1.body_size < body_threshold * 0.3 and
                not c2.is_green and c2.body_size > body_threshold and
                c2.close < (c0.open + c0.close) / 2
            ) else 0.0
        
        # Inside bar patterns
        elif feature_name == "pattern_inside_bar":
            return 1.0 if (c2.high < c1.high and c2.low > c1.low) else 0.0
        elif feature_name == "pattern_inside_bar_bullish":
            inside = (c2.high < c1.high and c2.low > c1.low)
            return 1.0 if (inside and c2.is_green) else 0.0
        elif feature_name == "pattern_inside_bar_bearish":
            inside = (c2.high < c1.high and c2.low > c1.low)
            return 1.0 if (inside and not c2.is_green) else 0.0
        
        # Not implemented yet - return None
        else:
            return None


class TemporalPattern(MarketPattern):
    """
    Temporal pattern implementation for time-based features.
    
    Supports features like is_asia_session, is_london_session, is_ny_session.
    """
    
    SUPPORTED_FEATURES = {
        "is_asia_session",
        "is_london_session",
        "is_ny_session",
    }
    
    def supports(self, feature_def: FeatureDefinition) -> bool:
        """Check if this pattern supports the feature."""
        if feature_def.name not in self.SUPPORTED_FEATURES:
            return False
        
        # Temporal patterns don't need input sources (or have empty list)
        if feature_def.input_sources and len(feature_def.input_sources) > 0:
            # Allow if input_sources is empty or doesn't require kline/trades
            if "kline" in feature_def.input_sources or "trades" in feature_def.input_sources:
                return False
        
        return True
    
    def get_requirements(self, feature_def: FeatureDefinition) -> PatternRequirements:
        """Get data requirements for temporal pattern."""
        return PatternRequirements(
            window=timedelta(0),  # No lookback needed
            inputs={
                "time": InputRequirement(needs_timestamp=True),
            },
            pattern_name=feature_def.name,
        )
    
    def compute(self, feature_def: FeatureDefinition, data: Dict[str, Any]) -> Optional[float]:
        """Compute temporal pattern value."""
        if "time" not in data:
            return None
        
        timestamp = data["time"]
        if not isinstance(timestamp, datetime):
            return None
        
        # Ensure timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        
        hour = timestamp.hour
        
        if feature_def.name == "is_asia_session":
            # Asia session: 00:00-09:00 UTC (Tokyo/Hong Kong)
            return 1.0 if 0 <= hour < 9 else 0.0
        elif feature_def.name == "is_london_session":
            # London session: 08:00-17:00 UTC
            return 1.0 if 8 <= hour < 17 else 0.0
        elif feature_def.name == "is_ny_session":
            # NY session: 13:00-22:00 UTC
            return 1.0 if 13 <= hour < 22 else 0.0
        
        return None


def _aggregate_klines_to_segments(
    klines: pd.DataFrame,
    window: timedelta,
    num_segments: int,
    base_interval: timedelta,
    now: datetime,
) -> List[CandleSegment]:
    """
    Aggregate 1-minute klines into N segments of equal time duration.
    
    Args:
        klines: DataFrame with 1-minute klines (columns: timestamp, open, high, low, close, volume)
        window: Total time window to cover
        num_segments: Number of segments to divide window into
        base_interval: Base interval of klines (should be 1 minute)
        now: Current timestamp (end of window)
        
    Returns:
        List of CandleSegment objects (oldest first)
    """
    import structlog
    logger = structlog.get_logger(__name__)
    
    if klines.empty or num_segments < 1:
        logger.debug(
            "aggregate_klines_empty_or_invalid_segments",
            klines_empty=klines.empty,
            num_segments=num_segments,
            window_seconds=window.total_seconds(),
        )
        return []
    
    # Ensure timestamp column is datetime
    if "timestamp" not in klines.columns:
        return []
    
    if not pd.api.types.is_datetime64_any_dtype(klines["timestamp"]):
        klines = klines.copy()
        klines["timestamp"] = pd.to_datetime(klines["timestamp"], utc=True)
    
    # Ensure timezone-aware
    if klines["timestamp"].dt.tz is None:
        klines = klines.copy()
        klines["timestamp"] = klines["timestamp"].dt.tz_localize(timezone.utc)
    else:
        klines = klines.copy()
        klines["timestamp"] = klines["timestamp"].dt.tz_convert(timezone.utc)
    
    # Sort by timestamp
    klines_sorted = klines.sort_values("timestamp").reset_index(drop=True)
    
    # Filter klines within window
    window_start = now - window
    klines_in_window = klines_sorted[
        (klines_sorted["timestamp"] >= window_start) &
        (klines_sorted["timestamp"] <= now)
    ].copy()
    
    if klines_in_window.empty:
        return []
    
    # Divide window into equal segments
    segment_duration = window / num_segments
    segments = []
    
    for i in range(num_segments):
        segment_start = window_start + (segment_duration * i)
        segment_end = window_start + (segment_duration * (i + 1))
        
        # Get klines in this segment
        segment_klines = klines_in_window[
            (klines_in_window["timestamp"] >= segment_start) &
            (klines_in_window["timestamp"] < segment_end)
        ]
        
        if segment_klines.empty:
            # Approximate: use last available kline's close
            if segments:
                last_seg = segments[-1]
                approximated = CandleSegment(
                    open=last_seg.close,
                    high=last_seg.close,
                    low=last_seg.close,
                    close=last_seg.close,
                    volume=0.0,
                    timestamp=segment_start,
                    body_size=0.0,
                    is_green=False,
                    is_red=False,
                    upper_shadow=0.0,
                    lower_shadow=0.0,
                    total_range=0.0,
                )
                segments.append(approximated)
            elif not klines_in_window.empty:
                # Use last kline before window
                last_kline = klines_sorted[klines_sorted["timestamp"] < window_start]
                if not last_kline.empty:
                    last_close = float(pd.to_numeric(last_kline.iloc[-1]["close"], errors='coerce') or 0)
                    approximated = CandleSegment(
                        open=last_close,
                        high=last_close,
                        low=last_close,
                        close=last_close,
                        volume=0.0,
                        timestamp=segment_start,
                        body_size=0.0,
                        is_green=False,
                        is_red=False,
                        upper_shadow=0.0,
                        lower_shadow=0.0,
                        total_range=0.0,
                    )
                    segments.append(approximated)
                else:
                    return []  # No data at all
            else:
                return []
        else:
            # Aggregate segment klines into one candle
            open_price = float(pd.to_numeric(segment_klines.iloc[0]["open"], errors='coerce') or 0)
            high = float(pd.to_numeric(segment_klines["high"].max(), errors='coerce') or 0)
            low = float(pd.to_numeric(segment_klines["low"].min(), errors='coerce') or 0)
            close = float(pd.to_numeric(segment_klines.iloc[-1]["close"], errors='coerce') or 0)
            volume = float(pd.to_numeric(segment_klines["volume"].sum(), errors='coerce') or 0)
            
            # Safety check
            if open_price < 1e-8 or high < 1e-8 or low < 1e-8 or close < 1e-8:
                # Use approximation
                if segments:
                    last_seg = segments[-1]
                    approximated = CandleSegment(
                        open=last_seg.close,
                        high=last_seg.close,
                        low=last_seg.close,
                        close=last_seg.close,
                        volume=0.0,
                        timestamp=segment_start,
                        body_size=0.0,
                        is_green=False,
                        is_red=False,
                        upper_shadow=0.0,
                        lower_shadow=0.0,
                        total_range=0.0,
                    )
                    segments.append(approximated)
                continue
            
            # Compute derived properties
            body_size = abs(close - open_price) / open_price if open_price > 0 else 0.0
            is_green = close > open_price
            is_red = close < open_price
            total_range = (high - low) / open_price if open_price > 0 else 0.0
            upper_shadow = (high - max(open_price, close)) / open_price if open_price > 0 else 0.0
            lower_shadow = (min(open_price, close) - low) / open_price if open_price > 0 else 0.0
            
            segment = CandleSegment(
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                timestamp=segment_start,
                body_size=body_size,
                is_green=is_green,
                is_red=is_red,
                upper_shadow=upper_shadow,
                lower_shadow=lower_shadow,
                total_range=total_range,
            )
            segments.append(segment)
    
    return segments


def compute_all_patterns_dynamic(
    rolling_windows: "RollingWindows",
    feature_definitions: Sequence[Any],
    pattern_objects: Optional[List[MarketPattern]] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute all market patterns dynamically based on Feature Registry definitions.
    
    This is the main entry point that:
    1. Matches features to pattern objects
    2. Collects requirements from all patterns
    3. Groups requirements by (window, num_segments)
    4. Aggregates data once per group
    5. Computes all features
    
    Args:
        rolling_windows: RollingWindows instance with market data
        feature_definitions: List of FeatureDefinition objects (or dicts) from Registry
        pattern_objects: Optional list of pattern objects (defaults to built-in patterns)
        
    Returns:
        Dictionary of feature_name -> value (only features from Registry)
    """
    if not feature_definitions:
        return {}
    
    # Default pattern objects
    if pattern_objects is None:
        pattern_objects = [CandlePattern(), TemporalPattern()]
    
    # Match features to patterns
    feature_pattern_pairs: List[Tuple[Any, MarketPattern]] = []
    for feature_def in feature_definitions:
        # Convert dict to FeatureDefinition if needed
        if isinstance(feature_def, dict):
            try:
                from src.models.feature_registry import FeatureDefinition
                feature_def = FeatureDefinition(**feature_def)
            except Exception:
                continue
        
        # Find supporting pattern
        for pattern_obj in pattern_objects:
            if pattern_obj.supports(feature_def):
                feature_pattern_pairs.append((feature_def, pattern_obj))
                break
    
    if not feature_pattern_pairs:
        return {}
    
    # Collect requirements and group by data needs
    requirements_by_group: Dict[Tuple[timedelta, int], List[Tuple[Any, MarketPattern]]] = {}
    temporal_features: List[Tuple[Any, MarketPattern]] = []
    
    for feature_def, pattern_obj in feature_pattern_pairs:
        req = pattern_obj.get_requirements(feature_def)
        
        # Group by (window, num_segments) for klines
        if "klines" in req.inputs:
            kline_req = req.inputs["klines"]
            if kline_req.num_segments is not None:
                key = (req.window, kline_req.num_segments)
                if key not in requirements_by_group:
                    requirements_by_group[key] = []
                requirements_by_group[key].append((feature_def, pattern_obj))
        elif "time" in req.inputs:
            # Temporal patterns don't need klines
            temporal_features.append((feature_def, pattern_obj))
    
    # Prepare data for each group
    now = rolling_windows.last_update
    if not isinstance(now, datetime):
        from src.features.candle_patterns import _ensure_datetime
        now = _ensure_datetime(now)
    
    prepared_data: Dict[Tuple[timedelta, int], List[CandleSegment]] = {}
    
    for (window, num_segments), pairs in requirements_by_group.items():
        # Get klines for this window
        window_start = now - window
        klines = rolling_windows.get_klines_for_window("1m", window_start, now)
        
        # Aggregate to segments
        segments = _aggregate_klines_to_segments(
            klines=klines,
            window=window,
            num_segments=num_segments,
            base_interval=timedelta(minutes=1),
            now=now,
        )
        
        prepared_data[(window, num_segments)] = segments
    
    # Compute all features
    result: Dict[str, Optional[float]] = {}
    
    # Process candle patterns
    for feature_def, pattern_obj in feature_pattern_pairs:
        if (feature_def, pattern_obj) in temporal_features:
            continue  # Process temporal separately
        
        req = pattern_obj.get_requirements(feature_def)
        
        # Prepare data dict for this pattern
        pattern_data: Dict[str, Any] = {}
        
        if "klines" in req.inputs:
            kline_req = req.inputs["klines"]
            if kline_req.num_segments is not None:
                key = (req.window, kline_req.num_segments)
                if key in prepared_data:
                    pattern_data["klines"] = prepared_data[key]
        
        # Compute feature value
        value = pattern_obj.compute(feature_def, pattern_data)
        result[feature_def.name] = value
    
    # Process temporal patterns
    for feature_def, pattern_obj in temporal_features:
        req = pattern_obj.get_requirements(feature_def)
        pattern_data: Dict[str, Any] = {"time": now}
        value = pattern_obj.compute(feature_def, pattern_data)
        result[feature_def.name] = value
    
    return result

