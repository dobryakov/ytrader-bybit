"""
Feature Registry model for feature configuration.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from src.logging import get_logger


class DataSource(BaseModel):
    """Data source configuration for a feature."""
    
    source: str = Field(description="Data source name (e.g., 'orderbook', 'kline', 'trades')")
    timestamp_required: bool = Field(
        default=True,
        description="Whether timestamp is required for this data source"
    )


class FeatureDefinition(BaseModel):
    """Feature definition in Feature Registry."""
    
    name: str = Field(description="Feature name")
    input_sources: List[str] = Field(
        description="List of input data sources required for this feature"
    )
    lookback_window: str = Field(
        description="Lookback window duration (e.g., '0s', '1s', '1m', '5m')"
    )
    lookahead_forbidden: bool = Field(
        default=True,
        description="Whether lookahead (future data) is forbidden for this feature"
    )
    max_lookback_days: int = Field(
        default=0,
        description="Maximum lookback period in days (0 = no limit, used for data leakage prevention)"
    )
    data_sources: Optional[List[DataSource]] = Field(
        default=None,
        description="Detailed data source configurations"
    )
    
    @field_validator("lookback_window")
    @classmethod
    def validate_lookback_window(cls, v: str) -> str:
        """Validate lookback window format."""
        if not isinstance(v, str):
            raise ValueError("lookback_window must be a string")
        
        # Basic format validation: number + unit (s, m, h, d)
        if not v:
            raise ValueError("lookback_window cannot be empty")
        
        # Check for negative values (data leakage)
        if v.startswith("-"):
            raise ValueError("lookback_window cannot be negative (data leakage)")
        
        # Validate format: number followed by unit
        unit = v[-1] if v else ""
        if unit not in ["s", "m", "h", "d"]:
            raise ValueError(
                f"lookback_window unit must be one of: s, m, h, d. Got: {unit}"
            )
        
        try:
            value = int(v[:-1])
            if value < 0:
                raise ValueError("lookback_window value cannot be negative")
        except ValueError:
            raise ValueError(f"lookback_window must be in format '<number><unit>', got: {v}")
        
        return v
    
    @field_validator("max_lookback_days")
    @classmethod
    def validate_max_lookback_days(cls, v: int) -> int:
        """Validate max_lookback_days."""
        if v < 0:
            raise ValueError("max_lookback_days cannot be negative")
        return v
    
    @model_validator(mode='after')
    def validate_temporal_boundaries(self):
        """Validate temporal boundaries to prevent data leakage."""
        # Check for lookahead violations
        if not self.lookahead_forbidden:
            raise ValueError(
                f"Feature '{self.name}': lookahead_forbidden must be True to prevent data leakage"
            )
        
        # Check for excessive lookback
        if self.max_lookback_days > 90:
            raise ValueError(
                f"Feature '{self.name}': max_lookback_days ({self.max_lookback_days}) exceeds "
                f"recommended limit of 90 days (data leakage risk)"
            )
        
        return self


class FeatureRegistryConfig(BaseModel):
    """Feature Registry configuration options."""
    
    timestamp_interval_minutes: Optional[int] = Field(
        default=1,
        description="Timestamp interval in minutes for dataset generation. Default: 1 minute. Timestamps are aligned to interval boundaries (e.g., 00:00, 05:00, 10:00 for 5-minute intervals)."
    )
    
    @field_validator("timestamp_interval_minutes")
    @classmethod
    def validate_timestamp_interval_minutes(cls, v: Optional[int]) -> int:
        """Validate timestamp_interval_minutes."""
        if v is None:
            return 1
        if v <= 0:
            raise ValueError("timestamp_interval_minutes must be positive")
        if v > 60:
            raise ValueError("timestamp_interval_minutes should not exceed 60 minutes")
        return v


class FeatureRegistry(BaseModel):
    """Feature Registry configuration model."""
    
    version: str = Field(description="Feature Registry version identifier")
    config: Optional[FeatureRegistryConfig] = Field(
        default=None,
        description="Optional configuration for dataset generation and other registry-level settings"
    )
    features: List[FeatureDefinition] = Field(
        description="List of feature definitions"
    )
    
    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version format."""
        if not isinstance(v, str) or not v:
            raise ValueError("version must be a non-empty string")
        return v
    
    @field_validator("features")
    @classmethod
    def validate_features(cls, v: List[Any]) -> List[FeatureDefinition]:
        """Validate features list."""
        if not isinstance(v, list):
            raise ValueError("features must be a list")
        
        if len(v) == 0:
            raise ValueError("features list cannot be empty")
        
        # Convert dicts to FeatureDefinition objects
        validated_features = []
        for feature in v:
            if isinstance(feature, dict):
                validated_features.append(FeatureDefinition(**feature))
            elif isinstance(feature, FeatureDefinition):
                validated_features.append(feature)
            else:
                raise ValueError(f"Invalid feature definition type: {type(feature)}")
        
        # Check for duplicate feature names
        feature_names = [f.name for f in validated_features]
        if len(feature_names) != len(set(feature_names)):
            duplicates = [
                name for name in feature_names
                if feature_names.count(name) > 1
            ]
            raise ValueError(f"Duplicate feature names found: {set(duplicates)}")
        
        return validated_features
    
    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """
        Get a feature definition by name.
        
        Args:
            name: Feature name
            
        Returns:
            FeatureDefinition if found, None otherwise
        """
        for feature in self.features:
            if feature.name == name:
                return feature
        return None
    
    def get_required_data_types(self) -> set:
        """
        Get set of required data types from all features.
        
        Returns:
            Set of required data types
        """
        data_types = set()
        for feature in self.features:
            data_types.update(feature.input_sources)
        return data_types
    
    def get_timestamp_interval_minutes(self) -> int:
        """
        Get timestamp interval in minutes for dataset generation.
        
        Returns:
            Timestamp interval in minutes (default: 1)
        """
        if self.config and self.config.timestamp_interval_minutes:
            return self.config.timestamp_interval_minutes
        return 1
    
    def validate_timestamp_interval_compatibility(self) -> tuple[bool, Optional[str]]:
        """
        Validate that timestamp_interval_minutes is compatible with feature lookback windows.
        
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if interval is compatible, False otherwise
            - error_message: Error message if incompatible, None otherwise
        """
        interval_minutes = self.get_timestamp_interval_minutes()
        
        # Parse all lookback windows to find minimum required interval
        min_lookback_minutes = None
        
        for feature in self.features:
            lookback_window = feature.lookback_window
            
            # Parse lookback_window (e.g., "15m" -> 15, "3s" -> 0.05, "1h" -> 60)
            if not lookback_window:
                continue
            
            unit = lookback_window[-1]
            try:
                value = int(lookback_window[:-1])
            except (ValueError, IndexError):
                continue
            
            # Convert to minutes
            if unit == "s":
                lookback_minutes = value / 60.0
            elif unit == "m":
                lookback_minutes = value
            elif unit == "h":
                lookback_minutes = value * 60
            elif unit == "d":
                lookback_minutes = value * 24 * 60
            else:
                continue
            
            # Track minimum lookback (most restrictive)
            if min_lookback_minutes is None or lookback_minutes < min_lookback_minutes:
                min_lookback_minutes = lookback_minutes
        
        # If no lookback windows found, allow any interval
        if min_lookback_minutes is None:
            return True, None
        
        # Rule: timestamp_interval_minutes should not exceed min_lookback_minutes
        # This ensures we have enough granularity for features with small lookback windows
        # However, for features with large lookback windows (e.g., 15m), we can use larger intervals
        # So we allow interval up to min_lookback_minutes, but warn if it's too large
        
        if interval_minutes > min_lookback_minutes:
            return False, (
                f"timestamp_interval_minutes ({interval_minutes}) exceeds minimum "
                f"lookback_window ({min_lookback_minutes:.2f} minutes). "
                f"This may cause loss of granularity for features with small lookback windows."
            )
        
        # Warn if interval is too large relative to lookback (but don't fail)
        if interval_minutes > min_lookback_minutes * 0.5:
            logger = get_logger(__name__)
            logger.warning(
                "timestamp_interval_large_relative_to_lookback",
                interval_minutes=interval_minutes,
                min_lookback_minutes=min_lookback_minutes,
                message="Timestamp interval is large relative to minimum lookback window",
            )
        
        return True, None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Feature Registry to dictionary.
        
        Returns:
            Dictionary representation
        """
        result = {
            "version": self.version,
            "features": [feature.model_dump() for feature in self.features]
        }
        if self.config:
            result["config"] = self.config.model_dump()
        return result
    
    model_config = ConfigDict(
        use_enum_values=True,
    )

