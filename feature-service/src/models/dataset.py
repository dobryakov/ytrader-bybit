"""
Dataset model for training datasets.
"""
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Literal
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum


class DatasetStatus(str, Enum):
    """Dataset status enum."""
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


class SplitStrategy(str, Enum):
    """Dataset split strategy enum."""
    TIME_BASED = "time_based"
    WALK_FORWARD = "walk_forward"


class TargetComputationOverrides(BaseModel):
    """Overrides for target computation preset."""
    price_source: Optional[str] = Field(
        default=None,
        description="Price source: 'close', 'open', 'high', 'low', 'price'"
    )
    future_price_source: Optional[str] = Field(
        default=None,
        description="Future price source: 'close', 'open', 'high', 'low', 'price'"
    )
    lookup_method: Optional[str] = Field(
        default=None,
        description="Lookup method: 'nearest_forward', 'nearest_backward', 'nearest', 'exact'"
    )
    tolerance_seconds: Optional[int] = Field(
        default=None,
        description="Maximum time tolerance for future price lookup (seconds)"
    )


class TargetComputationConfig(BaseModel):
    """Target computation configuration."""
    preset: Literal[
        "returns",
        "log_returns",
        "sharpe_ratio",
        "price_change",
        "volatility_normalized_std",
        "next_candle_direction",
    ] = Field(
        description="Preset computation method"
    )
    overrides: Optional[TargetComputationOverrides] = Field(
        default=None,
        description="Optional overrides for preset defaults"
    )
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional options (e.g., volatility_window for sharpe_ratio)"
    )


class TargetConfig(BaseModel):
    """Target variable configuration."""
    type: Literal["regression", "classification", "risk_adjusted"] = Field(
        description="Target type"
    )
    horizon: int = Field(
        description="Prediction horizon in seconds"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Threshold for classification (default 0.005 = 0.5%)"
    )
    computation: Optional[TargetComputationConfig] = Field(
        default=None,
        description="Target computation configuration (if not provided, uses default 'returns' preset)"
    )


class WalkForwardConfig(BaseModel):
    """Walk-forward validation configuration."""
    train_window_days: int = Field(description="Training window size in days")
    validation_window_days: int = Field(description="Validation window size in days")
    test_window_days: int = Field(description="Test window size in days")
    step_days: int = Field(description="Step size in days")
    start_date: str = Field(description="Start date (ISO format)")
    end_date: str = Field(description="End date (ISO format)")


class Dataset(BaseModel):
    """Dataset metadata model."""
    
    id: UUID = Field(description="Unique dataset identifier")
    symbol: str = Field(description="Trading pair symbol (e.g., 'BTCUSDT')")
    status: DatasetStatus = Field(description="Dataset status")
    split_strategy: SplitStrategy = Field(description="Split strategy")
    
    # Time-based split fields
    train_period_start: Optional[datetime] = Field(
        default=None,
        description="Train period start (for time_based)"
    )
    train_period_end: Optional[datetime] = Field(
        default=None,
        description="Train period end (for time_based)"
    )
    validation_period_start: Optional[datetime] = Field(
        default=None,
        description="Validation period start (for time_based)"
    )
    validation_period_end: Optional[datetime] = Field(
        default=None,
        description="Validation period end (for time_based)"
    )
    test_period_start: Optional[datetime] = Field(
        default=None,
        description="Test period start (for time_based)"
    )
    test_period_end: Optional[datetime] = Field(
        default=None,
        description="Test period end (for time_based)"
    )
    
    # Walk-forward split fields
    walk_forward_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Walk-forward configuration (for walk_forward)"
    )
    
    target_registry_version: str = Field(description="Target Registry version used")
    feature_registry_version: str = Field(description="Feature Registry version used")
    
    train_records: int = Field(default=0, description="Number of records in train split")
    validation_records: int = Field(
        default=0,
        description="Number of records in validation split"
    )
    test_records: int = Field(default=0, description="Number of records in test split")
    
    output_format: Literal["parquet", "csv", "hdf5"] = Field(
        default="parquet",
        description="Output format"
    )
    storage_path: Optional[str] = Field(
        default=None,
        description="Path to dataset files on filesystem"
    )
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When dataset build was requested"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When dataset building completed"
    )
    estimated_completion: Optional[datetime] = Field(
        default=None,
        description="Estimated completion time (updated during build)"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if status is 'failed'"
    )
    
    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v):
        """Validate status enum."""
        if isinstance(v, str):
            try:
                return DatasetStatus(v)
            except ValueError:
                # Let Pydantic handle the ValidationError
                raise
        return v
    
    @field_validator("split_strategy", mode="before")
    @classmethod
    def validate_split_strategy(cls, v):
        """Validate split strategy enum."""
        if isinstance(v, str):
            try:
                return SplitStrategy(v)
            except ValueError:
                # Let Pydantic handle the ValidationError
                raise
        return v
    
    
    @model_validator(mode='after')
    def validate_periods(self):
        """Validate periods after initialization."""
        if self.split_strategy == SplitStrategy.TIME_BASED:
            # Validate periods are specified
            if not all([
                self.train_period_start,
                self.train_period_end,
                self.validation_period_start,
                self.validation_period_end,
                self.test_period_start,
                self.test_period_end,
            ]):
                raise ValueError(
                    "All periods must be specified for time_based split strategy"
                )
            
            # Validate chronological order
            try:
                if not (
                    self.train_period_start < self.train_period_end <
                    self.validation_period_start < self.validation_period_end <
                    self.test_period_start < self.test_period_end
                ):
                    raise ValueError("Periods must be in chronological order")
            except TypeError:
                # Handle case where periods might be None or wrong type
                raise ValueError("All periods must be valid datetime objects")
        
        elif self.split_strategy == SplitStrategy.WALK_FORWARD:
            # Validate walk-forward config is specified
            if not self.walk_forward_config:
                raise ValueError(
                    "walk_forward_config must be specified for walk_forward split strategy"
                )
        
        return self
    
    model_config = ConfigDict(
        use_enum_values=True,
        # Note: json_encoders deprecated in Pydantic v2, but kept for backward compatibility
        # In production, use model_serializer instead
    )
