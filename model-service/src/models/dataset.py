"""
Dataset models for Feature Service integration.

Represents dataset metadata and configuration for building training datasets from Feature Service.
"""

from datetime import datetime
from typing import Dict, Any, Optional, Literal
from uuid import UUID
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class DatasetStatus(str, Enum):
    """Dataset status enum."""
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


class SplitStrategy(str, Enum):
    """Dataset split strategy enum."""
    TIME_BASED = "time_based"
    WALK_FORWARD = "walk_forward"


class TargetConfig(BaseModel):
    """Target variable configuration."""
    type: Literal["regression", "classification", "risk_adjusted"] = Field(
        description="Target type"
    )
    horizon: int = Field(
        description="Prediction horizon in seconds"
    )
    threshold: Optional[float] = Field(
        default=0.005,
        description="Threshold for classification (default 0.005 = 0.5%)"
    )


class WalkForwardConfig(BaseModel):
    """Walk-forward validation configuration."""
    train_window_days: int = Field(description="Training window size in days")
    validation_window_days: int = Field(description="Validation window size in days")
    test_window_days: Optional[int] = Field(default=None, description="Test window size in days")
    step_days: int = Field(description="Step size in days")
    start_date: str = Field(description="Start date (ISO format)")
    end_date: str = Field(description="End date (ISO format)")
    min_train_samples: Optional[int] = Field(default=None, description="Minimum training samples")


class DatasetBuildRequest(BaseModel):
    """Request model for dataset build."""
    symbol: str = Field(description="Trading pair symbol (e.g., 'BTCUSDT')")
    split_strategy: SplitStrategy = Field(description="Split strategy")
    train_period_start: Optional[datetime] = Field(default=None, description="Train period start (for time_based)")
    train_period_end: Optional[datetime] = Field(default=None, description="Train period end (for time_based)")
    validation_period_start: Optional[datetime] = Field(default=None, description="Validation period start (for time_based)")
    validation_period_end: Optional[datetime] = Field(default=None, description="Validation period end (for time_based)")
    test_period_start: Optional[datetime] = Field(default=None, description="Test period start (for time_based)")
    test_period_end: Optional[datetime] = Field(default=None, description="Test period end (for time_based)")
    walk_forward_config: Optional[WalkForwardConfig] = Field(default=None, description="Walk-forward configuration (for walk_forward)")
    target_registry_version: str = Field(description="Target Registry version used")
    feature_registry_version: str = Field(description="Feature Registry version used")
    output_format: str = Field(default="parquet", description="Output format: 'parquet', 'csv', 'hdf5'")

    model_config = ConfigDict(use_enum_values=True)


class Dataset(BaseModel):
    """Dataset metadata model from Feature Service."""
    
    id: UUID = Field(description="Unique dataset identifier")
    symbol: str = Field(description="Trading pair symbol (e.g., 'BTCUSDT')")
    status: DatasetStatus = Field(description="Dataset status")
    split_strategy: SplitStrategy = Field(description="Split strategy")
    
    # Time-based split fields
    train_period_start: Optional[datetime] = Field(default=None, description="Train period start (for time_based)")
    train_period_end: Optional[datetime] = Field(default=None, description="Train period end (for time_based)")
    validation_period_start: Optional[datetime] = Field(default=None, description="Validation period start (for time_based)")
    validation_period_end: Optional[datetime] = Field(default=None, description="Validation period end (for time_based)")
    test_period_start: Optional[datetime] = Field(default=None, description="Test period start (for time_based)")
    test_period_end: Optional[datetime] = Field(default=None, description="Test period end (for time_based)")
    
    # Walk-forward split fields
    walk_forward_config: Optional[Dict[str, Any]] = Field(default=None, description="Walk-forward configuration (for walk_forward)")
    
    target_registry_version: str = Field(description="Target Registry version used")
    feature_registry_version: str = Field(description="Feature Registry version used")
    target_config: Optional[TargetConfig] = Field(default=None, description="Target configuration (loaded from Target Registry, for backward compatibility)")
    
    train_records: int = Field(default=0, description="Number of records in train split")
    validation_records: int = Field(default=0, description="Number of records in validation split")
    test_records: int = Field(default=0, description="Number of records in test split")
    
    output_format: Literal["parquet", "csv", "hdf5"] = Field(default="parquet", description="Output format")
    storage_path: Optional[str] = Field(default=None, description="Path to dataset files on filesystem")
    
    created_at: datetime = Field(description="When dataset build was requested")
    completed_at: Optional[datetime] = Field(default=None, description="When dataset building completed")
    estimated_completion: Optional[datetime] = Field(default=None, description="Estimated completion time (updated during build)")
    error_message: Optional[str] = Field(default=None, description="Error message if status is 'failed'")

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v):
        """Validate status enum."""
        if isinstance(v, str):
            try:
                return DatasetStatus(v)
            except ValueError:
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
                raise
        return v
    
    @field_validator("target_config", mode="before")
    @classmethod
    def validate_target_config(cls, v):
        """Validate target config - convert dict to TargetConfig if needed."""
        if isinstance(v, dict):
            return TargetConfig(**v)
        return v
    
    model_config = ConfigDict(use_enum_values=True)

