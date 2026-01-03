"""Pydantic models for Mode entity."""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


class ModeCreate(BaseModel):
    """Request model for creating a new mode."""
    name: str = Field(..., description="Mode name", min_length=1, max_length=200)
    asset: str = Field(..., description="Trading asset symbol (e.g., BTCUSDT)", min_length=1, max_length=20)
    strategy_id: str = Field(..., description="Strategy identifier (e.g., test-strategy)", min_length=1, max_length=100)
    feature_registry_version: str = Field(..., description="Feature Registry version (e.g., 1.7.2)", min_length=1, max_length=50)
    target_registry_version: str = Field(..., description="Target Registry version (e.g., 1.4.0)", min_length=1, max_length=50)
    train_duration_days: int = Field(..., gt=0, description="Training period duration in days")
    validation_duration_days: int = Field(..., gt=0, description="Validation period duration in days")
    test_duration_days: int = Field(..., gt=0, description="Test period duration in days")
    description: Optional[str] = Field(None, description="Mode description")


class ModeUpdate(BaseModel):
    """Request model for updating a mode."""
    name: Optional[str] = Field(None, description="Mode name", min_length=1, max_length=200)
    asset: Optional[str] = Field(None, description="Trading asset symbol", min_length=1, max_length=20)
    strategy_id: Optional[str] = Field(None, description="Strategy identifier", min_length=1, max_length=100)
    feature_registry_version: Optional[str] = Field(None, description="Feature Registry version", min_length=1, max_length=50)
    target_registry_version: Optional[str] = Field(None, description="Target Registry version", min_length=1, max_length=50)
    train_duration_days: Optional[int] = Field(None, gt=0, description="Training period duration in days")
    validation_duration_days: Optional[int] = Field(None, gt=0, description="Validation period duration in days")
    test_duration_days: Optional[int] = Field(None, gt=0, description="Test period duration in days")
    description: Optional[str] = Field(None, description="Mode description")
    is_active: Optional[bool] = Field(None, description="Whether mode is active")


class ModeResponse(BaseModel):
    """Response model for mode."""
    id: UUID
    name: str
    asset: str
    strategy_id: str
    feature_registry_version: str
    target_registry_version: str
    train_duration_days: int
    validation_duration_days: int
    test_duration_days: int
    description: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]

    class Config:
        from_attributes = True


class RebuildDatasetRequest(BaseModel):
    """Request model for rebuilding dataset from mode."""
    reference_time: Optional[datetime] = Field(None, description="Reference time for period calculation (defaults to now - 1 day)")


class RebuildDatasetResponse(BaseModel):
    """Response model for rebuild dataset operation."""
    mode_id: UUID
    dataset_id: UUID
    computed_periods: dict = Field(..., description="Computed periods (train/validation/test start/end)")
    message: str

