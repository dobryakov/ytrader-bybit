"""
Prediction Target model.

Represents a prediction target with predicted and actual values.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class PredictionTarget(BaseModel):
    """Prediction target model."""

    id: UUID = Field(description="Prediction target UUID")
    signal_id: UUID = Field(description="Trading signal UUID")
    prediction_timestamp: datetime = Field(description="Timestamp when prediction was made")
    target_timestamp: datetime = Field(description="Timestamp when target should be evaluated")
    model_version: str = Field(description="Model version used")
    feature_registry_version: str = Field(description="Feature registry version used")
    target_registry_version: str = Field(description="Target registry version used")
    target_config: Dict[str, Any] = Field(description="Full target configuration snapshot")
    predicted_values: Dict[str, Any] = Field(description="Predicted values")
    actual_values: Optional[Dict[str, Any]] = Field(default=None, description="Actual values")
    actual_values_computed_at: Optional[datetime] = Field(default=None, description="When actual values were computed")
    actual_values_computation_error: Optional[str] = Field(default=None, description="Error message if computation failed")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")

    model_config = ConfigDict()

