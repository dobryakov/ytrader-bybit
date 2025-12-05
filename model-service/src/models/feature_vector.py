"""
Feature Vector model.

Represents a computed feature vector from Feature Service.
Matches Feature Service FeatureVector structure.
"""

from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel, Field, ConfigDict


class FeatureVector(BaseModel):
    """Feature vector containing computed features for a symbol at a timestamp."""
    
    timestamp: datetime = Field(description="Timestamp when features were computed")
    symbol: str = Field(description="Trading pair symbol (e.g., 'BTCUSDT')")
    features: Dict[str, float] = Field(description="Dictionary of feature name to value")
    feature_registry_version: str = Field(description="Version of Feature Registry used")
    trace_id: Optional[str] = Field(default=None, description="Trace ID for request flow tracking")
    
    model_config = ConfigDict()

