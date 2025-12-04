"""
Feature Vector model.
"""
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class FeatureVector(BaseModel):
    """Feature vector containing computed features for a symbol at a timestamp."""
    
    timestamp: datetime = Field(description="Timestamp when features were computed")
    symbol: str = Field(description="Trading pair symbol (e.g., 'BTCUSDT')")
    features: Dict[str, float] = Field(description="Dictionary of feature name to value")
    feature_registry_version: str = Field(description="Version of Feature Registry used")
    trace_id: Optional[str] = Field(default=None, description="Trace ID for request flow tracking")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

