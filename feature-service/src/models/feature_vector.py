"""
Feature Vector model.
"""
import math
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict, model_serializer


class FeatureVector(BaseModel):
    """Feature vector containing computed features for a symbol at a timestamp."""
    
    timestamp: datetime = Field(description="Timestamp when features were computed")
    symbol: str = Field(description="Trading pair symbol (e.g., 'BTCUSDT')")
    features: Dict[str, float] = Field(description="Dictionary of feature name to value")
    feature_registry_version: str = Field(description="Version of Feature Registry used")
    trace_id: Optional[str] = Field(default=None, description="Trace ID for request flow tracking")
    
    model_config = ConfigDict()
    # Note: json_encoders deprecated in Pydantic v2
    # datetime serialization handled automatically by Pydantic
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Serialize model, replacing NaN/Inf values with None for JSON compatibility."""
        result = {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "features": {},
            "feature_registry_version": self.feature_registry_version,
        }
        if self.trace_id is not None:
            result["trace_id"] = self.trace_id
        
        # Filter out NaN and Inf values (replace with None, which will be excluded from JSON)
        for key, value in self.features.items():
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    # Skip NaN/Inf values - they are not JSON compliant
                    continue
            result["features"][key] = value
        
        return result

