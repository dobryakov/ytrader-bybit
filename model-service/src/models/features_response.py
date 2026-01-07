"""
Features Response model.

Response from /features/latest endpoint containing both feature vector and market data.
Matches Feature Service FeaturesResponse structure.
"""
from pydantic import BaseModel, Field

from .feature_vector import FeatureVector
from .market_data import MarketData


class FeaturesResponse(BaseModel):
    """Response containing feature vector and market data."""
    
    feature_vector: FeatureVector = Field(..., description="Computed feature vector")
    market_data: MarketData = Field(..., description="Current market data snapshot")

