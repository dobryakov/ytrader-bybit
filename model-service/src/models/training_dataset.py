"""
Training Dataset data model.

Represents an aggregated collection of order execution events and associated
market data organized for model training. This is a transient entity (not
persisted to database) used during training operations.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4
import pandas as pd
from pydantic import BaseModel, Field, field_validator


class TrainingDataset(BaseModel):
    """
    Training dataset data model.

    Represents an aggregated collection of order execution events and associated
    market data organized for model training. This is a transient entity used
    during training operations.
    """

    dataset_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for this dataset")
    strategy_id: str = Field(..., description="Trading strategy identifier")
    features: pd.DataFrame = Field(..., description="pandas DataFrame with feature columns")
    labels: pd.Series = Field(..., description="pandas Series with target labels")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dataset metadata (record_count, date_range, feature_names, data_quality_score, coverage, etc.)",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When dataset was created")
    source_events: Optional[List[str]] = Field(
        default=None, description="Order execution event IDs (if tracked)"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: pd.DataFrame) -> pd.DataFrame:
        """Validate features DataFrame is not empty."""
        if v.empty:
            raise ValueError("Features DataFrame cannot be empty")
        return v

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, v: pd.Series) -> pd.Series:
        """Validate labels Series is not empty."""
        if v.empty:
            raise ValueError("Labels Series cannot be empty")
        return v

    def validate_consistency(self) -> None:
        """
        Validate that features and labels have consistent dimensions.

        Raises:
            ValueError: If features and labels dimensions don't match
        """
        if len(self.features) != len(self.labels):
            raise ValueError(
                f"Features and labels must have the same length: "
                f"features={len(self.features)}, labels={len(self.labels)}"
            )

    def get_record_count(self) -> int:
        """Get the number of records in the dataset."""
        return len(self.features)

    def get_feature_names(self) -> List[str]:
        """Get the list of feature column names."""
        return list(self.features.columns)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dataset to dictionary (excluding DataFrame/Series for serialization).

        Returns:
            Dictionary representation of the dataset (without features/labels DataFrames)
        """
        return {
            "dataset_id": self.dataset_id,
            "strategy_id": self.strategy_id,
            "metadata": {
                **self.metadata,
                "record_count": self.get_record_count(),
                "feature_names": self.get_feature_names(),
            },
            "created_at": self.created_at.isoformat() + "Z",
            "source_events": self.source_events,
        }

