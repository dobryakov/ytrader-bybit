"""
Training dataset builder.

Aggregates execution events, matches them with corresponding trading signals,
applies feature engineering, generates labels, and validates dataset quality.
"""

from typing import List, Optional, Dict
from datetime import datetime
import pandas as pd
import numpy as np

from ..models.execution_event import OrderExecutionEvent
from ..models.training_dataset import TrainingDataset
from ..models.signal import MarketDataSnapshot
from ..models.position_state import OrderPositionState
from ..services.feature_engineer import feature_engineer
from ..services.label_generator import label_generator
from ..config.logging import get_logger

logger = get_logger(__name__)


class DatasetBuilder:
    """Builds training datasets from execution events and trading signals."""

    def __init__(self):
        """Initialize dataset builder."""
        pass

    def build_dataset(
        self,
        execution_events: List[OrderExecutionEvent],
        signal_market_data: Optional[Dict[str, MarketDataSnapshot]] = None,
        order_position_state: Optional[OrderPositionState] = None,
        strategy_id: Optional[str] = None,
        label_type: str = "binary",
        min_quality_score: float = 0.5,
    ) -> Optional[TrainingDataset]:
        """
        Build a training dataset from execution events.

        Args:
            execution_events: List of order execution events
            signal_market_data: Optional dictionary mapping signal_id to MarketDataSnapshot
                               (for market state at decision time)
            order_position_state: Optional order position state at event time
                                 (for open orders features in training data)
                                 Note: For accurate historical reconstruction, retrieve state
                                 for each execution event timestamp. Current implementation
                                 uses a single state for all events if provided.
            strategy_id: Trading strategy identifier (if None, inferred from events)
            label_type: Type of labels to generate ('binary', 'multi_class', 'regression')
            min_quality_score: Minimum data quality score to accept dataset

        Returns:
            TrainingDataset or None if quality check fails
        """
        if not execution_events:
            logger.warning("No execution events provided for dataset building")
            return None

        # Infer strategy_id from events if not provided
        if strategy_id is None:
            strategy_ids = {event.strategy_id for event in execution_events}
            if len(strategy_ids) > 1:
                logger.warning("Multiple strategy IDs in events, using first one", strategy_ids=list(strategy_ids))
            strategy_id = list(strategy_ids)[0] if strategy_ids else "unknown"

        logger.info("Building training dataset", strategy_id=strategy_id, event_count=len(execution_events))

        # Apply feature engineering (includes open orders features if order_position_state provided)
        try:
            features_df = feature_engineer.engineer_features(
                execution_events,
                signal_market_data,
                order_position_state=order_position_state,
            )
            if features_df.empty:
                logger.error("Feature engineering produced empty DataFrame")
                return None
        except Exception as e:
            logger.error("Feature engineering failed", error=str(e), exc_info=True)
            return None

        # Generate labels
        try:
            labels_series = label_generator.generate_labels(execution_events, label_type=label_type)
            if labels_series.empty:
                logger.error("Label generation produced empty Series")
                return None
        except Exception as e:
            logger.error("Label generation failed", error=str(e), exc_info=True)
            return None

        # Validate consistency
        if len(features_df) != len(labels_series):
            logger.error(
                "Features and labels have inconsistent lengths",
                features_count=len(features_df),
                labels_count=len(labels_series),
            )
            return None

        # Validate dataset quality
        quality_score = self._validate_dataset_quality(features_df, labels_series)
        if quality_score < min_quality_score:
            logger.warning(
                "Dataset quality below threshold",
                quality_score=quality_score,
                min_quality_score=min_quality_score,
            )
            return None

        # Build metadata
        metadata = self._build_metadata(execution_events, features_df, labels_series, quality_score)

        # Create training dataset
        dataset = TrainingDataset(
            strategy_id=strategy_id,
            features=features_df,
            labels=labels_series,
            metadata=metadata,
            source_events=[event.event_id for event in execution_events],
        )

        # Validate dataset consistency
        dataset.validate_consistency()

        logger.info(
            "Training dataset built successfully",
            dataset_id=dataset.dataset_id,
            strategy_id=strategy_id,
            record_count=dataset.get_record_count(),
            feature_count=len(dataset.get_feature_names()),
            quality_score=quality_score,
        )

        return dataset

    def _validate_dataset_quality(self, features_df: pd.DataFrame, labels_series: pd.Series) -> float:
        """
        Validate dataset quality and return a quality score.

        Args:
            features_df: Features DataFrame
            labels_series: Labels Series

        Returns:
            Quality score between 0 and 1
        """
        quality_checks = []

        # Check 1: No missing values
        missing_features = features_df.isnull().sum().sum()
        missing_labels = labels_series.isnull().sum()
        missing_ratio = (missing_features + missing_labels) / (len(features_df) * len(features_df.columns) + len(labels_series))
        quality_checks.append(1.0 - min(missing_ratio, 1.0))  # Penalize missing values

        # Check 2: No infinite values
        inf_features = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
        # For Series, check if dtype is numeric and use np.isinf directly (Series doesn't have select_dtypes)
        if pd.api.types.is_numeric_dtype(labels_series):
            inf_labels = np.isinf(labels_series).sum()
        else:
            inf_labels = 0
        inf_ratio = (inf_features + inf_labels) / (len(features_df) * len(features_df.columns) + len(labels_series))
        quality_checks.append(1.0 - min(inf_ratio, 1.0))  # Penalize infinite values

        # Check 3: Sufficient data points
        min_records = 10  # Minimum records for meaningful training
        sufficiency_score = min(len(features_df) / min_records, 1.0) if len(features_df) >= min_records else 0.0
        quality_checks.append(sufficiency_score)

        # Check 4: Label distribution (for classification)
        if labels_series.dtype in [np.int64, int]:
            unique_labels = labels_series.nunique()
            if unique_labels > 1:
                # Check for class imbalance (penalize extreme imbalance)
                label_counts = labels_series.value_counts()
                min_class_ratio = label_counts.min() / label_counts.max()
                quality_checks.append(min_class_ratio)  # Prefer balanced classes
            else:
                quality_checks.append(0.0)  # All same label is bad
        else:
            # For regression, check label variance
            label_variance = labels_series.var()
            if label_variance > 0:
                quality_checks.append(min(label_variance / 100.0, 1.0))  # Normalize variance
            else:
                quality_checks.append(0.0)  # Zero variance is bad

        # Check 5: Feature variance (avoid constant features)
        feature_variances = features_df.select_dtypes(include=[np.number]).var()
        non_zero_variance_ratio = (feature_variances > 1e-10).sum() / len(feature_variances) if len(feature_variances) > 0 else 0.0
        quality_checks.append(non_zero_variance_ratio)

        # Calculate overall quality score (weighted average)
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal weights for now
        quality_score = sum(w * check for w, check in zip(weights, quality_checks))

        logger.debug(
            "Dataset quality validation",
            quality_score=quality_score,
            checks=quality_checks,
            missing_ratio=missing_ratio,
            inf_ratio=inf_ratio,
        )

        return quality_score

    def _build_metadata(
        self,
        execution_events: List[OrderExecutionEvent],
        features_df: pd.DataFrame,
        labels_series: pd.Series,
        quality_score: float,
    ) -> Dict:
        """
        Build metadata for the training dataset.

        Args:
            execution_events: List of execution events
            features_df: Features DataFrame
            labels_series: Labels Series
            quality_score: Dataset quality score

        Returns:
            Metadata dictionary
        """
        # Calculate date range
        timestamps = [event.executed_at for event in execution_events]
        if timestamps:
            min_timestamp = min(timestamps)
            max_timestamp = max(timestamps)
        else:
            min_timestamp = datetime.utcnow()
            max_timestamp = datetime.utcnow()

        metadata = {
            "record_count": len(features_df),
            "date_range": {
                "start": min_timestamp.isoformat() + "Z",
                "end": max_timestamp.isoformat() + "Z",
            },
            "feature_names": list(features_df.columns),
            "data_quality_score": quality_score,
            "coverage": {
                "assets": list({event.asset for event in execution_events}),
                "strategies": list({event.strategy_id for event in execution_events}),
            },
            "label_statistics": {
                "mean": float(labels_series.mean()) if labels_series.dtype in [np.float64, np.int64] else None,
                "std": float(labels_series.std()) if labels_series.dtype in [np.float64, np.int64] else None,
                "min": float(labels_series.min()) if labels_series.dtype in [np.float64, np.int64] else None,
                "max": float(labels_series.max()) if labels_series.dtype in [np.float64, np.int64] else None,
                "unique_values": int(labels_series.nunique()) if labels_series.dtype in [np.int64, int] else None,
            },
            "feature_statistics": {
                "mean_missing_ratio": float(features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns))),
                "mean_variance": float(features_df.select_dtypes(include=[np.number]).var().mean()),
            },
        }

        return metadata


# Global dataset builder instance
dataset_builder = DatasetBuilder()

