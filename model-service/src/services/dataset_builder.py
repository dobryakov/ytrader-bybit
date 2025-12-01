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
        check_details = {}

        # Check 1: No missing values
        missing_features = features_df.isnull().sum().sum()
        missing_labels = labels_series.isnull().sum()
        total_cells = len(features_df) * len(features_df.columns) + len(labels_series)
        missing_ratio = (missing_features + missing_labels) / total_cells if total_cells > 0 else 0.0
        missing_score = 1.0 - min(missing_ratio, 1.0)
        quality_checks.append(missing_score)
        check_details["missing_values"] = {
            "missing_features": int(missing_features),
            "missing_labels": int(missing_labels),
            "missing_ratio": float(missing_ratio),
            "score": float(missing_score),
        }

        # Check 2: No infinite values
        inf_features = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
        # For Series, check if dtype is numeric and use np.isinf directly (Series doesn't have select_dtypes)
        if pd.api.types.is_numeric_dtype(labels_series):
            inf_labels = np.isinf(labels_series).sum()
        else:
            inf_labels = 0
        inf_ratio = (inf_features + inf_labels) / total_cells if total_cells > 0 else 0.0
        inf_score = 1.0 - min(inf_ratio, 1.0)
        quality_checks.append(inf_score)
        check_details["infinite_values"] = {
            "inf_features": int(inf_features),
            "inf_labels": int(inf_labels),
            "inf_ratio": float(inf_ratio),
            "score": float(inf_score),
        }

        # Check 3: Sufficient data points
        min_records = 10  # Minimum records for meaningful training
        record_count = len(features_df)
        sufficiency_score = min(record_count / min_records, 1.0) if record_count >= min_records else 0.0
        quality_checks.append(sufficiency_score)
        check_details["sufficient_data"] = {
            "record_count": int(record_count),
            "min_records": min_records,
            "score": float(sufficiency_score),
        }

        # Check 4: Label distribution (for classification)
        if labels_series.dtype in [np.int64, int]:
            unique_labels = labels_series.nunique()
            if unique_labels > 1:
                # Check for class imbalance (penalize extreme imbalance)
                label_counts = labels_series.value_counts()
                min_class_ratio = label_counts.min() / label_counts.max()
                label_distribution_score = min_class_ratio
                quality_checks.append(label_distribution_score)
                check_details["label_distribution"] = {
                    "unique_labels": int(unique_labels),
                    "label_counts": label_counts.to_dict(),
                    "min_class_ratio": float(min_class_ratio),
                    "score": float(label_distribution_score),
                }
            else:
                quality_checks.append(0.0)  # All same label is bad
                check_details["label_distribution"] = {
                    "unique_labels": int(unique_labels),
                    "label_counts": labels_series.value_counts().to_dict(),
                    "score": 0.0,
                    "reason": "all_same_label",
                }
        else:
            # For regression, check label variance
            label_variance = labels_series.var()
            if label_variance > 0:
                label_variance_score = min(label_variance / 100.0, 1.0)
                quality_checks.append(label_variance_score)
                check_details["label_distribution"] = {
                    "label_variance": float(label_variance),
                    "score": float(label_variance_score),
                }
            else:
                quality_checks.append(0.0)  # Zero variance is bad
                check_details["label_distribution"] = {
                    "label_variance": 0.0,
                    "score": 0.0,
                    "reason": "zero_variance",
                }

        # Check 5: Feature variance (avoid constant features)
        numeric_features = features_df.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 0:
            feature_variances = numeric_features.var()
            non_zero_variance_count = (feature_variances > 1e-10).sum()
            total_features = len(feature_variances)
            non_zero_variance_ratio = non_zero_variance_count / total_features if total_features > 0 else 0.0
            quality_checks.append(non_zero_variance_ratio)
            check_details["feature_variance"] = {
                "total_features": int(total_features),
                "non_zero_variance_count": int(non_zero_variance_count),
                "ratio": float(non_zero_variance_ratio),
                "score": float(non_zero_variance_ratio),
            }
        else:
            quality_checks.append(0.0)
            check_details["feature_variance"] = {
                "total_features": 0,
                "score": 0.0,
                "reason": "no_numeric_features",
            }

        # Calculate overall quality score (weighted average)
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal weights for now
        if len(quality_checks) != len(weights):
            # Adjust weights if number of checks changed
            weights = [1.0 / len(quality_checks)] * len(quality_checks)

        quality_score = sum(w * check for w, check in zip(weights, quality_checks))

        # Log detailed quality check results
        logger.info(
            "Dataset quality validation",
            record_count=len(features_df),
            feature_count=len(features_df.columns),
            quality_score=float(quality_score),
            check_scores={
                "missing_values": float(quality_checks[0]) if len(quality_checks) > 0 else 0.0,
                "infinite_values": float(quality_checks[1]) if len(quality_checks) > 1 else 0.0,
                "sufficient_data": float(quality_checks[2]) if len(quality_checks) > 2 else 0.0,
                "label_distribution": float(quality_checks[3]) if len(quality_checks) > 3 else 0.0,
                "feature_variance": float(quality_checks[4]) if len(quality_checks) > 4 else 0.0,
            },
            check_details=check_details,
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

