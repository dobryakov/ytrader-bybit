"""
Model Predictions database repository.

Provides CRUD operations for model_predictions table.
Stores raw model predictions (probabilities) for test split analysis.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
import json

from ..base import BaseRepository
from ...config.logging import get_logger

logger = get_logger(__name__)


class ModelPredictionRepository(BaseRepository[Dict[str, Any]]):
    """Repository for model_predictions table operations."""

    @property
    def table_name(self) -> str:
        """Return the table name."""
        return "model_predictions"

    async def create(
        self,
        model_version: str,
        dataset_id: UUID,
        split: str,
        predictions: List[Dict[str, Any]],
        training_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new prediction record.

        Args:
            model_version: Model version string (e.g., 'v1234567890')
            dataset_id: Dataset UUID identifier
            split: Dataset split ('train', 'validation', 'test')
            predictions: List of prediction dictionaries, each containing:
                - y_true: True label (int)
                - probabilities: List of probabilities for each class [float, ...]
                - confidence: Max probability (float)
            training_id: Optional training ID for tracking
            metadata: Optional metadata (task_type, task_variant, num_classes, etc.)

        Returns:
            Created prediction record

        Raises:
            DatabaseQueryError: If creation fails
        """
        query = f"""
            INSERT INTO {self.table_name} (
                model_version, dataset_id, split, training_id, predictions, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *
        """
        try:
            # Convert predictions list to JSON string for JSONB field
            predictions_json = json.dumps(predictions)
            metadata_json = json.dumps(metadata) if metadata else None

            record = await self._fetchrow(
                query,
                model_version,
                dataset_id,
                split,
                training_id,
                predictions_json,
                metadata_json,
            )
            if not record:
                raise ValueError("Failed to create prediction record")
            result = self._record_to_dict(record)
            logger.info(
                "Prediction record created",
                model_version=model_version,
                dataset_id=str(dataset_id),
                split=split,
                prediction_count=len(predictions),
            )
            return result
        except Exception as e:
            logger.error(
                "Failed to create prediction record",
                model_version=model_version,
                dataset_id=str(dataset_id),
                split=split,
                error=str(e),
                exc_info=True,
            )
            raise

    async def get_by_model_version(
        self,
        model_version: str,
        split: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get prediction records for a model version.

        Args:
            model_version: Model version string
            split: Optional split filter ('train', 'validation', 'test')

        Returns:
            List of prediction records
        """
        conditions = ["model_version = $1"]
        params = [model_version]
        param_index = 2

        if split:
            conditions.append(f"split = ${param_index}")
            params.append(split)
            param_index += 1

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
        """
        records = await self._fetch(query, *params)
        results = self._records_to_dicts(records)
        return results

    async def get_by_dataset_id(
        self,
        dataset_id: UUID,
        split: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get prediction records for a dataset.

        Args:
            dataset_id: Dataset UUID identifier
            split: Optional split filter ('train', 'validation', 'test')

        Returns:
            List of prediction records
        """
        conditions = ["dataset_id = $1"]
        params = [dataset_id]
        param_index = 2

        if split:
            conditions.append(f"split = ${param_index}")
            params.append(split)
            param_index += 1

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
        """
        records = await self._fetch(query, *params)
        results = self._records_to_dicts(records)
        return results

    async def get_latest_by_model_version(
        self,
        model_version: str,
        split: str = "test",
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest prediction record for a model version.

        Args:
            model_version: Model version string
            split: Split to filter by (default: 'test')

        Returns:
            Latest prediction record or None if not found
        """
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE model_version = $1 AND split = $2
            ORDER BY created_at DESC
            LIMIT 1
        """
        record = await self._fetchrow(query, model_version, split)
        if record:
            return self._record_to_dict(record)
        return None

    async def delete_by_model_version(
        self,
        model_version: str,
        split: Optional[str] = None,
    ) -> int:
        """
        Delete prediction records for a model version.

        Args:
            model_version: Model version string
            split: Optional split filter

        Returns:
            Number of deleted records
        """
        conditions = ["model_version = $1"]
        params = [model_version]
        param_index = 2

        if split:
            conditions.append(f"split = ${param_index}")
            params.append(split)
            param_index += 1

        query = f"DELETE FROM {self.table_name} WHERE {' AND '.join(conditions)}"
        result = await self._execute(query, *params)
        count = int(result.split()[-1]) if result else 0
        logger.info(
            "Deleted prediction records",
            model_version=model_version,
            split=split,
            count=count,
        )
        return count

