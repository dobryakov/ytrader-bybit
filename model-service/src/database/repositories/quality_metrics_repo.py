"""
Model Quality Metrics database repository.

Provides CRUD operations for model_quality_metrics table.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal
from uuid import UUID
import asyncpg

from ..base import BaseRepository
from ...config.logging import get_logger

logger = get_logger(__name__)


class ModelQualityMetricsRepository(BaseRepository[Dict[str, Any]]):
    """Repository for model_quality_metrics table operations."""

    @property
    def table_name(self) -> str:
        """Return the table name."""
        return "model_quality_metrics"

    async def create(
        self,
        model_version_id: UUID,
        metric_name: str,
        metric_value: float,
        metric_type: str,
        evaluation_dataset_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new quality metric.

        Args:
            model_version_id: Associated model version UUID
            metric_name: Metric name (e.g., 'accuracy', 'precision', 'sharpe_ratio')
            metric_value: Metric value
            metric_type: Metric type ('classification', 'regression', 'trading_performance')
            evaluation_dataset_size: Number of records in evaluation dataset
            metadata: Additional metric metadata

        Returns:
            Created quality metric record

        Raises:
            DatabaseQueryError: If creation fails
        """
        query = f"""
            INSERT INTO {self.table_name} (
                model_version_id, metric_name, metric_value, metric_type,
                evaluation_dataset_size, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *
        """
        try:
            record = await self._fetchrow(
                query,
                model_version_id,
                metric_name,
                Decimal(str(metric_value)),
                metric_type,
                evaluation_dataset_size,
                metadata,
            )
            if not record:
                raise ValueError("Failed to create quality metric")
            result = self._record_to_dict(record)
            # Convert Decimal to float for JSON serialization
            if isinstance(result.get("metric_value"), Decimal):
                result["metric_value"] = float(result["metric_value"])
            logger.info(
                "Quality metric created",
                model_version_id=str(model_version_id),
                metric_name=metric_name,
                metric_value=metric_value,
            )
            return result
        except Exception as e:
            logger.error(
                "Failed to create quality metric",
                model_version_id=str(model_version_id),
                metric_name=metric_name,
                error=str(e),
                exc_info=True,
            )
            raise

    async def get_by_id(self, metric_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get quality metric by ID.

        Args:
            metric_id: Quality metric UUID

        Returns:
            Quality metric record or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE id = $1"
        record = await self._fetchrow(query, metric_id)
        if record:
            result = self._record_to_dict(record)
            # Convert Decimal to float
            if isinstance(result.get("metric_value"), Decimal):
                result["metric_value"] = float(result["metric_value"])
            return result
        return None

    async def get_by_model_version(
        self,
        model_version_id: UUID,
        metric_name: Optional[str] = None,
        metric_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get quality metrics for a model version.

        Args:
            model_version_id: Model version UUID
            metric_name: Filter by metric name (optional)
            metric_type: Filter by metric type (optional)

        Returns:
            List of quality metric records
        """
        conditions = ["model_version_id = $1"]
        params = [model_version_id]
        param_index = 2

        if metric_name:
            conditions.append(f"metric_name = ${param_index}")
            params.append(metric_name)
            param_index += 1

        if metric_type:
            conditions.append(f"metric_type = ${param_index}")
            params.append(metric_type)
            param_index += 1

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {' AND '.join(conditions)}
            ORDER BY evaluated_at DESC
        """
        records = await self._fetch(query, *params)
        results = self._records_to_dicts(records)
        # Convert Decimal to float
        for result in results:
            if isinstance(result.get("metric_value"), Decimal):
                result["metric_value"] = float(result["metric_value"])
        return results

    async def get_latest_by_model_version(
        self,
        model_version_id: UUID,
        metric_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest quality metric for a model version.

        Args:
            model_version_id: Model version UUID
            metric_name: Filter by metric name (optional)

        Returns:
            Latest quality metric record or None if not found
        """
        conditions = ["model_version_id = $1"]
        params = [model_version_id]
        param_index = 2

        if metric_name:
            conditions.append(f"metric_name = ${param_index}")
            params.append(metric_name)
            param_index += 1

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {' AND '.join(conditions)}
            ORDER BY evaluated_at DESC
            LIMIT 1
        """
        record = await self._fetchrow(query, *params)
        if record:
            result = self._record_to_dict(record)
            # Convert Decimal to float
            if isinstance(result.get("metric_value"), Decimal):
                result["metric_value"] = float(result["metric_value"])
            return result
        return None

    async def list_by_metric_name(
        self,
        metric_name: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List quality metrics by metric name.

        Args:
            metric_name: Metric name to filter by
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of quality metric records
        """
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE metric_name = $1
            ORDER BY evaluated_at DESC
        """
        if limit:
            query += f" LIMIT ${2} OFFSET ${3}"
            records = await self._fetch(query, metric_name, limit, offset)
        else:
            query += f" OFFSET ${2}"
            records = await self._fetch(query, metric_name, offset)

        results = self._records_to_dicts(records)
        # Convert Decimal to float
        for result in results:
            if isinstance(result.get("metric_value"), Decimal):
                result["metric_value"] = float(result["metric_value"])
        return results

    async def delete_by_model_version(self, model_version_id: UUID) -> int:
        """
        Delete all quality metrics for a model version.

        Args:
            model_version_id: Model version UUID

        Returns:
            Number of deleted metrics
        """
        query = f"DELETE FROM {self.table_name} WHERE model_version_id = $1"
        result = await self._execute(query, model_version_id)
        count = int(result.split()[-1]) if result else 0
        logger.info("Deleted quality metrics", model_version_id=str(model_version_id), count=count)
        return count

    async def delete(self, metric_id: UUID) -> bool:
        """
        Delete a quality metric.

        Args:
            metric_id: Quality metric UUID

        Returns:
            True if deleted, False if not found
        """
        query = f"DELETE FROM {self.table_name} WHERE id = $1 RETURNING id"
        record = await self._fetchrow(query, metric_id)
        if record:
            logger.info("Quality metric deleted", metric_id=str(metric_id))
            return True
        return False

