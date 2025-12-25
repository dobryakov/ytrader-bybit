"""
Prediction Target database repository.

Provides CRUD operations for prediction_targets table.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from uuid import UUID
import asyncpg
import json
from decimal import Decimal

from ..base import BaseRepository
from ...config.logging import get_logger

logger = get_logger(__name__)


class PredictionTargetRepository(BaseRepository[Dict[str, Any]]):
    """Repository for prediction_targets table operations."""

    @property
    def table_name(self) -> str:
        """Return the table name."""
        return "prediction_targets"

    async def create(
        self,
        signal_id: str,
        prediction_timestamp: datetime,
        target_timestamp: datetime,
        model_version: str,
        feature_registry_version: str,
        target_registry_version: str,
        target_config: Dict[str, Any],
        predicted_values: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a new prediction target.

        Args:
            signal_id: Trading signal UUID
            prediction_timestamp: Timestamp when prediction was made
            target_timestamp: Timestamp when target should be evaluated
            model_version: Model version used
            feature_registry_version: Feature registry version used
            target_registry_version: Target registry version used
            target_config: Full target configuration snapshot (JSONB)
            predicted_values: Predicted values (JSONB)

        Returns:
            Created prediction target record

        Raises:
            DatabaseQueryError: If creation fails
        """
        query = f"""
            INSERT INTO {self.table_name} (
                signal_id, prediction_timestamp, target_timestamp,
                model_version, feature_registry_version, target_registry_version,
                target_config, predicted_values
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *
        """
        try:
            signal_uuid = UUID(signal_id) if isinstance(signal_id, str) else signal_id
            
            # Normalize all datetime objects to timezone-aware UTC, then to naive for PostgreSQL timestamp without time zone
            # This prevents asyncpg errors when comparing datetime objects
            if prediction_timestamp.tzinfo is None:
                prediction_timestamp = prediction_timestamp.replace(tzinfo=timezone.utc)
            else:
                prediction_timestamp = prediction_timestamp.astimezone(timezone.utc)
            prediction_timestamp = prediction_timestamp.replace(tzinfo=None)  # Convert to naive for PostgreSQL
            
            if target_timestamp.tzinfo is None:
                target_timestamp = target_timestamp.replace(tzinfo=timezone.utc)
            else:
                target_timestamp = target_timestamp.astimezone(timezone.utc)
            target_timestamp = target_timestamp.replace(tzinfo=None)  # Convert to naive for PostgreSQL
            
            # Convert dicts to JSON strings for JSONB columns
            target_config_json = json.dumps(target_config)
            predicted_values_json = json.dumps(predicted_values)
            
            record = await self._fetchrow(
                query,
                signal_uuid,
                prediction_timestamp,
                target_timestamp,
                model_version,
                feature_registry_version,
                target_registry_version,
                target_config_json,
                predicted_values_json,
            )
            if not record:
                raise ValueError("Failed to create prediction target")
            result = self._record_to_dict(record)
            logger.info(
                "Prediction target created",
                signal_id=signal_id,
                target_timestamp=target_timestamp.isoformat(),
                model_version=model_version,
            )
            return result
        except Exception as e:
            logger.error(
                "Failed to create prediction target",
                signal_id=signal_id,
                error=str(e),
                exc_info=True,
            )
            raise

    async def get_by_id(self, prediction_target_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get prediction target by ID.

        Args:
            prediction_target_id: Prediction target UUID

        Returns:
            Prediction target record or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE id = $1"
        record = await self._fetchrow(query, prediction_target_id)
        return self._record_to_dict(record) if record else None

    async def get_by_signal_id(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get prediction target by signal_id.

        Args:
            signal_id: Trading signal identifier

        Returns:
            Prediction target record or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE signal_id = $1 ORDER BY prediction_timestamp DESC LIMIT 1"
        signal_uuid = UUID(signal_id) if isinstance(signal_id, str) else signal_id
        record = await self._fetchrow(query, signal_uuid)
        return self._record_to_dict(record) if record else None

    async def get_pending_computations(
        self,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get prediction targets pending actual values computation.

        Args:
            limit: Maximum number of results

        Returns:
            List of prediction targets pending computation
        """
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE target_timestamp <= NOW()
              AND (actual_values IS NULL OR actual_values = '{{}}'::jsonb)
              AND (actual_values_computed_at IS NULL 
                   OR (actual_values_computation_error IS NOT NULL 
                       AND actual_values_computed_at < NOW() - INTERVAL '1 hour'))
              AND (is_obsolete IS NULL OR is_obsolete = FALSE)
            ORDER BY target_timestamp ASC
        """
        if limit:
            query += f" LIMIT ${1}"
            records = await self._fetch(query, limit)
        else:
            records = await self._fetch(query)
        results = self._records_to_dicts(records)
        
        # Ensure JSONB fields are properly parsed as dicts
        # asyncpg should parse JSONB automatically, but handle edge cases
        for result in results:
            for jsonb_field in ["target_config", "predicted_values", "actual_values"]:
                if jsonb_field in result and result[jsonb_field] is not None:
                    value = result[jsonb_field]
                    if isinstance(value, str):
                        try:
                            result[jsonb_field] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(
                                "Failed to parse JSONB field",
                                field=jsonb_field,
                                prediction_target_id=str(result.get("id")),
                            )
                            result[jsonb_field] = {}
        
        return results

    async def update_actual_values(
        self,
        prediction_target_id: UUID,
        actual_values: Dict[str, Any],
        computation_error: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update actual values for a prediction target.

        Args:
            prediction_target_id: Prediction target UUID
            actual_values: Actual values (JSONB)
            computation_error: Error message if computation failed (optional)

        Returns:
            Updated prediction target record or None if not found
        """
        query = f"""
            UPDATE {self.table_name}
            SET actual_values = $1,
                actual_values_computed_at = $2,
                actual_values_computation_error = $3,
                updated_at = $2
            WHERE id = $4
            RETURNING *
        """
        try:
            actual_values_json = json.dumps(actual_values) if actual_values else None
            computed_at = datetime.utcnow()
            
            record = await self._fetchrow(
                query,
                actual_values_json,
                computed_at,
                computation_error,
                prediction_target_id,
            )
            if record:
                result = self._record_to_dict(record)
                logger.info(
                    "Prediction target actual values updated",
                    prediction_target_id=str(prediction_target_id),
                    has_error=computation_error is not None,
                )
                return result
            return None
        except Exception as e:
            logger.error(
                "Failed to update actual values",
                prediction_target_id=str(prediction_target_id),
                error=str(e),
                exc_info=True,
            )
            raise

    async def list_by_model_version(
        self,
        model_version: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List prediction targets by model version.

        Args:
            model_version: Model version identifier
            start_time: Start time filter (optional)
            end_time: End time filter (optional)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of prediction target records
        """
        conditions = ["model_version = $1"]
        params = [model_version]
        param_index = 2

        if start_time:
            conditions.append(f"prediction_timestamp >= ${param_index}")
            params.append(start_time)
            param_index += 1

        if end_time:
            conditions.append(f"prediction_timestamp <= ${param_index}")
            params.append(end_time)
            param_index += 1

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {' AND '.join(conditions)}
            ORDER BY prediction_timestamp DESC
        """
        if limit:
            query += f" LIMIT ${param_index} OFFSET ${param_index + 1}"
            params.extend([limit, offset])
        else:
            query += f" OFFSET ${param_index}"
            params.append(offset)

        records = await self._fetch(query, *params)
        return self._records_to_dicts(records)

    async def mark_obsolete(
        self,
        age_days: int,
        limit: Optional[int] = None,
    ) -> int:
        """
        Mark very old prediction targets as obsolete.

        Args:
            age_days: Age in days to consider targets as obsolete
            limit: Maximum number of targets to mark (optional)

        Returns:
            Number of targets marked as obsolete
        """
        if limit:
            # Use subquery to limit results
            query = f"""
                UPDATE {self.table_name}
                SET is_obsolete = TRUE,
                    updated_at = NOW()
                WHERE id IN (
                    SELECT id FROM {self.table_name}
                    WHERE target_timestamp < NOW() - INTERVAL '{age_days} days'
                      AND (is_obsolete IS NULL OR is_obsolete = FALSE)
                      AND (actual_values IS NULL OR actual_values = '{{}}'::jsonb)
                    LIMIT {limit}
                )
                RETURNING id
            """
        else:
            query = f"""
                UPDATE {self.table_name}
                SET is_obsolete = TRUE,
                    updated_at = NOW()
                WHERE target_timestamp < NOW() - INTERVAL '{age_days} days'
                  AND (is_obsolete IS NULL OR is_obsolete = FALSE)
                  AND (actual_values IS NULL OR actual_values = '{{}}'::jsonb)
                RETURNING id
            """
        
        try:
            records = await self._fetch(query)
            count = len(records) if records else 0
            if count > 0:
                logger.info(
                    "Prediction targets marked as obsolete",
                    count=count,
                    age_days=age_days,
                )
            return count
        except Exception as e:
            logger.error(
                "Failed to mark prediction targets as obsolete",
                age_days=age_days,
                error=str(e),
                exc_info=True,
            )
            raise

    async def mark_target_obsolete(
        self,
        prediction_target_id: UUID,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Mark a specific prediction target as obsolete.

        Args:
            prediction_target_id: Prediction target UUID
            reason: Optional reason for marking as obsolete

        Returns:
            True if target was marked, False if not found
        """
        query = f"""
            UPDATE {self.table_name}
            SET is_obsolete = TRUE,
                updated_at = NOW()
            WHERE id = $1
            RETURNING id
        """
        try:
            record = await self._fetchrow(query, prediction_target_id)
            if record:
                logger.info(
                    "Prediction target marked as obsolete",
                    prediction_target_id=str(prediction_target_id),
                    reason=reason,
                )
                return True
            return False
        except Exception as e:
            logger.error(
                "Failed to mark prediction target as obsolete",
                prediction_target_id=str(prediction_target_id),
                error=str(e),
                exc_info=True,
            )
            raise

