"""
Execution Event database repository.

Provides CRUD operations for execution_events table.
"""

from typing import Optional, List, Dict, Any, Sequence
from datetime import datetime, timezone
from uuid import UUID
import asyncpg
import json
from decimal import Decimal

from ..base import BaseRepository
from ...config.logging import get_logger

logger = get_logger(__name__)


class ExecutionEventRepository(BaseRepository[Dict[str, Any]]):
    """Repository for execution_events table operations."""

    @property
    def table_name(self) -> str:
        """Return the table name."""
        return "execution_events"

    async def create(
        self,
        signal_id: str,
        strategy_id: str,
        asset: str,
        side: str,
        execution_price: float,
        execution_quantity: float,
        execution_fees: float,
        executed_at: datetime,
        signal_price: float,
        signal_timestamp: datetime,
        performance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a new execution event.

        Args:
            signal_id: Original trading signal identifier
            strategy_id: Trading strategy identifier
            asset: Trading pair
            side: Order side ('buy' or 'sell')
            execution_price: Actual execution price
            execution_quantity: Executed quantity
            execution_fees: Total fees paid
            executed_at: Execution timestamp
            signal_price: Original signal price
            signal_timestamp: Original signal timestamp
            performance: Performance metrics dictionary

        Returns:
            Created execution event record

        Raises:
            DatabaseQueryError: If creation fails
        """
        query = f"""
            INSERT INTO {self.table_name} (
                signal_id, strategy_id, asset, side,
                execution_price, execution_quantity, execution_fees,
                executed_at, signal_price, signal_timestamp, performance
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING *
        """
        try:
            # Convert to UTC timezone-aware first, then to naive for PostgreSQL timestamp without time zone
            if executed_at.tzinfo is None:
                executed_at = executed_at.replace(tzinfo=timezone.utc)
            else:
                executed_at = executed_at.astimezone(timezone.utc)
            # Convert to naive datetime for PostgreSQL timestamp without time zone
            executed_at = executed_at.replace(tzinfo=None)
            
            if signal_timestamp.tzinfo is None:
                signal_timestamp = signal_timestamp.replace(tzinfo=timezone.utc)
            else:
                signal_timestamp = signal_timestamp.astimezone(timezone.utc)
            # Convert to naive datetime for PostgreSQL timestamp without time zone
            signal_timestamp = signal_timestamp.replace(tzinfo=None)
            
            logger.debug(
                "Inserting execution event",
                signal_id=signal_id,
                executed_at_iso=executed_at.isoformat(),
                signal_timestamp_iso=signal_timestamp.isoformat(),
            )
            
            # Serialize performance dict to JSON string for JSONB column
            performance_json = json.dumps(performance) if isinstance(performance, dict) else performance
            
            record = await self._fetchrow(
                query,
                UUID(signal_id) if isinstance(signal_id, str) else signal_id,
                strategy_id,
                asset,
                side,
                Decimal(str(execution_price)),
                Decimal(str(execution_quantity)),
                Decimal(str(execution_fees)),
                executed_at,
                Decimal(str(signal_price)),
                signal_timestamp,
                performance_json,
            )
            if not record:
                raise ValueError("Failed to create execution event")
            result = self._record_to_dict(record)
            # Convert Decimal to float for JSON serialization
            for key in ["execution_price", "execution_quantity", "execution_fees", "signal_price"]:
                if isinstance(result.get(key), Decimal):
                    result[key] = float(result[key])
            logger.debug("Execution event created", signal_id=signal_id, strategy_id=strategy_id)
            return result
        except asyncpg.UniqueViolationError as e:
            logger.warning("Execution event already exists", signal_id=signal_id, error=str(e))
            # Return existing record
            return await self.get_by_signal_id(signal_id)
        except Exception as e:
            logger.error("Failed to create execution event", signal_id=signal_id, error=str(e), exc_info=True)
            raise

    async def get_by_id(self, event_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get execution event by ID.

        Args:
            event_id: Execution event UUID

        Returns:
            Execution event record or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE id = $1"
        record = await self._fetchrow(query, event_id)
        if record:
            result = self._record_to_dict(record)
            # Convert Decimal to float
            for key in ["execution_price", "execution_quantity", "execution_fees", "signal_price"]:
                if isinstance(result.get(key), Decimal):
                    result[key] = float(result[key])
            return result
        return None

    async def get_by_signal_id(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get execution event by signal ID.

        Args:
            signal_id: Trading signal identifier

        Returns:
            Execution event record or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE signal_id = $1 ORDER BY executed_at DESC LIMIT 1"
        signal_uuid = UUID(signal_id) if isinstance(signal_id, str) else signal_id
        record = await self._fetchrow(query, signal_uuid)
        if record:
            result = self._record_to_dict(record)
            # Convert Decimal to float
            for key in ["execution_price", "execution_quantity", "execution_fees", "signal_price"]:
                if isinstance(result.get(key), Decimal):
                    result[key] = float(result[key])
            return result
        return None

    async def list_by_strategy(
        self,
        strategy_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List execution events for a strategy.

        Args:
            strategy_id: Trading strategy identifier
            start_time: Start time filter (optional)
            end_time: End time filter (optional)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of execution event records
        """
        conditions = ["strategy_id = $1"]
        params = [strategy_id]
        param_index = 2

        if start_time:
            conditions.append(f"executed_at >= ${param_index}")
            params.append(start_time)
            param_index += 1

        if end_time:
            conditions.append(f"executed_at <= ${param_index}")
            params.append(end_time)
            param_index += 1

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {' AND '.join(conditions)}
            ORDER BY executed_at DESC
        """
        if limit:
            query += f" LIMIT ${param_index} OFFSET ${param_index + 1}"
            params.extend([limit, offset])
        else:
            query += f" OFFSET ${param_index}"
            params.append(offset)

        records = await self._fetch(query, *params)
        results = self._records_to_dicts(records)
        # Convert Decimal to float
        for result in results:
            for key in ["execution_price", "execution_quantity", "execution_fees", "signal_price"]:
                if isinstance(result.get(key), Decimal):
                    result[key] = float(result[key])
        return results

    async def mark_as_used_for_training(
        self,
        event_ids: Sequence[UUID] | Sequence[str],
        training_id: UUID | str,
    ) -> int:
        """
        Mark execution events as used for training.

        Args:
            event_ids: Sequence of execution event UUIDs
            training_id: Model version UUID associated with this training run

        Returns:
            Number of rows updated
        """
        if not event_ids:
            return 0

        # Normalize to UUID list
        uuid_ids = [UUID(str(eid)) for eid in event_ids]

        query = f"""
            UPDATE {self.table_name}
            SET used_for_training = TRUE,
                training_id = $2
            WHERE id = ANY($1::uuid[])
        """
        try:
            result = await self._execute(
                query,
                uuid_ids,
                UUID(str(training_id)) if isinstance(training_id, str) else training_id,
            )
            # _execute returns string like "UPDATE 0", extract the number
            count = int(result.split()[-1]) if result else 0
            return count
        except Exception as e:
            logger.error(
                "Failed to mark execution events as used for training",
                error=str(e),
                event_count=len(event_ids),
                exc_info=True,
            )
            raise

    async def get_unused_events(
        self,
        strategy_id: str,
        limit: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get execution events that have not yet been used for training.

        Args:
            strategy_id: Trading strategy identifier
            limit: Maximum number of events to return
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of unused execution event records
        """
        conditions = ["strategy_id = $1", "used_for_training = FALSE"]
        params: List[Any] = [strategy_id]
        param_index = 2

        if start_time:
            conditions.append(f"executed_at >= ${param_index}")
            params.append(start_time)
            param_index += 1

        if end_time:
            conditions.append(f"executed_at <= ${param_index}")
            params.append(end_time)
            param_index += 1

        query = f"""
            SELECT *
            FROM {self.table_name}
            WHERE {' AND '.join(conditions)}
            ORDER BY executed_at ASC
        """

        if limit is not None:
            query += f" LIMIT ${param_index}"
            params.append(limit)

        records = await self._fetch(query, *params)
        results = self._records_to_dicts(records)

        # Convert Decimal to float for numeric fields
        for result in results:
            for key in ["execution_price", "execution_quantity", "execution_fees", "signal_price"]:
                if isinstance(result.get(key), Decimal):
                    result[key] = float(result[key])

        return results

    async def get_unused_events_count(self, strategy_id: str) -> int:
        """
        Get count of execution events that have not yet been used for training.

        Args:
            strategy_id: Trading strategy identifier

        Returns:
            Count of unused events
        """
        query = f"""
            SELECT COUNT(*) AS count
            FROM {self.table_name}
            WHERE strategy_id = $1
              AND used_for_training = FALSE
        """
        record = await self._fetchrow(query, strategy_id)
        if not record:
            return 0
        # asyncpg returns numeric as Decimal; cast to int safely
        return int(record["count"])

