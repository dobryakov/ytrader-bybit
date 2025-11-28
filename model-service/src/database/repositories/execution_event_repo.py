"""
Execution Event database repository.

Provides CRUD operations for execution_events table.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
import asyncpg
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
                performance,
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

