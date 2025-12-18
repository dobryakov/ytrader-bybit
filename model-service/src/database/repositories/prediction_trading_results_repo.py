"""
Prediction Trading Results database repository.

Provides CRUD operations for prediction_trading_results table.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
import asyncpg
from decimal import Decimal

from ..base import BaseRepository
from ...config.logging import get_logger

logger = get_logger(__name__)


class PredictionTradingResultsRepository(BaseRepository[Dict[str, Any]]):
    """Repository for prediction_trading_results table operations."""

    @property
    def table_name(self) -> str:
        """Return the table name."""
        return "prediction_trading_results"

    async def create(
        self,
        prediction_target_id: UUID,
        signal_id: str,
        entry_signal_id: Optional[str] = None,
        position_id: Optional[UUID] = None,
        entry_price: Optional[Decimal] = None,
        entry_timestamp: Optional[datetime] = None,
        position_size_at_entry: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """
        Create a new prediction trading result.

        Args:
            prediction_target_id: Prediction target UUID
            signal_id: Trading signal UUID
            entry_signal_id: Entry signal UUID (optional)
            position_id: Position UUID (optional)
            entry_price: Entry price (optional)
            entry_timestamp: Entry timestamp (optional)
            position_size_at_entry: Position size at entry (optional)

        Returns:
            Created prediction trading result record

        Raises:
            DatabaseQueryError: If creation fails
        """
        query = f"""
            INSERT INTO {self.table_name} (
                prediction_target_id, signal_id, entry_signal_id,
                position_id, entry_price, entry_timestamp,
                position_size_at_entry, realized_pnl, unrealized_pnl, total_pnl
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING *
        """
        try:
            signal_uuid = UUID(signal_id) if isinstance(signal_id, str) else signal_id
            entry_signal_uuid = UUID(entry_signal_id) if entry_signal_id and isinstance(entry_signal_id, str) else entry_signal_id
            
            record = await self._fetchrow(
                query,
                prediction_target_id,
                signal_uuid,
                entry_signal_uuid,
                position_id,
                str(entry_price) if entry_price is not None else None,
                entry_timestamp,
                str(position_size_at_entry) if position_size_at_entry is not None else None,
                Decimal("0"),
                Decimal("0"),
                Decimal("0"),
            )
            if not record:
                raise ValueError("Failed to create prediction trading result")
            result = self._record_to_dict(record)
            logger.info(
                "Prediction trading result created",
                prediction_target_id=str(prediction_target_id),
                signal_id=signal_id,
            )
            return result
        except Exception as e:
            logger.error(
                "Failed to create prediction trading result",
                prediction_target_id=str(prediction_target_id),
                signal_id=signal_id,
                error=str(e),
                exc_info=True,
            )
            raise

    async def get_by_id(self, result_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get prediction trading result by ID.

        Args:
            result_id: Prediction trading result UUID

        Returns:
            Prediction trading result record or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE id = $1"
        record = await self._fetchrow(query, result_id)
        return self._record_to_dict(record) if record else None

    async def get_by_prediction_target_id(
        self,
        prediction_target_id: UUID,
    ) -> Optional[Dict[str, Any]]:
        """
        Get prediction trading result by prediction_target_id.

        Args:
            prediction_target_id: Prediction target UUID

        Returns:
            Prediction trading result record or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE prediction_target_id = $1 ORDER BY computed_at DESC LIMIT 1"
        record = await self._fetchrow(query, prediction_target_id)
        return self._record_to_dict(record) if record else None

    async def get_by_signal_id(
        self,
        signal_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get prediction trading results by signal_id.

        Args:
            signal_id: Trading signal identifier

        Returns:
            List of prediction trading result records
        """
        query = f"SELECT * FROM {self.table_name} WHERE signal_id = $1 ORDER BY computed_at DESC"
        signal_uuid = UUID(signal_id) if isinstance(signal_id, str) else signal_id
        records = await self._fetch(query, signal_uuid)
        return self._records_to_dicts(records)

    async def get_open_positions(
        self,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get open positions for predictions.

        Args:
            limit: Maximum number of results

        Returns:
            List of open prediction trading result records
        """
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE is_closed = false
            ORDER BY entry_timestamp ASC
        """
        if limit:
            query += f" LIMIT ${1}"
            records = await self._fetch(query, limit)
        else:
            records = await self._fetch(query)
        return self._records_to_dicts(records)

    async def update(
        self,
        result_id: UUID,
        realized_pnl: Optional[Decimal] = None,
        unrealized_pnl: Optional[Decimal] = None,
        total_pnl: Optional[Decimal] = None,
        fees: Optional[Decimal] = None,
        exit_price: Optional[Decimal] = None,
        exit_signal_id: Optional[str] = None,
        exit_timestamp: Optional[datetime] = None,
        position_size_at_exit: Optional[Decimal] = None,
        is_closed: Optional[bool] = None,
        is_partial_close: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update prediction trading result.

        Args:
            result_id: Prediction trading result UUID
            realized_pnl: Realized PnL (optional)
            unrealized_pnl: Unrealized PnL (optional)
            total_pnl: Total PnL (optional)
            fees: Fees (optional)
            exit_price: Exit price (optional)
            exit_signal_id: Exit signal UUID (optional)
            exit_timestamp: Exit timestamp (optional)
            position_size_at_exit: Position size at exit (optional)
            is_closed: Whether position is closed (optional)
            is_partial_close: Whether this is a partial close (optional)

        Returns:
            Updated prediction trading result record or None if not found
        """
        updates = []
        values = []
        param_index = 1

        if realized_pnl is not None:
            updates.append(f"realized_pnl = ${param_index}")
            values.append(str(realized_pnl))
            param_index += 1

        if unrealized_pnl is not None:
            updates.append(f"unrealized_pnl = ${param_index}")
            values.append(str(unrealized_pnl))
            param_index += 1

        if total_pnl is not None:
            updates.append(f"total_pnl = ${param_index}")
            values.append(str(total_pnl))
            param_index += 1

        if fees is not None:
            updates.append(f"fees = ${param_index}")
            values.append(str(fees))
            param_index += 1

        if exit_price is not None:
            updates.append(f"exit_price = ${param_index}")
            values.append(str(exit_price))
            param_index += 1

        if exit_signal_id is not None:
            exit_signal_uuid = UUID(exit_signal_id) if isinstance(exit_signal_id, str) else exit_signal_id
            updates.append(f"exit_signal_id = ${param_index}")
            values.append(exit_signal_uuid)
            param_index += 1

        if exit_timestamp is not None:
            updates.append(f"exit_timestamp = ${param_index}")
            values.append(exit_timestamp)
            param_index += 1

        if position_size_at_exit is not None:
            updates.append(f"position_size_at_exit = ${param_index}")
            values.append(str(position_size_at_exit))
            param_index += 1

        if is_closed is not None:
            updates.append(f"is_closed = ${param_index}")
            values.append(is_closed)
            param_index += 1

        if is_partial_close is not None:
            updates.append(f"is_partial_close = ${param_index}")
            values.append(is_partial_close)
            param_index += 1

        if not updates:
            return await self.get_by_id(result_id)

        updates.append(f"updated_at = ${param_index}")
        values.append(datetime.utcnow())
        param_index += 1

        values.append(result_id)

        query = f"""
            UPDATE {self.table_name}
            SET {', '.join(updates)}
            WHERE id = ${param_index}
            RETURNING *
        """
        try:
            record = await self._fetchrow(query, *values)
            if record:
                result = self._record_to_dict(record)
                logger.info(
                    "Prediction trading result updated",
                    result_id=str(result_id),
                    is_closed=is_closed,
                )
                return result
            return None
        except Exception as e:
            logger.error(
                "Failed to update prediction trading result",
                result_id=str(result_id),
                error=str(e),
                exc_info=True,
            )
            raise

    async def aggregate_pnl_by_signal(
        self,
        signal_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Aggregate PnL metrics by signal_id.

        Args:
            signal_id: Trading signal identifier

        Returns:
            Aggregated PnL metrics or None if not found
        """
        query = f"""
            SELECT 
                SUM(realized_pnl) as total_realized_pnl,
                SUM(unrealized_pnl) as total_unrealized_pnl,
                SUM(total_pnl) as total_pnl,
                SUM(fees) as total_fees,
                COUNT(*) as total_records,
                COUNT(CASE WHEN is_closed THEN 1 END) as closed_count
            FROM {self.table_name}
            WHERE signal_id = $1
        """
        signal_uuid = UUID(signal_id) if isinstance(signal_id, str) else signal_id
        record = await self._fetchrow(query, signal_uuid)
        if record:
            return self._record_to_dict(record)
        return None

