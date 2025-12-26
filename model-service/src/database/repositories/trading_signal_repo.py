"""
Trading Signal database repository.

Provides CRUD operations for trading_signals table.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from uuid import UUID
import asyncpg
from decimal import Decimal
import json

from ..base import BaseRepository
from ...config.logging import get_logger

logger = get_logger(__name__)


class TradingSignalRepository(BaseRepository[Dict[str, Any]]):
    """Repository for trading_signals table operations."""

    @property
    def table_name(self) -> str:
        """Return the table name."""
        return "trading_signals"

    async def create(
        self,
        signal_id: str,
        strategy_id: str,
        asset: str,
        side: str,
        price: float,
        confidence: Optional[float],
        timestamp: datetime,
        model_version: Optional[str] = None,
        is_warmup: bool = False,
        market_data_snapshot: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        prediction_horizon_seconds: Optional[int] = None,
        target_timestamp: Optional[datetime] = None,
        is_rejected: bool = False,
        rejection_reason: Optional[str] = None,
        effective_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Create a new trading signal.

        Args:
            signal_id: Unique signal identifier
            strategy_id: Trading strategy identifier
            asset: Asset identifier (trading pair)
            side: Trading signal side ('buy' or 'sell')
            price: Signal price
            confidence: Confidence score (0-1, optional)
            timestamp: Signal generation timestamp
            model_version: Model version used (optional)
            is_warmup: Whether signal generated in warm-up mode
            market_data_snapshot: Market data at signal generation time (optional)
            metadata: Additional signal metadata (optional)
            trace_id: Trace ID for request flow tracking (optional)

        Returns:
            Created trading signal record

        Raises:
            DatabaseQueryError: If creation fails
        """
        query = f"""
            INSERT INTO {self.table_name} (
                signal_id, strategy_id, asset, side, price, confidence,
                timestamp, model_version, is_warmup, market_data_snapshot,
                metadata, trace_id, prediction_horizon_seconds, target_timestamp,
                is_rejected, rejection_reason, effective_threshold
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            RETURNING *
        """
        try:
            # Convert dict to JSON string for JSONB columns
            market_data_json = json.dumps(market_data_snapshot) if market_data_snapshot else None
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Normalize all datetime objects to timezone-aware UTC, then to naive for PostgreSQL timestamp without time zone
            # This prevents asyncpg errors when comparing datetime objects
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = timestamp.astimezone(timezone.utc)
            timestamp = timestamp.replace(tzinfo=None)  # Convert to naive for PostgreSQL
            
            if target_timestamp is not None:
                if target_timestamp.tzinfo is None:
                    target_timestamp = target_timestamp.replace(tzinfo=timezone.utc)
                else:
                    target_timestamp = target_timestamp.astimezone(timezone.utc)
                target_timestamp = target_timestamp.replace(tzinfo=None)  # Convert to naive for PostgreSQL
            
            record = await self._fetchrow(
                query,
                UUID(signal_id) if isinstance(signal_id, str) else signal_id,
                strategy_id,
                asset,
                side,
                Decimal(str(price)),
                Decimal(str(confidence)) if confidence is not None else None,
                timestamp,
                model_version,
                is_warmup,
                market_data_json,
                metadata_json,
                trace_id,
                prediction_horizon_seconds,
                target_timestamp,
                is_rejected,
                rejection_reason,
                Decimal(str(effective_threshold)) if effective_threshold is not None else None,
            )
            if not record:
                raise ValueError("Failed to create trading signal")
            result = self._record_to_dict(record)
            # Convert Decimal to float for JSON serialization
            for key in ["price", "confidence", "effective_threshold"]:
                if isinstance(result.get(key), Decimal):
                    result[key] = float(result[key])
            logger.debug("Trading signal created", signal_id=signal_id, strategy_id=strategy_id, asset=asset)
            return result
        except asyncpg.UniqueViolationError as e:
            logger.warning("Trading signal already exists", signal_id=signal_id, error=str(e))
            # Return existing record
            return await self.get_by_signal_id(signal_id)
        except Exception as e:
            logger.error("Failed to create trading signal", signal_id=signal_id, error=str(e), exc_info=True)
            raise

    async def get_by_id(self, signal_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get trading signal by ID.

        Args:
            signal_id: Trading signal UUID

        Returns:
            Trading signal record or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE id = $1"
        record = await self._fetchrow(query, signal_id)
        if record:
            result = self._record_to_dict(record)
            # Convert Decimal to float
            for key in ["price", "confidence"]:
                if isinstance(result.get(key), Decimal):
                    result[key] = float(result[key])
            return result
        return None

    async def get_by_signal_id(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trading signal by signal_id.

        Args:
            signal_id: Trading signal identifier

        Returns:
            Trading signal record or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE signal_id = $1 ORDER BY timestamp DESC LIMIT 1"
        signal_uuid = UUID(signal_id) if isinstance(signal_id, str) else signal_id
        record = await self._fetchrow(query, signal_uuid)
        if record:
            result = self._record_to_dict(record)
            # Convert Decimal to float
            for key in ["price", "confidence", "effective_threshold"]:
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
        List trading signals for a strategy.

        Args:
            strategy_id: Trading strategy identifier
            start_time: Start time filter (optional)
            end_time: End time filter (optional)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of trading signal records
        """
        conditions = ["strategy_id = $1"]
        params = [strategy_id]
        param_index = 2

        if start_time:
            conditions.append(f"timestamp >= ${param_index}")
            params.append(start_time)
            param_index += 1

        if end_time:
            conditions.append(f"timestamp <= ${param_index}")
            params.append(end_time)
            param_index += 1

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {' AND '.join(conditions)}
            ORDER BY timestamp DESC
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
            for key in ["price", "confidence", "effective_threshold"]:
                if isinstance(result.get(key), Decimal):
                    result[key] = float(result[key])
        return results

    async def list_by_asset(
        self,
        asset: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List trading signals for an asset.

        Args:
            asset: Asset identifier (trading pair)
            start_time: Start time filter (optional)
            end_time: End time filter (optional)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of trading signal records
        """
        conditions = ["asset = $1"]
        params = [asset]
        param_index = 2

        if start_time:
            conditions.append(f"timestamp >= ${param_index}")
            params.append(start_time)
            param_index += 1

        if end_time:
            conditions.append(f"timestamp <= ${param_index}")
            params.append(end_time)
            param_index += 1

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {' AND '.join(conditions)}
            ORDER BY timestamp DESC
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
            for key in ["price", "confidence", "effective_threshold"]:
                if isinstance(result.get(key), Decimal):
                    result[key] = float(result[key])
        return results

