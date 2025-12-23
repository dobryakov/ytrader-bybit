"""
Model Version database repository.

Provides CRUD operations for model_versions table.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncpg
import json
from uuid import UUID

from ..base import BaseRepository
from ...config.logging import get_logger

logger = get_logger(__name__)


class ModelVersionRepository(BaseRepository[Dict[str, Any]]):
    """Repository for model_versions table operations."""

    @property
    def table_name(self) -> str:
        """Return the table name."""
        return "model_versions"

    async def create(
        self,
        version: str,
        file_path: str,
        model_type: str,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        training_duration_seconds: Optional[int] = None,
        training_dataset_size: Optional[int] = None,
        training_config: Optional[Dict[str, Any]] = None,
        is_active: bool = False,
        is_warmup_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a new model version.

        Args:
            version: Human-readable version identifier (e.g., 'v1', 'v2.1')
            file_path: File system path to model file
            model_type: Model type (e.g., 'xgboost', 'random_forest')
            strategy_id: Trading strategy identifier (optional)
            symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT') - optional for universal models
            training_duration_seconds: Training duration in seconds
            training_dataset_size: Number of records in training dataset
            training_config: Training configuration parameters
            is_active: Whether this version is currently active
            is_warmup_mode: Whether system is in warm-up mode

        Returns:
            Created model version record

        Raises:
            DatabaseQueryError: If creation fails
        """
        query = f"""
            INSERT INTO {self.table_name} (
                version, file_path, model_type, strategy_id, symbol,
                training_duration_seconds, training_dataset_size, training_config,
                is_active, is_warmup_mode
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING *
        """
        try:
            # Serialize training_config dict to JSON string for JSONB column
            training_config_json = json.dumps(training_config) if training_config else None
            
            record = await self._fetchrow(
                query,
                version,
                file_path,
                model_type,
                strategy_id,
                symbol,
                training_duration_seconds,
                training_dataset_size,
                training_config_json,
                is_active,
                is_warmup_mode,
            )
            if not record:
                raise ValueError("Failed to create model version")
            result = self._record_to_dict(record)
            logger.info("Model version created", version=version, strategy_id=strategy_id, symbol=symbol, model_type=model_type)
            return result
        except asyncpg.UniqueViolationError as e:
            logger.error("Model version already exists", version=version, error=str(e))
            raise ValueError(f"Model version {version} already exists") from e
        except Exception as e:
            logger.error("Failed to create model version", version=version, error=str(e), exc_info=True)
            raise

    async def get_by_id(self, model_version_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get model version by ID.

        Args:
            model_version_id: Model version UUID

        Returns:
            Model version record or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE id = $1"
        record = await self._fetchrow(query, model_version_id)
        return self._record_to_dict(record) if record else None

    async def get_by_version(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get model version by version string.

        Args:
            version: Version identifier (e.g., 'v1', 'v2.1')

        Returns:
            Model version record or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE version = $1"
        record = await self._fetchrow(query, version)
        return self._record_to_dict(record) if record else None

    async def get_active_by_strategy(self, strategy_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get active model version for a strategy (without symbol filter, for backward compatibility).

        Args:
            strategy_id: Trading strategy identifier (None for general models)

        Returns:
            Active model version record or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE strategy_id = $1 AND is_active = true AND symbol IS NULL"
        record = await self._fetchrow(query, strategy_id)
        return self._record_to_dict(record) if record else None

    async def has_active_models_for_strategy(self, strategy_id: Optional[str] = None) -> bool:
        """
        Check if there are any active models for a strategy (with any symbol or without symbol).

        Args:
            strategy_id: Trading strategy identifier (None for general models)

        Returns:
            True if at least one active model exists for the strategy, False otherwise
        """
        query = f"SELECT COUNT(*) FROM {self.table_name} WHERE strategy_id = $1 AND is_active = true"
        count = await self._fetchval(query, strategy_id)
        return count > 0 if count is not None else False

    async def get_active_by_strategy_and_symbol(
        self, 
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get active model version for a strategy and symbol.

        Args:
            strategy_id: Trading strategy identifier (None for general models)
            symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')
                   If None, looks for universal model (symbol IS NULL)

        Returns:
            Active model version record or None if not found
        """
        if symbol:
            query = f"SELECT * FROM {self.table_name} WHERE strategy_id = $1 AND symbol = $2 AND is_active = true"
            record = await self._fetchrow(query, strategy_id, symbol)
        else:
            # Fallback: look for universal model (symbol IS NULL) for backward compatibility
            query = f"SELECT * FROM {self.table_name} WHERE strategy_id = $1 AND symbol IS NULL AND is_active = true"
            record = await self._fetchrow(query, strategy_id)
        return self._record_to_dict(record) if record else None

    async def list_by_strategy(
        self,
        strategy_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        order_by: str = "trained_at DESC",
    ) -> List[Dict[str, Any]]:
        """
        List model versions for a strategy.

        Args:
            strategy_id: Trading strategy identifier (None for all strategies)
            limit: Maximum number of results
            offset: Number of results to skip
            order_by: ORDER BY clause (default: trained_at DESC)

        Returns:
            List of model version records
        """
        if strategy_id is not None:
            query = f"SELECT * FROM {self.table_name} WHERE strategy_id = $1 ORDER BY {order_by}"
            if limit:
                query += f" LIMIT ${2} OFFSET ${3}"
                records = await self._fetch(query, strategy_id, limit, offset)
            else:
                query += f" OFFSET ${2}"
                records = await self._fetch(query, strategy_id, offset)
        else:
            query = f"SELECT * FROM {self.table_name} ORDER BY {order_by}"
            if limit:
                query += f" LIMIT ${1} OFFSET ${2}"
                records = await self._fetch(query, limit, offset)
            else:
                query += f" OFFSET ${1}"
                records = await self._fetch(query, offset)

        return self._records_to_dicts(records)

    async def update(
        self,
        model_version_id: UUID,
        is_active: Optional[bool] = None,
        is_warmup_mode: Optional[bool] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Update model version.

        Args:
            model_version_id: Model version UUID
            is_active: Whether this version is currently active
            is_warmup_mode: Whether system is in warm-up mode
            **kwargs: Additional fields to update

        Returns:
            Updated model version record or None if not found
        """
        updates = []
        values = []
        param_index = 1

        if is_active is not None:
            updates.append(f"is_active = ${param_index}")
            values.append(is_active)
            param_index += 1

        if is_warmup_mode is not None:
            updates.append(f"is_warmup_mode = ${param_index}")
            values.append(is_warmup_mode)
            param_index += 1

        for key, value in kwargs.items():
            updates.append(f"{key} = ${param_index}")
            values.append(value)
            param_index += 1

        if not updates:
            return await self.get_by_id(model_version_id)

        updates.append(f"updated_at = ${param_index}")
        values.append(datetime.utcnow())
        param_index += 1

        values.append(model_version_id)

        query = f"""
            UPDATE {self.table_name}
            SET {', '.join(updates)}
            WHERE id = ${param_index}
            RETURNING *
        """
        record = await self._fetchrow(query, *values)
        if record:
            logger.info("Model version updated", model_version_id=str(model_version_id), updates=updates)
            return self._record_to_dict(record)
        return None

    async def deactivate_all_for_strategy(self, strategy_id: Optional[str] = None) -> int:
        """
        Deactivate all model versions for a strategy (without symbol filter, for backward compatibility).

        Args:
            strategy_id: Trading strategy identifier (None for general models)

        Returns:
            Number of deactivated models
        """
        query = f"""
            UPDATE {self.table_name}
            SET is_active = false, auto_activation_disabled = true, updated_at = $1
            WHERE strategy_id = $2 AND is_active = true AND symbol IS NULL
        """
        result = await self._execute(query, datetime.utcnow(), strategy_id)
        count = int(result.split()[-1]) if result else 0
        logger.info("Deactivated model versions", strategy_id=strategy_id, count=count)
        return count

    async def deactivate_all_for_strategy_and_symbol(
        self, 
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> int:
        """
        Deactivate all model versions for a strategy and symbol.

        Args:
            strategy_id: Trading strategy identifier (None for general models)
            symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')
                   If None, deactivates universal models (symbol IS NULL)

        Returns:
            Number of deactivated models
        """
        if symbol:
            query = f"""
                UPDATE {self.table_name}
                SET is_active = false, auto_activation_disabled = true, updated_at = $1
                WHERE strategy_id = $2 AND symbol = $3 AND is_active = true
            """
            result = await self._execute(query, datetime.utcnow(), strategy_id, symbol)
        else:
            query = f"""
                UPDATE {self.table_name}
                SET is_active = false, auto_activation_disabled = true, updated_at = $1
                WHERE strategy_id = $2 AND symbol IS NULL AND is_active = true
            """
            result = await self._execute(query, datetime.utcnow(), strategy_id)
        count = int(result.split()[-1]) if result else 0
        logger.info("Deactivated model versions", strategy_id=strategy_id, symbol=symbol, count=count)
        return count

    async def activate(self, model_version_id: UUID, strategy_id: Optional[str] = None, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Activate a model version (deactivates previous active model for the strategy and symbol).

        Args:
            model_version_id: Model version UUID to activate
            strategy_id: Trading strategy identifier (None for general models)
            symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT') - if None, uses symbol from model record

        Returns:
            Activated model version record or None if not found
        """
        async with self._transaction() as conn:
            # Get model version to determine symbol if not provided
            if symbol is None:
                model_version = await self.get_by_id(model_version_id)
                if model_version:
                    symbol = model_version.get("symbol")
            
            # Deactivate all models for this strategy and symbol combination
            # Note: We don't set auto_activation_disabled here because these models
            # are being deactivated as part of activating another model, not manually
            if symbol:
                await conn.execute(
                    f"""
                    UPDATE {self.table_name}
                    SET is_active = false, updated_at = $1
                    WHERE strategy_id = $2 AND symbol = $3 AND is_active = true
                    """,
                    datetime.utcnow(),
                    strategy_id,
                    symbol,
                )
            else:
                # Universal model (symbol IS NULL)
                await conn.execute(
                    f"""
                    UPDATE {self.table_name}
                    SET is_active = false, updated_at = $1
                    WHERE strategy_id = $2 AND symbol IS NULL AND is_active = true
                    """,
                    datetime.utcnow(),
                    strategy_id,
                )

            # Activate the specified model and remove auto-activation block
            query = f"""
                UPDATE {self.table_name}
                SET is_active = true, auto_activation_disabled = false, updated_at = $1
                WHERE id = $2
                RETURNING *
            """
            record = await conn.fetchrow(query, datetime.utcnow(), model_version_id)
            if record:
                logger.info("Model version activated", model_version_id=str(model_version_id), strategy_id=strategy_id, symbol=symbol)
                return self._record_to_dict(record)
            return None

    async def delete(self, model_version_id: UUID) -> bool:
        """
        Delete a model version.

        Args:
            model_version_id: Model version UUID

        Returns:
            True if deleted, False if not found
        """
        query = f"DELETE FROM {self.table_name} WHERE id = $1 RETURNING id"
        record = await self._fetchrow(query, model_version_id)
        if record:
            logger.info("Model version deleted", model_version_id=str(model_version_id))
            return True
        return False

