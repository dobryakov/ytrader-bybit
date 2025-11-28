"""
Model version history service.

Provides functionality to query and manage version history,
supporting rollback operations.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from ..database.repositories.model_version_repo import ModelVersionRepository
from ..services.model_version_manager import model_version_manager
from ..config.logging import get_logger

logger = get_logger(__name__)


class VersionHistoryService:
    """Service for managing model version history."""

    def __init__(self):
        """Initialize version history service."""
        self.model_version_repo = ModelVersionRepository()

    async def get_version_history(
        self,
        strategy_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Get version history for a strategy.

        Args:
            strategy_id: Trading strategy identifier (None for all strategies)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of model version records ordered by trained_at DESC
        """
        versions = await self.model_version_repo.list_by_strategy(
            strategy_id=strategy_id,
            limit=limit,
            offset=offset,
            order_by="trained_at DESC",
        )

        logger.info("Retrieved version history", strategy_id=strategy_id, count=len(versions))

        return versions

    async def get_version_by_id(self, model_version_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get a specific model version by ID.

        Args:
            model_version_id: Model version UUID

        Returns:
            Model version record or None if not found
        """
        version = await self.model_version_repo.get_by_id(model_version_id)

        if version:
            logger.info("Retrieved model version", model_version_id=str(model_version_id))
        else:
            logger.warning("Model version not found", model_version_id=str(model_version_id))

        return version

    async def rollback_to_version(
        self,
        model_version_id: UUID,
        strategy_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Rollback to a previous model version.

        Args:
            model_version_id: Model version UUID to rollback to
            strategy_id: Trading strategy identifier (optional, inferred from model version)

        Returns:
            Activated model version record or None if not found
        """
        # Use model_version_manager's rollback functionality
        rolled_back_version = await model_version_manager.rollback_to_version(model_version_id)

        if rolled_back_version:
            logger.info("Rolled back to model version", model_version_id=str(model_version_id), strategy_id=strategy_id)
        else:
            logger.error("Failed to rollback to model version", model_version_id=str(model_version_id))

        return rolled_back_version

    async def get_active_version(self, strategy_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get active model version for a strategy.

        Args:
            strategy_id: Trading strategy identifier

        Returns:
            Active model version record or None if not found
        """
        active_version = await self.model_version_repo.get_active_by_strategy(strategy_id)

        if active_version:
            logger.info("Retrieved active model version", strategy_id=strategy_id, version=active_version.get("version"))
        else:
            logger.debug("No active model version found", strategy_id=strategy_id)

        return active_version

    async def get_previous_versions(
        self,
        model_version_id: UUID,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get previous model versions (versions trained before the specified version).

        Args:
            model_version_id: Current model version UUID
            limit: Maximum number of previous versions to return

        Returns:
            List of previous model version records
        """
        current_version = await self.model_version_repo.get_by_id(model_version_id)

        if not current_version:
            logger.warning("Model version not found", model_version_id=str(model_version_id))
            return []

        strategy_id = current_version.get("strategy_id")
        trained_at = current_version.get("trained_at")

        # Get versions trained before this one
        query = """
            SELECT * FROM model_versions
            WHERE strategy_id = $1
                AND trained_at < $2
            ORDER BY trained_at DESC
            LIMIT $3
        """
        records = await self.model_version_repo._fetch(query, strategy_id, trained_at, limit)

        previous_versions = [self.model_version_repo._record_to_dict(record) for record in records]

        logger.info(
            "Retrieved previous model versions",
            model_version_id=str(model_version_id),
            count=len(previous_versions),
        )

        return previous_versions


# Global version history service instance
version_history_service = VersionHistoryService()

