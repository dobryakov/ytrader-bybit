"""
Model cleanup policy service.

Manages model file cleanup, keeping last N versions,
archiving old versions, and managing disk space.
"""

from typing import Optional, List
from pathlib import Path
from datetime import datetime, timedelta

from ..database.repositories.model_version_repo import ModelVersionRepository
from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)


class ModelCleanupService:
    """Service for managing model cleanup policies."""

    def __init__(self):
        """Initialize model cleanup service."""
        self.model_version_repo = ModelVersionRepository()
        self.storage_path = Path(settings.model_storage_path)
        self.max_versions_to_keep = getattr(settings, "model_max_versions_to_keep", 100)
        self.archive_older_than_days = getattr(settings, "model_archive_older_than_days", 90)

    async def cleanup_old_versions(
        self,
        strategy_id: Optional[str] = None,
        keep_count: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Clean up old model versions, keeping only the most recent N versions.

        Args:
            strategy_id: Trading strategy identifier (None for all strategies)
            keep_count: Number of versions to keep (default: max_versions_to_keep)

        Returns:
            Dictionary with cleanup statistics
        """
        if keep_count is None:
            keep_count = self.max_versions_to_keep

        try:
            # Get all versions for the strategy, ordered by trained_at DESC
            all_versions = await self.model_version_repo.list_by_strategy(
                strategy_id=strategy_id,
                limit=None,
                offset=0,
                order_by="trained_at DESC",
            )

            if len(all_versions) <= keep_count:
                logger.info("No cleanup needed", strategy_id=strategy_id, total_versions=len(all_versions), keep_count=keep_count)
                return {
                    "total_versions": len(all_versions),
                    "kept_versions": len(all_versions),
                    "deleted_versions": 0,
                    "deleted_files": 0,
                }

            # Keep the most recent N versions (including active ones)
            versions_to_keep = all_versions[:keep_count]
            versions_to_delete = all_versions[keep_count:]

            # Never delete active versions
            versions_to_delete = [v for v in versions_to_delete if not v.get("is_active", False)]

            deleted_count = 0
            deleted_files_count = 0

            for version in versions_to_delete:
                try:
                    # Delete model file
                    file_path = Path(version["file_path"])
                    if file_path.exists():
                        file_path.unlink()
                        deleted_files_count += 1
                        logger.debug("Deleted model file", file_path=str(file_path), version=version["version"])

                    # Delete database record (cascade will delete quality metrics)
                    await self.model_version_repo.delete(version["id"])
                    deleted_count += 1

                    logger.info("Deleted old model version", version=version["version"], strategy_id=strategy_id)

                except Exception as e:
                    logger.error("Failed to delete model version", version=version["version"], error=str(e), exc_info=True)

            logger.info(
                "Model cleanup completed",
                strategy_id=strategy_id,
                total_versions=len(all_versions),
                kept_versions=len(versions_to_keep),
                deleted_versions=deleted_count,
                deleted_files=deleted_files_count,
            )

            return {
                "total_versions": len(all_versions),
                "kept_versions": len(versions_to_keep),
                "deleted_versions": deleted_count,
                "deleted_files": deleted_files_count,
            }

        except Exception as e:
            logger.error("Failed to cleanup old versions", strategy_id=strategy_id, error=str(e), exc_info=True)
            raise

    async def archive_old_versions(
        self,
        strategy_id: Optional[str] = None,
        older_than_days: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Archive model versions older than specified days.

        Args:
            strategy_id: Trading strategy identifier (None for all strategies)
            older_than_days: Archive versions older than this many days (default: archive_older_than_days)

        Returns:
            Dictionary with archive statistics
        """
        if older_than_days is None:
            older_than_days = self.archive_older_than_days

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)

            # Get old versions
            all_versions = await self.model_version_repo.list_by_strategy(
                strategy_id=strategy_id,
                limit=None,
                offset=0,
                order_by="trained_at ASC",
            )

            old_versions = [
                v
                for v in all_versions
                if v.get("trained_at") and datetime.fromisoformat(v["trained_at"].replace("Z", "+00:00")) < cutoff_date and not v.get("is_active", False)
            ]

            archived_count = 0
            archived_files_count = 0

            archive_dir = self.storage_path / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)

            for version in old_versions:
                try:
                    # Move model file to archive directory
                    file_path = Path(version["file_path"])
                    if file_path.exists():
                        archive_file_path = archive_dir / file_path.name
                        file_path.rename(archive_file_path)
                        archived_files_count += 1

                        # Update database record with new file path
                        await self.model_version_repo.update(version["id"], file_path=str(archive_file_path))

                        logger.debug("Archived model file", file_path=str(file_path), archive_path=str(archive_file_path), version=version["version"])

                    archived_count += 1
                    logger.info("Archived old model version", version=version["version"], strategy_id=strategy_id)

                except Exception as e:
                    logger.error("Failed to archive model version", version=version["version"], error=str(e), exc_info=True)

            logger.info(
                "Model archiving completed",
                strategy_id=strategy_id,
                older_than_days=older_than_days,
                archived_versions=archived_count,
                archived_files=archived_files_count,
            )

            return {
                "archived_versions": archived_count,
                "archived_files": archived_files_count,
            }

        except Exception as e:
            logger.error("Failed to archive old versions", strategy_id=strategy_id, error=str(e), exc_info=True)
            raise

    async def get_disk_usage(self) -> Dict[str, Any]:
        """
        Get disk usage statistics for model storage.

        Returns:
            Dictionary with disk usage information
        """
        try:
            total_size = 0
            file_count = 0

            if self.storage_path.exists():
                for file_path in self.storage_path.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        file_count += 1

            archive_size = 0
            archive_file_count = 0
            archive_dir = self.storage_path / "archive"
            if archive_dir.exists():
                for file_path in archive_dir.rglob("*"):
                    if file_path.is_file():
                        archive_size += file_path.stat().st_size
                        archive_file_count += 1

            return {
                "storage_path": str(self.storage_path),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_count": file_count,
                "archive_size_bytes": archive_size,
                "archive_size_mb": round(archive_size / (1024 * 1024), 2),
                "archive_file_count": archive_file_count,
            }

        except Exception as e:
            logger.error("Failed to get disk usage", error=str(e), exc_info=True)
            raise


# Global model cleanup service instance
model_cleanup_service = ModelCleanupService()

