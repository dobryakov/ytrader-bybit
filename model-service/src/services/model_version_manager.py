"""
Model version manager.

Creates model versions, stores model files in /models/v{version}/,
updates database metadata, and handles version activation.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from uuid import UUID
import json

from ..database.repositories.model_version_repo import ModelVersionRepository
from ..database.repositories.quality_metrics_repo import ModelQualityMetricsRepository
from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)


class ModelVersionManager:
    """Manages model versions and files."""

    def __init__(self):
        """Initialize model version manager."""
        self.model_version_repo = ModelVersionRepository()
        self.quality_metrics_repo = ModelQualityMetricsRepository()
        self.storage_path = Path(settings.model_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def create_version(
        self,
        version: str,
        model_type: str,
        file_path: str,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        training_duration_seconds: Optional[int] = None,
        training_dataset_size: Optional[int] = None,
        training_config: Optional[Dict[str, Any]] = None,
        is_active: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a new model version.

        Args:
            version: Version identifier (e.g., 'v1', 'v2.1')
            model_type: Model type ('xgboost', 'random_forest', etc.)
            file_path: Path to model file (relative to storage_path or absolute)
            strategy_id: Trading strategy identifier
            symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT') - optional for universal models
            training_duration_seconds: Training duration in seconds
            training_dataset_size: Number of records in training dataset
            training_config: Training configuration parameters
            is_active: Whether to activate this version

        Returns:
            Created model version record
        """
        # Ensure file path is absolute
        if not Path(file_path).is_absolute():
            file_path = str(self.storage_path / file_path)

        # Deactivate previous active model if activating this one
        if is_active:
            await self.model_version_repo.deactivate_all_for_strategy_and_symbol(strategy_id, symbol)

        # Create model version record
        model_version = await self.model_version_repo.create(
            version=version,
            file_path=file_path,
            model_type=model_type,
            strategy_id=strategy_id,
            symbol=symbol,
            training_duration_seconds=training_duration_seconds,
            training_dataset_size=training_dataset_size,
            training_config=training_config,
            is_active=is_active,
        )

        logger.info(
            "Model version created",
            version=version,
            model_version_id=model_version["id"],
            strategy_id=strategy_id,
            symbol=symbol,
            is_active=is_active,
        )

        return model_version

    async def activate_version(self, model_version_id: UUID, strategy_id: Optional[str] = None, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Activate a model version (deactivates previous active model).

        Args:
            model_version_id: Model version UUID to activate
            strategy_id: Trading strategy identifier (optional, inferred from model version)
            symbol: Trading pair symbol (optional, inferred from model version)

        Returns:
            Activated model version record or None if not found
        """
        # Get model version to infer strategy_id and symbol if not provided
        if strategy_id is None or symbol is None:
            model_version = await self.model_version_repo.get_by_id(model_version_id)
            if not model_version:
                logger.error("Model version not found", model_version_id=str(model_version_id))
                return None
            if strategy_id is None:
                strategy_id = model_version.get("strategy_id")
            if symbol is None:
                symbol = model_version.get("symbol")

        # Activate the model version
        activated_version = await self.model_version_repo.activate(model_version_id, strategy_id, symbol)

        if activated_version:
            logger.info("Model version activated", model_version_id=str(model_version_id), strategy_id=strategy_id, symbol=symbol)
        else:
            logger.error("Failed to activate model version", model_version_id=str(model_version_id))

        return activated_version

    async def deactivate_version(self, model_version_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Deactivate a model version and block automatic reactivation.

        Args:
            model_version_id: Model version UUID to deactivate

        Returns:
            Deactivated model version record or None if not found
        """
        # Deactivate and block automatic reactivation
        updated_version = await self.model_version_repo.update(
            model_version_id, 
            is_active=False,
            auto_activation_disabled=True
        )

        if updated_version:
            logger.info(
                "Model version deactivated and auto-activation blocked",
                model_version_id=str(model_version_id),
                version=updated_version.get("version")
            )
        else:
            logger.warning("Model version not found for deactivation", model_version_id=str(model_version_id))

        return updated_version

    async def save_quality_metrics(
        self,
        model_version_id: UUID,
        metrics: Dict[str, float],
        evaluation_dataset_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dataset_split: Optional[str] = None,
    ) -> None:
        """
        Save quality metrics for a model version.

        Args:
            model_version_id: Model version UUID
            metrics: Dictionary of metric names to values
            evaluation_dataset_size: Number of records in evaluation dataset
            metadata: Additional metric metadata
            dataset_split: Dataset split identifier (e.g., 'train', 'validation', 'test')
        """
        # Determine metric types based on metric names
        metric_type_mapping = {
            "accuracy": "classification",
            "balanced_accuracy": "classification",
            "precision": "classification",
            "recall": "classification",
            "f1_score": "classification",
            "roc_auc": "classification",
            "pr_auc": "classification",
            "mse": "regression",
            "mae": "regression",
            "rmse": "regression",
            "r2_score": "regression",
            "sharpe_ratio": "trading_performance",
            "profit_factor": "trading_performance",
            "win_rate": "trading_performance",
            "total_pnl": "trading_performance",
            "avg_pnl": "trading_performance",
            "max_drawdown": "trading_performance",
        }

        # Include dataset_split in metadata if provided
        final_metadata = metadata.copy() if metadata else {}
        if dataset_split:
            final_metadata["dataset_split"] = dataset_split

        for metric_name, metric_value in metrics.items():
            metric_type = metric_type_mapping.get(metric_name, "trading_performance")

            try:
                await self.quality_metrics_repo.create(
                    model_version_id=model_version_id,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    metric_type=metric_type,
                    evaluation_dataset_size=evaluation_dataset_size,
                    metadata=final_metadata if final_metadata else None,
                )
                logger.debug("Saved quality metric", metric_name=metric_name, metric_value=metric_value)
            except Exception as e:
                logger.error(
                    "Failed to save quality metric",
                    metric_name=metric_name,
                    error=str(e),
                    exc_info=True,
                )

        logger.info("Quality metrics saved", model_version_id=str(model_version_id), metric_count=len(metrics))

    async def get_active_version(self, strategy_id: Optional[str] = None, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get active model version for a strategy and symbol.

        Args:
            strategy_id: Trading strategy identifier
            symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT') - optional for universal models

        Returns:
            Active model version record or None if not found
        """
        return await self.model_version_repo.get_active_by_strategy_and_symbol(strategy_id, symbol)

    async def rollback_to_version(self, model_version_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Rollback to a previous model version.

        Args:
            model_version_id: Model version UUID to rollback to

        Returns:
            Activated model version record or None if not found
        """
        # Get model version to get strategy_id and symbol
        model_version = await self.model_version_repo.get_by_id(model_version_id)
        if not model_version:
            logger.error("Model version not found for rollback", model_version_id=str(model_version_id))
            return None

        strategy_id = model_version.get("strategy_id")
        symbol = model_version.get("symbol")

        # Activate the specified version (this will deactivate current active version)
        activated_version = await self.activate_version(model_version_id, strategy_id, symbol)

        if activated_version:
            logger.info("Rolled back to model version", model_version_id=str(model_version_id), strategy_id=strategy_id, symbol=symbol)
        else:
            logger.error("Failed to rollback to model version", model_version_id=str(model_version_id))

        return activated_version


# Global model version manager instance
model_version_manager = ModelVersionManager()

